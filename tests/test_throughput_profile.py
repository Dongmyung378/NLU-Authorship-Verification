"""CPU-only control-flow fixtures; timings below are never real GPU evidence."""
import contextlib
import copy
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import weakref

import torch

from nlu_authorship import training


def synthetic_profile(microbatch, seconds, *, reserved=600, total=1000, initial='a'*64):
    return {'microbatch': microbatch, 'effective_batch': 32,
            'median_optimizer_step_seconds': seconds,
            'peak_reserved_gpu_bytes': reserved, 'gpu_total_bytes': total,
            'initial_state_sha256': initial, 'profile_weights_discarded': True,
            'gpu_name': 'CPU MOCK CONTROL-FLOW ONLY', 'optimizer_steps': 20}


class ThroughputProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root/'repo'
        (self.repo/'nlu_authorship').mkdir(parents=True)
        self.source = self.repo/'nlu_authorship/synthetic.py'
        self.source.write_text('VALUE=1\n')
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text('{}\n')
        self.plan = self.root/'plan.json'
        self.plan.write_text('{}\n')
        self.records = [{'id': 'train-'+str(i), 'text_1': 'First passage',
                         'text_2': 'Second passage', 'label': i%2} for i in range(64)]
        self.profile_calls = []

    def tearDown(self):
        self.temp.cleanup()

    def run_fixture(self, responses, *, callback=None, clock=None, output_name='profile'):
        """Exercise the real selector with isolated fake profile results only."""
        def load_records(repo, manifest, split):
            if split != 'train':
                raise AssertionError('Timing may access train only, never selection/calibration/holdout')
            return self.records

        def profile(pretrained, plan, records, encodings, tokenizer, **kwargs):
            microbatch = plan['microbatch']
            self.profile_calls.append(microbatch)
            self.assertEqual(kwargs['methods_count'], 1)
            self.assertEqual(plan['effective_batch'], 32)
            self.assertEqual(plan['profile_optimizer_steps'], 20)
            self.assertEqual(records, self.records)
            if callback is not None:
                callback(microbatch)
            response = responses[microbatch]
            if isinstance(response, Exception):
                raise response
            return copy.deepcopy(response)

        protected = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(training, 'require_kaggle_gpu'))
            stack.enter_context(patch.object(training, 'validated_plan', return_value=training.DEFAULT_PLAN.copy()))
            load = stack.enter_context(patch('nlu_authorship.data.load_records', side_effect=load_records))
            stack.enter_context(patch.object(training, '_prepare_pretrained', return_value=(self.root, {})))
            stack.enter_context(patch('transformers.AutoTokenizer.from_pretrained', return_value=object()))
            stack.enter_context(patch.object(training, '_tokenize', return_value=['cpu-fixture-ab', 'cpu-fixture-ba']))
            stack.enter_context(patch.object(training, '_profile', side_effect=profile))
            for name in ('_predict', '_new_model', '_optimizer', '_step',
                         'write_artifact_inventory', 'run_transformer_experiment'):
                protected.append(stack.enter_context(patch.object(training, name)))
            for name in ('nlu_authorship.stylometric.train_stylometric',
                         'nlu_authorship.calibration.fit_calibration',
                         'nlu_authorship.release.release_models'):
                protected.append(stack.enter_context(patch(name)))
            if clock is not None:
                stack.enter_context(patch.object(training.time, 'perf_counter', side_effect=clock))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                return training.run_throughput_profile(self.repo, self.manifest,
                                                       self.root/output_name, self.plan)
            finally:
                load.assert_called_once_with(self.repo, self.manifest, 'train')
                for function in protected:
                    function.assert_not_called()

    def test_fastest_adoption_requires_five_percent_gain(self):
        for fastest, expected in [(9.55, 2), (9., 32)]:
            with self.subTest(fastest=fastest):
                result = self.run_fixture({2: synthetic_profile(2, 10.),
                                           8: synthetic_profile(8, 9.8),
                                           32: synthetic_profile(32, fastest)},
                                          output_name='gain-'+str(fastest))
                self.assertEqual(result['selected_microbatch'], expected)
                self.assertEqual(result['speedup_vs_microbatch2'], 10./(10. if expected==2 else fastest))

    def test_exact_speedup_and_memory_boundaries_are_admissible(self):
        result = self.run_fixture({2: synthetic_profile(2, 21.),
                                   8: synthetic_profile(8, 20., reserved=900),
                                   32: synthetic_profile(32, 19., reserved=901)})
        self.assertEqual(result['selected_microbatch'], 8)
        self.assertEqual(result['speedup_vs_microbatch2'], 1.05)
        self.assertEqual(result['selected_peak_reserved_bytes'], 900)

    def test_fastest_over_memory_limit_is_excluded(self):
        result = self.run_fixture({2: synthetic_profile(2, 10.),
                                   8: synthetic_profile(8, 8., reserved=850),
                                   32: synthetic_profile(32, 1., reserved=951)})
        self.assertEqual(result['selected_microbatch'], 8)
        self.assertLessEqual(result['selected_peak_reserved_bytes'], 900)

    def test_unsafe_reference_cannot_be_retained_when_gain_gate_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'memory|admissible|margin'):
            self.run_fixture({2: synthetic_profile(2, 10., reserved=950),
                              8: synthetic_profile(8, 9.8, reserved=800),
                              32: synthetic_profile(32, 9.9, reserved=800)})
        self.assertFalse((self.root/'profile/throughput-profile.json').exists())

    def test_all_completed_candidates_outside_margin_fail_without_result(self):
        with self.assertRaisesRegex(RuntimeError, 'memory|margin'):
            self.run_fixture({2: synthetic_profile(2, 10., reserved=951),
                              8: synthetic_profile(8, 8., reserved=952),
                              32: synthetic_profile(32, 7., reserved=953)})
        self.assertFalse((self.root/'profile/throughput-profile.json').exists())

    def test_failed_oom_candidates_are_ignored_and_no_task_weights_retained(self):
        result = self.run_fixture({2: synthetic_profile(2, 10.),
                                   8: synthetic_profile(4, 8.),
                                   32: RuntimeError('Both attempts ran out of memory')})
        self.assertEqual(result['selected_microbatch'], 4)
        self.assertEqual(result['candidates'][-1]['status'], 'failed_oom')
        self.assertTrue(result['profile_weights_discarded'])
        self.assertFalse(result['retained_task_models_trained'])
        for key in ('selection_inference_performed', 'calibration_evaluated', 'locked_holdout_evaluated'):
            self.assertFalse(result[key])
        self.assertEqual({p.name for p in (self.root/'profile').iterdir()}, {'throughput-profile.json'})

    def test_initial_state_mismatch_rejects_before_selection(self):
        with self.assertRaisesRegex(ValueError, 'initial'):
            self.run_fixture({2: synthetic_profile(2, 10.),
                              8: synthetic_profile(8, 8., initial='b'*64),
                              32: synthetic_profile(32, 7.)})
        self.assertEqual(self.profile_calls, [2, 8])
        self.assertFalse((self.root/'profile/throughput-profile.json').exists())

    def test_changed_source_manifest_or_plan_rejects_result(self):
        for target, name in [(self.source, 'source'), (self.manifest, 'manifest'), (self.plan, 'plan')]:
            with self.subTest(target=name):
                def change(microbatch):
                    if microbatch == 2:
                        target.write_text(target.read_text()+'\n')
                with self.assertRaisesRegex(RuntimeError, 'changed'):
                    self.run_fixture({2: synthetic_profile(2, 10.),
                                      8: synthetic_profile(8, 8.),
                                      32: synthetic_profile(32, 7.)}, callback=change,
                                     output_name='changed-'+name)
                self.assertFalse((self.root/('changed-'+name)/'throughput-profile.json').exists())

    def test_no_further_candidate_starts_after_timing_budget(self):
        result = self.run_fixture({2: synthetic_profile(2, 10.),
                                   8: synthetic_profile(8, 8.),
                                   32: synthetic_profile(32, 7.)}, clock=[0., 0., 1200., 1200.])
        self.assertEqual(self.profile_calls, [2])
        self.assertEqual(result['candidates'][-1]['status'], 'not_run_budget')
        self.assertEqual(result['selected_microbatch'], 2)

    def test_non_oom_profile_failure_is_not_silently_ignored(self):
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            self.run_fixture({2: synthetic_profile(2, 10.),
                              8: RuntimeError('nonfinite classification loss'),
                              32: synthetic_profile(32, 7.)})
        self.assertEqual(self.profile_calls, [2, 8])
        self.assertFalse((self.root/'profile/throughput-profile.json').exists())

    def test_failed_required_reference_has_explicit_failure_without_selection(self):
        with self.assertRaisesRegex(RuntimeError, 'reference'):
            self.run_fixture({2: RuntimeError('Both attempts ran out of memory'),
                              8: synthetic_profile(8, 8.),
                              32: synthetic_profile(32, 7.)})
        self.assertFalse((self.root/'profile/throughput-profile.json').exists())

    def test_candidate_timeout_keeps_only_completed_admissible_reference(self):
        result = self.run_fixture({2: synthetic_profile(2, 10.),
                                   8: TimeoutError('Synthetic timing deadline'),
                                   32: synthetic_profile(32, 7.)})
        self.assertEqual(self.profile_calls, [2, 8])
        self.assertEqual(result['selected_microbatch'], 2)
        self.assertEqual(result['candidates'][-1]['status'], 'failed_time_budget')
        self.assertFalse(result['all_declared_candidates_completed'])

    def test_reference_timeout_is_failure_not_successful_timing_evidence(self):
        with self.assertRaisesRegex(RuntimeError, 'reference'):
            self.run_fixture({2: TimeoutError('Synthetic timing deadline'),
                              8: synthetic_profile(8, 8.),
                              32: synthetic_profile(32, 7.)})
        self.assertEqual(self.profile_calls, [2])
        self.assertFalse((self.root/'profile/throughput-profile.json').exists())

    def test_actual_profile_helper_deadline_releases_tiny_cpu_model(self):
        references, clock = [], {'now': 0.}

        def new_model(*args):
            model = torch.nn.Linear(1, 2)
            references.append(weakref.ref(model))
            return model, 'a'*64

        def step(*args):
            clock['now'] = 2.
            return 1., len(args[5]), False

        plan = training.DEFAULT_PLAN.copy()
        plan['profile_optimizer_steps'] = 2
        with patch.object(training, '_new_model', new=new_model), \
                patch.object(training, '_optimizer', new=lambda *args: object()), \
                patch.object(training, '_step', new=step), \
                patch.object(training.time, 'perf_counter', new=lambda: clock['now']), \
                patch('torch.amp.GradScaler', new=lambda *args: object()), \
                patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.synchronize'), \
                patch('torch.cuda.empty_cache') as cleanup:
            with self.assertRaisesRegex(TimeoutError, 'deadline'):
                training._profile(self.root, plan, self.records, [], object(), methods_count=1, deadline=1.)
            cleanup.assert_called_once()
        gc.collect()
        self.assertEqual(len(references), 1)
        self.assertIsNone(references[0]())

    def test_oom_retry_releases_each_tiny_cpu_model_without_saving(self):
        references = []

        class ProbeModel(torch.nn.Linear):
            def save_pretrained(self, *args, **kwargs):
                raise AssertionError('Timing must not save updated weights')

        def new_model(*args):
            model = ProbeModel(1, 2)
            model.should_oom = not references
            references.append(weakref.ref(model))
            return model, 'a'*64

        def step(*args):
            if args[0].should_oom:
                raise torch.cuda.OutOfMemoryError('CPU synthetic OOM control flow only')
            return 1., len(args[5]), False

        plan = training.DEFAULT_PLAN.copy()
        plan.update(microbatch=2, profile_optimizer_steps=2)
        with patch.object(training, '_new_model', new=new_model), \
                patch.object(training, '_optimizer', new=lambda *args: object()), \
                patch.object(training, '_step', new=step), \
                patch('torch.amp.GradScaler', new=lambda *args: object()), \
                patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.synchronize'), \
                patch('torch.cuda.max_memory_allocated', return_value=0), \
                patch('torch.cuda.max_memory_reserved', return_value=0), \
                patch('torch.cuda.get_device_name', return_value='CPU MOCK CONTROL-FLOW ONLY'), \
                patch('torch.cuda.device_count', return_value=0), \
                patch('torch.cuda.get_device_properties', return_value=types.SimpleNamespace(total_memory=1000)), \
                patch('torch.cuda.empty_cache') as cleanup:
            result = training._profile(self.root, plan, self.records, [], object(), methods_count=1)
            self.assertEqual(result['microbatch'], 1)
            self.assertEqual(len(result['oom_attempts']), 1)
            self.assertTrue(result['profile_weights_discarded'])
            self.assertEqual(cleanup.call_count, 2)
        gc.collect()
        self.assertEqual(len(references), 2)
        self.assertTrue(all(reference() is None for reference in references))


class ThroughputWorkerTests(unittest.TestCase):
    def test_throughput_worker_returns_before_measured_training_ml_or_release(self):
        from scripts import run_kaggle_experiment as worker
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root/'out'
            argv = ['worker', '--repo-root', str(root), '--manifest', str(root/'manifest'),
                    '--output', str(output), '--plan', str(root/'plan'), '--throughput-profile']
            fake = {'status': 'profile_completed', 'selected_microbatch': 8,
                    'selected_optimizer_median_seconds': 1., 'speedup_vs_microbatch2': 2.,
                    'profile_weights_discarded': True, 'retained_task_models_trained': False,
                    'selection_inference_performed': False, 'calibration_evaluated': False,
                    'locked_holdout_evaluated': False, 'scope': 'CPU MOCK CONTROL-FLOW ONLY'}
            with patch.object(sys, 'argv', argv), \
                    patch.object(training, 'run_throughput_profile', return_value=fake) as throughput, \
                    patch.object(training, 'run_transformer_experiment') as measured_training, \
                    patch('scripts.run_parallel_transformers.run_parallel_transformers') as parallel, \
                    patch('nlu_authorship.stylometric.train_stylometric') as ml, \
                    patch('nlu_authorship.release.release_models') as release, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(), 0)
            throughput.assert_called_once()
            for function in (measured_training, parallel, ml, release):
                function.assert_not_called()
            self.assertEqual(json.loads((output/'throughput-completion.json').read_text()), fake)
            self.assertFalse((output/'experiment-completion.json').exists())
            self.assertFalse((output/'release').exists())

    def test_profile_flags_are_mutually_exclusive(self):
        from scripts import run_kaggle_experiment as worker
        for conflicting in ('--profile-only', '--complete-release'):
            with self.subTest(flag=conflicting), \
                    patch.object(sys, 'argv', ['worker', '--repo-root', '.', '--manifest', 'x',
                                             '--output', 'y', '--plan', 'z', '--throughput-profile', conflicting]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    worker.main()
                self.assertEqual(error.exception.code, 2)

    def test_throughput_mode_rejects_trained_ml_input_before_any_work(self):
        from scripts import run_kaggle_experiment as worker
        with patch.object(sys, 'argv', ['worker', '--repo-root', '.', '--manifest', 'x',
                                      '--output', 'y', '--plan', 'z', '--throughput-profile',
                                      '--stylometric-model-dir', 'trained-model']), \
                patch.object(training, 'run_throughput_profile') as throughput, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                worker.main()
            self.assertEqual(error.exception.code, 2)
            throughput.assert_not_called()
