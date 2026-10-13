"""Scientific and numerical contracts, using tiny CPU-only fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nlu_authorship.losses import binary_asymmetric_focal_loss
from nlu_authorship.models import verify_artifact_inventory,write_artifact_inventory,load_transformer,export_dynamic_int8,sha256_file
from nlu_authorship.stylometric import PairFeatureExtractor,StylometricRunner
from nlu_authorship.training import epoch_schedule,schedule_sha256,validated_plan,require_kaggle_gpu


class LossTests(unittest.TestCase):
    def test_equal_logits_analytic_class_asymmetry(self):
        logits=torch.zeros((2,2),requires_grad=True)
        loss=binary_asymmetric_focal_loss(logits,torch.tensor([0,1]))
        self.assertAlmostEqual(float(loss.detach()),np.log(2)*(.25+.5)/2,places=7)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_gamma_recovers_cross_entropy_values_and_gradients(self):
        x=torch.tensor([[2.,-1.],[-.3,.7],[1.,1.]],requires_grad=True)
        labels=torch.tensor([0,1,0])
        focal=binary_asymmetric_focal_loss(x,labels,gamma_neg=0,gamma_pos=0)
        ce=torch.nn.functional.cross_entropy(x,labels)
        self.assertTrue(torch.allclose(focal,ce))
        self.assertTrue(torch.allclose(torch.autograd.grad(focal,x,retain_graph=True)[0],torch.autograd.grad(ce,x)[0]))

    def test_half_precision_extreme_wrong_logits_remain_finite(self):
        x=torch.tensor([[-1000.,1000.],[1000.,-1000.]],dtype=torch.float16,requires_grad=True)
        loss=binary_asymmetric_focal_loss(x,torch.tensor([0,1]))
        self.assertEqual(loss.dtype,torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward();self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_labels_and_exponents_fail(self):
        with self.assertRaises(ValueError):binary_asymmetric_focal_loss(torch.zeros(2,2),torch.tensor([0,2]))
        with self.assertRaises(ValueError):binary_asymmetric_focal_loss(torch.zeros(2,2),torch.tensor([0,1]),gamma_neg=float('nan'))


class ScheduleTests(unittest.TestCase):
    def test_schedule_is_replayable_and_each_pair_occurs_once(self):
        records=[{'id':f'pair-{i}'} for i in range(100)]
        first=epoch_schedule(records,378,0)
        self.assertEqual(first,epoch_schedule(records,378,0))
        self.assertEqual(sorted(i for i,_ in first),list(range(100)))
        self.assertNotEqual(first,epoch_schedule(records,378,1))
        self.assertNotEqual(schedule_sha256(records,378,0),schedule_sha256(records,379,0))
        self.assertGreater(sum(s for _,s in first),20)
        self.assertLess(sum(s for _,s in first),80)

    def test_root_nested_plan_is_accepted_but_changed_recipe_rejected(self):
        root=Path(__file__).resolve().parents[1]
        self.assertEqual(validated_plan(root/'configs/experiment-plan.json')['seed'],378)
        plan=json.loads((root/'configs/experiment-plan.json').read_text())
        plan['transformer']['learning_rate']=.001
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json';p.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):validated_plan(p)

    def test_training_cannot_run_on_local_cpu(self):
        with patch('platform.system',return_value='Windows'):
            with self.assertRaisesRegex(RuntimeError,'Kaggle'):require_kaggle_gpu()


class FeatureTests(unittest.TestCase):
    def test_runner_ba_physically_swaps_without_mutating_input(self):
        class AsymmetricFeatures:
            def transform(self,records):
                return np.array([[float(r['text_1'])] for r in records])
        class Classifier:
            def predict_proba(self,x):
                return np.column_stack((1-x[:,0],x[:,0]))
        runner=StylometricRunner.__new__(StylometricRunner)
        runner.features=AsymmetricFeatures();runner.xgb=Classifier();runner.lgb=Classifier()
        runner.metadata={'xgb_weight':.5,'lgb_weight':.5}
        records=[{'id':'pair','text_1':'0.2','text_2':'0.8'}]
        self.assertAlmostEqual(float(runner.predict_proba(records,orientation='ab')[0,0]),.2)
        self.assertAlmostEqual(float(runner.predict_proba(records,orientation='ba')[0,0]),.8)
        self.assertEqual(records,[{'id':'pair','text_1':'0.2','text_2':'0.8'}])

    def test_selection_vocabulary_never_fits_and_pair_swap_is_symmetric(self):
        train=[{'text_1':'The river was calm and blue today.', 'text_2':'We walked slowly along the river.'},
               {'text_1':'I have a book and read it every day.', 'text_2':'They have a small garden and a tree.'}]
        extractor=PairFeatureExtractor().fit(train)
        selection=[{'text_1':'uniqueselectionvocabulary appears suddenly.', 'text_2':'The river was calm and blue today.'}]
        before=dict(extractor.word.vocabulary_)
        ab=extractor.transform(selection)
        ba=extractor.transform([{'text_1':r['text_2'],'text_2':r['text_1']} for r in selection])
        self.assertEqual(before,extractor.word.vocabulary_)
        self.assertNotIn('uniqueselectionvocabulary',before)
        self.assertEqual(ab.shape,ba.shape)
        self.assertEqual((ab!=ba).nnz,0)


class ArtifactTests(unittest.TestCase):
    def test_tiny_transformer_and_dynamic_int8_artifacts_roundtrip(self):
        from transformers import RobertaConfig,RobertaForSequenceClassification,PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        with tempfile.TemporaryDirectory() as d:
            fp32=Path(d)/'fp32';fp32.mkdir()
            backend=Tokenizer(WordLevel({'<s>':0,'<pad>':1,'</s>':2,'<unk>':3,'the':4,'river':5,'book':6,'today':7},unk_token='<unk>'))
            backend.pre_tokenizer=Whitespace()
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,bos_token='<s>',pad_token='<pad>',eos_token='</s>',unk_token='<unk>',model_max_length=512)
            tokenizer.save_pretrained(fp32)
            torch.manual_seed(378)
            model=RobertaForSequenceClassification(RobertaConfig(vocab_size=8,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=514,num_labels=2,pad_token_id=1))
            model.save_pretrained(fp32,safe_serialization=True)
            (fp32/'training-config.json').write_text(json.dumps({'format':'transformer-fp32','max_length':512}))
            write_artifact_inventory(fp32)
            records=[{'text_1':'the river today','text_2':'the book today'}]
            first=load_transformer(fp32).predict_logits(records)
            self.assertEqual(first.shape,(1,2));self.assertTrue(np.isfinite(first).all())
            target=Path(d)/'int8';report=export_dynamic_int8(fp32,target)
            runner=load_transformer(target)
            quantized=runner.predict_logits(records)
            self.assertTrue(np.isfinite(quantized).all())
            self.assertEqual(quantized.shape,(1,2))
            self.assertTrue(np.allclose(first,quantized,atol=.01))
            self.assertEqual(report['quantized_model_sha256'],sha256_file(target/'model-int8.pt'))

    def test_corrupted_weight_and_unexpected_pickle_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);weight=root/'model.bin';weight.write_bytes(b'weights')
            write_artifact_inventory(root)
            verify_artifact_inventory(root,('model.bin',))
            weight.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'SHA256'):verify_artifact_inventory(root)
            weight.write_bytes(b'weights');(root/'unlisted.pkl').write_bytes(b'untrusted')
            with self.assertRaisesRegex(ValueError,'unexpected'):verify_artifact_inventory(root)

    def test_dynamic_quantized_state_loads_with_weights_only(self):
        import io
        model=torch.nn.Sequential(torch.nn.Linear(4,3),torch.nn.ReLU(),torch.nn.Linear(3,2)).eval()
        quantized=torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear},dtype=torch.qint8)
        stream=io.BytesIO();torch.save(quantized.state_dict(),stream);stream.seek(0)
        restored=torch.ao.quantization.quantize_dynamic(copy.deepcopy(model),{torch.nn.Linear},dtype=torch.qint8)
        restored.load_state_dict(torch.load(stream,weights_only=True))
        x=torch.tensor([[.1,.2,.3,.4]])
        self.assertTrue(torch.equal(quantized(x),restored(x)))


class KaggleBootstrapTests(unittest.TestCase):
    def test_local_runtime_setup_is_rejected_before_any_install(self):
        from scripts import setup_kaggle_runtime as setup
        with patch.object(setup.platform,'system',return_value='Windows'),patch.object(setup.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'Kaggle'):setup.setup(Path.cwd(),Path.cwd()/'.local/forbidden-gpu-runtime')
            run.assert_not_called()

    def test_child_runtime_isolation_preserves_assigned_gpu(self):
        import os
        from scripts import setup_kaggle_runtime as setup
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1','PYTHONPATH':'unrelated','PIP_TARGET':'unrelated','UV_PROJECT_ENVIRONMENT':'unrelated'},clear=True):
            env=setup.child_environment(Path(d))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
            self.assertNotIn('PYTHONPATH',env)
            self.assertNotIn('PIP_TARGET',env)
            self.assertNotIn('UV_PROJECT_ENVIRONMENT',env)
            self.assertEqual(env['PYTHONNOUSERSITE'],'1')
            self.assertEqual(env['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
            self.assertTrue(Path(env['UV_PYTHON_INSTALL_DIR']).is_relative_to(Path(d)))


class ProfileOnlyTests(unittest.TestCase):
    """CPU/mock control-flow evidence, not a real GPU performance measurement."""

    def test_profile_only_returns_before_any_full_training_or_selection_inference(self):
        from nlu_authorship import training
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / 'manifest.json'
            manifest.write_text('{}')
            output = root / 'profile'
            records = [{'id':f'tiny-{i}','text_1':'First fixture passage.','text_2':'Second fixture passage.','label':i%2} for i in range(4)]
            loaded = []
            def load_records(repo, value, split):
                loaded.append(split)
                if split not in {'train','selection'}:
                    raise AssertionError('Calibration or final records must not be requested')
                return records
            profile = {'initial_state_sha256':'mock-only-state','profile_weights_discarded':True,
                       'projected_training_seconds_including_30pct_margin':training.DEFAULT_PLAN['training_budget_seconds']+1,
                       'within_training_budget':False,'microbatch':2}
            with patch.object(training,'require_kaggle_gpu'), \
                 patch('nlu_authorship.data.load_manifest',return_value={}), \
                 patch('nlu_authorship.data.load_records',side_effect=load_records), \
                 patch.object(training,'_prepare_pretrained',return_value=(root,{})), \
                 patch('transformers.AutoTokenizer.from_pretrained',return_value=object()), \
                 patch.object(training,'_tokenize',return_value=['mock-ab','mock-ba']), \
                 patch.object(training,'_profile',return_value=profile), \
                 patch.object(training,'_new_model') as new_model, \
                 patch.object(training,'_optimizer') as optimizer, \
                 patch.object(training,'_step') as step, \
                 patch.object(training,'_predict') as predict, \
                 patch.object(training,'write_artifact_inventory') as inventory, \
                 contextlib.redirect_stdout(io.StringIO()):
                report = training.run_transformer_experiment(root,manifest,output,profile_only=True)
            self.assertEqual(report['status'],'profile_completed')
            self.assertTrue(report['profile_only'])
            self.assertFalse(report['measured_models_trained'])
            self.assertFalse(report['calibration_evaluated'])
            self.assertFalse(report['locked_holdout_evaluated'])
            self.assertEqual(report['methods'],{})
            self.assertEqual(loaded,['train','selection'])
            for function in (new_model,optimizer,step,predict,inventory):
                function.assert_not_called()
            self.assertEqual({p.name for p in output.iterdir()},{'gpu-profile.json','profile-report.json'})

    def _check_profile_cleanup(self, fail=False):
        from nlu_authorship import training
        import gc
        import types
        import weakref
        references = []
        class ProbeModel(torch.nn.Linear):
            def save_pretrained(self,*args,**kwargs):
                raise AssertionError('Profiling must never save updated model weights')
        def new_model(*args):
            model = ProbeModel(1,2)
            references.append(weakref.ref(model))
            return model,'mock-cpu-initial-state'
        def step(*args):
            if fail:
                raise RuntimeError('synthetic profile failure')
            return 1.,len(args[5]),False
        plan = training.DEFAULT_PLAN.copy()
        plan['profile_optimizer_steps'] = 2
        records = [{'id':f'tiny-{i}'} for i in range(64)]
        with patch.object(training,'_new_model',new=new_model), \
             patch.object(training,'_optimizer',new=lambda *args:object()), \
             patch.object(training,'_step',new=step), \
             patch('torch.amp.GradScaler',new=lambda *args:object()), \
             patch('torch.cuda.reset_peak_memory_stats'), \
             patch('torch.cuda.synchronize'), \
             patch('torch.cuda.max_memory_allocated',return_value=0), \
             patch('torch.cuda.max_memory_reserved',return_value=0), \
             patch('torch.cuda.get_device_name',return_value='MOCK CPU CONTROL-FLOW FIXTURE'), \
             patch('torch.cuda.device_count',return_value=0), \
             patch('torch.cuda.get_device_properties',return_value=types.SimpleNamespace(total_memory=0)), \
             patch('torch.cuda.empty_cache') as empty_cache:
            if fail:
                with self.assertRaisesRegex(RuntimeError,'synthetic profile failure'):
                    training._profile(Path('.'),plan,records,[],object())
            else:
                result = training._profile(Path('.'),plan,records,[],object())
                self.assertTrue(result['profile_weights_discarded'])
                self.assertEqual(result['optimizer_steps'],2)
                self.assertEqual(result['gpu_name'],'MOCK CPU CONTROL-FLOW FIXTURE')
            empty_cache.assert_called_once()
        gc.collect()
        self.assertEqual(len(references),1)
        self.assertIsNone(references[0]())

    def test_profile_model_is_released_after_success_without_saving_weights(self):
        self._check_profile_cleanup()

    def test_profile_model_is_released_after_failure_without_saving_weights(self):
        self._check_profile_cleanup(fail=True)

    def test_profile_worker_returns_without_ml_or_release(self):
        import contextlib
        import io
        import sys
        from scripts import run_kaggle_experiment as worker
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            output = root / 'out'
            args = ['worker','--repo-root',str(root),'--manifest',str(root/'manifest.json'),
                    '--output',str(output),'--plan',str(root/'plan.json'),'--profile-only']
            result = {'profile':{'scope':'CPU mock control-flow only'}}
            with patch.object(sys,'argv',args), \
                 patch('nlu_authorship.training.run_transformer_experiment',return_value=result) as run, \
                 patch('nlu_authorship.stylometric.train_stylometric') as ml, \
                 patch('nlu_authorship.release.release_models') as release, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(worker.main(),0)
            self.assertTrue(run.call_args.kwargs['profile_only'])
            ml.assert_not_called()
            release.assert_not_called()
            report = json.loads((output/'profile-completion.json').read_text())
            self.assertFalse(report['measured_models_trained'])
            self.assertFalse(report['locked_holdout_evaluated'])
            self.assertFalse((output/'experiment-completion.json').exists())
            self.assertFalse((output/'release').exists())

    def test_profile_and_full_release_worker_flags_are_mutually_exclusive(self):
        import contextlib
        import io
        import sys
        from scripts import run_kaggle_experiment as worker
        with patch.object(sys,'argv',['worker','--repo-root','.','--manifest','x','--output','y','--plan','z','--profile-only','--complete-release']), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                worker.main()
        self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()
