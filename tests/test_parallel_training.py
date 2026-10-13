"""Real CPU subprocess orchestration tests; no GPU training or score claims."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.run_parallel_transformers import run_parallel_transformers


WORKER = r'''
import argparse,datetime,hashlib,json,os,pathlib,time
p=argparse.ArgumentParser()
for name in ['repo-root','manifest','output','plan','method','assigned-cuda-token','assigned-visible-index','epoch-coordinator-dir']:
 p.add_argument('--'+name,required=True)
p.add_argument('--pretrained-dir')
a=p.parse_args();out=pathlib.Path(a.output);out.mkdir()
plan=json.loads(pathlib.Path(a.plan).read_text());method=a.method
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
start=time.perf_counter();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();begin=utc()
write=lambda p,x:pathlib.Path(p).write_text(json.dumps(x)+'\n')
scenario=plan.get('scenario','normal')
if scenario=='failure':
 if method=='ce':time.sleep(.05);raise SystemExit(7)
 time.sleep(10)
if scenario=='timeout':time.sleep(10)
coordinator=pathlib.Path(a.epoch_coordinator_dir)
epochs=[]
for epoch in range(1,plan['epochs']+1):
 time.sleep(.035)
 metric=plan.get('metric_values',{}).get(method,[.5+e*.01 for e in range(plan['epochs'])])[epoch-1]
 item={'epoch':epoch,'selection_macro_f1_ab_threshold_0_5':metric,'schedule_sha256':plan['fixture_schedules'][epoch-1]}
 microbatch=1 if scenario=='microbatch_mismatch' and method=='custom_asl' else 2
 event={'method':method,'epoch':epoch,'item':item,'initial_state_sha256':'a'*64,'plan_sha256':sha(a.plan),'manifest_file_sha256':sha(a.manifest),'microbatch':microbatch}
 path=coordinator/f'epoch-{epoch:04d}-{method}.json';temporary=path.with_suffix('.tmp');write(temporary,event);temporary.replace(path)
 decision=coordinator/f'epoch-{epoch:04d}-decision.json'
 while not decision.exists():
  if (coordinator/'abort.json').exists():raise SystemExit(9)
  time.sleep(.005)
 epochs.append(item)
 if json.loads(decision.read_text())['action']=='stop':break
initial=('b'*64) if scenario=='initial_mismatch' and method=='custom_asl' else 'a'*64
model=out/method;model.mkdir();(model/'model.safetensors').write_bytes(('synthetic-'+method).encode())
checkpoint={'method':method,'best_epoch':1,'selection_macro_f1':epochs[0]['selection_macro_f1_ab_threshold_0_5'],
 'initial_state_sha256':initial,'epochs':epochs,'model_sha256':sha(model/'model.safetensors'),'model_bytes':(model/'model.safetensors').stat().st_size}
write(model/'checkpoint-report.json',checkpoint);write(model/'config.json',{'num_labels':2});write(model/'training-config.json',{'loss_method':method})
inventory={'schema_version':1,'files':[{'path':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(model.iterdir())]}
write(model/'artifact-inventory.json',inventory)
manifest=json.loads(pathlib.Path(a.manifest).read_text());selection=[r for r in manifest['records'] if r['split']=='selection']
if scenario=='selection_mismatch' and method=='custom_asl':selection=list(reversed(selection))
(out/f'selection-{method}-logits.jsonl').write_text(''.join(json.dumps({'id':r['id'],'label':r['label'],'logits_ab':[0.,1.],'logits_ba':[0.,.8]})+'\n' for r in selection))
report={'status':'completed','plan':plan,'plan_sha256':sha(a.plan),'manifest_file_sha256':sha(a.manifest),
 'source_snapshot':plan['fixture_source_snapshot'],'epoch_schedule_sha256':plan['fixture_schedules'],
 'calibration_evaluated':False,'locked_holdout_evaluated':False,'seed_count':1,
 'initial_state_sha256':initial,'pretrained_inventory':{'fixture':'no model download'},'profile':{'scope':'synthetic CPU process','microbatch':microbatch},
 'train_count':4,'selection_count':len(selection),'methods':{method:checkpoint}}
write(out/'training-report.json',report)
completion={'status':'completed','method':method,'report_filename':'training-report.json','calibration_evaluated':False,'locked_holdout_evaluated':False,
 'started_at_utc':begin,'completed_at_utc':utc(),'worker_elapsed_seconds':time.perf_counter()-start,
 'gpu_identity':{'assigned_cuda_token':os.environ['CUDA_VISIBLE_DEVICES'],'visible_gpu_count':1,'scope':'CPU fixture only'}}
write(out/'worker-completion.json',completion)
'''


class ParallelTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = self.root / 'synthetic_worker.py'
        self.worker.write_text(WORKER)
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps({'records':[{'id':'s0','label':0,'split':'selection'},
                                                        {'id':'s1','label':1,'split':'selection'}]}))
        self.plan_path = self.root / 'plan.json'
        self.output = self.root / 'transformers'
        self.devices = [{'visible_index':i,'token':f'cpu-fixture-{i}','name':'CPU mock only'} for i in range(2)]

    def tearDown(self):
        self.temp.cleanup()

    def run_case(self, scenario='normal', epochs=2, metric_values=None, **kwargs):
        plan = {'epochs':epochs,'seed':378,'fixture_source_snapshot':{'fixture':'same source'},
                'fixture_schedules':[f'scheduled-epoch-{i}' for i in range(1,epochs+1)],'scenario':scenario}
        if metric_values is not None:
            plan['metric_values'] = metric_values
        self.plan_path.write_text(json.dumps(plan))
        options = {'wall_timeout_seconds':5,'gpu_budget_seconds':10,'stop_grace_seconds':.2,'poll_seconds':.01,
                   'require_gpu':False,'worker_script':self.worker,'devices':self.devices}
        options.update(kwargs)
        return run_parallel_transformers(self.root,self.manifest,self.output,self.plan_path,**options)

    def test_real_subprocesses_overlap_and_promote_all_artifacts_atomically(self):
        report = self.run_case()
        self.assertEqual(report['status'],'completed')
        self.assertEqual(report['actual_matched_epochs'],2)
        workers = report['workers']
        self.assertLess(workers['ce']['started_at_utc'],workers['custom_asl']['completed_at_utc'])
        self.assertLess(workers['custom_asl']['started_at_utc'],workers['ce']['completed_at_utc'])
        for index,method in enumerate(('ce','custom_asl')):
            self.assertEqual(workers[method]['gpu_identity']['assigned_cuda_token'],f'cpu-fixture-{index}')
            self.assertTrue((self.output/method/'model.safetensors').is_file())
            self.assertFalse((self.root/'transformers-parallel/workers'/method/method).exists())
        self.assertIn('synthetic CPU',report['execution_scope'])
        self.assertFalse(report['locked_holdout_evaluated'])
        self.assertTrue((self.root/'transformers-parallel/logs/ce.log').is_file())

    def test_shared_patience_stops_both_at_identical_epoch(self):
        report = self.run_case(epochs=6,metric_values={'ce':[.6]*6,'custom_asl':[.6]*6})
        self.assertEqual(report['actual_matched_epochs'],4)
        self.assertEqual(report['common_stop_decisions'][-1]['reason'],'common_selection_patience')
        self.assertEqual({len(x['epochs']) for x in report['methods'].values()},{4})

    def test_either_method_improvement_resets_shared_patience(self):
        report = self.run_case(epochs=6,metric_values={'ce':[.6]*6,'custom_asl':[.6,.7,.7,.7,.7,.7]})
        self.assertEqual(report['actual_matched_epochs'],5)
        self.assertEqual(report['common_stop_decisions'][1]['consecutive_epochs_neither_improved'],0)

    def test_failure_terminates_only_own_other_worker_preserving_logs(self):
        kwargs = {'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
        sentinel = subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(5)'],**kwargs)
        try:
            with self.assertRaisesRegex(RuntimeError,'failed'):
                self.run_case('failure')
            self.assertIsNone(sentinel.poll())
            self.assertFalse(self.output.exists())
            failure = json.loads((self.root/'transformers-parallel/parallel-failure.json').read_text())
            self.assertTrue(failure['partial_outputs_preserved'])
            self.assertFalse(failure['release_performed'])
            self.assertTrue(all(value is not None for value in failure['owned_worker_return_codes'].values()))
            self.assertTrue((self.root/'transformers-parallel/logs/custom_asl.log').is_file())
        finally:
            sentinel.terminate();sentinel.wait(timeout=5)

    def test_mismatched_initialization_never_creates_half_canonical_result(self):
        with self.assertRaisesRegex(ValueError,'initial-state'):
            self.run_case('initial_mismatch')
        self.assertFalse(self.output.exists())

    def test_one_sided_oom_microbatch_fallback_aborts_shared_barrier(self):
        with self.assertRaisesRegex(ValueError,'microbatch'):
            self.run_case('microbatch_mismatch')
        self.assertFalse(self.output.exists())
        failure = json.loads((self.root/'transformers-parallel/parallel-failure.json').read_text())
        self.assertFalse(failure['release_performed'])
        self.assertTrue(all(value is not None for value in failure['owned_worker_return_codes'].values()))

    def test_deadline_is_failure_and_owned_workers_are_stopped(self):
        with self.assertRaises(TimeoutError):
            self.run_case('timeout',wall_timeout_seconds=.15)
        self.assertFalse(self.output.exists())
        report = json.loads((self.root/'transformers-parallel/parallel-failure.json').read_text())
        self.assertEqual(report['error_type'],'TimeoutError')
        self.assertTrue(all(value is not None for value in report['owned_worker_return_codes'].values()))

    def test_selection_order_mismatch_is_rejected_before_promotion(self):
        with self.assertRaisesRegex(ValueError,'ordered selection'):
            self.run_case('selection_mismatch')
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
