"""Run declared Kaggle training, optionally followed by the frozen release stage."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--pretrained-dir',type=Path)
    parser.add_argument('--stylometric-model-dir',type=Path,help='Verified CPU-trained ensemble; avoid spending assigned GPU time on CPU fits')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--complete-release',action='store_true',help='After all training, select, calibrate, freeze and evaluate fixed models')
    mode.add_argument('--profile-only',action='store_true',help='Measure Large forward/backward/update feasibility and discard all profile weights')
    mode.add_argument('--throughput-profile',action='store_true',help='Choose Large microbatch by actual T4 timing and memory only; discard all weights')
    args=parser.parse_args()
    if (args.profile_only or args.throughput_profile) and args.stylometric_model_dir is not None:
        parser.error('Profile-only runs must not load a trained ensemble')
    from nlu_authorship.training import run_transformer_experiment
    from nlu_authorship.stylometric import train_stylometric
    start=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True)
    try:
        if args.throughput_profile:
            from nlu_authorship.training import run_throughput_profile
            report=run_throughput_profile(args.repo_root,args.manifest,args.output/'throughput',args.plan,args.pretrained_dir)
            (args.output/'throughput-completion.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
            print(json.dumps({'stage':'large-throughput-profile-complete','selected_microbatch':report['selected_microbatch'],
                              'median_optimizer_seconds':report['selected_optimizer_median_seconds'],
                              'speedup':report['speedup_vs_microbatch2']}),flush=True)
            return 0
        if args.profile_only:
            report=run_transformer_experiment(args.repo_root,args.manifest,args.output/'transformers',args.plan,args.pretrained_dir,profile_only=True)
            result={'schema_version':1,'status':'profile_completed','profile_only':True,
                    'transformer':report,'profile_weights_discarded':True,
                    'measured_models_trained':False,'locked_holdout_evaluated':False,
                    'calibration_evaluated':False,'seconds':time.perf_counter()-start}
            (args.output/'profile-completion.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
            print(json.dumps({'stage':'large-capability-profile-complete','profile':report['profile'],'seconds':result['seconds']}),flush=True)
            return 0
        plan=json.loads(args.plan.read_text(encoding='utf8'))
        parallel=plan['transformer'].get('execution_mode')=='parallel_independent_loss_workers'
        if parallel and args.stylometric_model_dir is None:
            raise ValueError('Prepare the ensemble on CPU and package --stylometric-model before the paired GPU job')
        ml=None
        if args.stylometric_model_dir is not None:
            from nlu_authorship.models import verify_artifact_inventory,sha256_file
            from nlu_authorship.evaluation import artifact_inventory
            required=('ensemble_config.json','xgb_model.json','lgb_model.pkl','tfidf_word.pkl','tfidf_char.pkl','tfidf_fw.pkl')
            verify_artifact_inventory(args.stylometric_model_dir,required)
            ml=json.loads((args.stylometric_model_dir/'ensemble_config.json').read_text(encoding='utf8'))
            if ml.get('manifest_sha256')!=sha256_file(args.manifest) or ml.get('tfidf_fit_split')!='train' or ml.get('validation_split')!='selection':
                raise ValueError('CPU ensemble does not bind the declared train/selection split')
            if ml.get('study_protocol')!=plan['stylometric']['bounded_selection_study']['protocol']:
                raise ValueError('CPU ensemble study protocol differs from the declared plan')
            if not ml.get('training_source_snapshot'):
                raise ValueError('CPU ensemble has no executable source binding')
            for relative,expected in ml['training_source_snapshot'].items():
                path=(args.repo_root/relative).resolve(strict=True)
                if not path.is_relative_to(args.repo_root.resolve()) or sha256_file(path)!=expected:
                    raise ValueError('CPU ensemble source binding differs: '+relative)
            original=artifact_inventory(args.stylometric_model_dir)
            shutil.copytree(args.stylometric_model_dir,args.output/'stylometric')
            if artifact_inventory(args.output/'stylometric')!=original:
                raise ValueError('Copied CPU ensemble bytes differ')
            print(json.dumps({'stage':'cpu-ensemble-input-verified','selected_recipe':ml['selected_recipe'],
                              'artifact_content_sha256':original['content_sha256']}),flush=True)
        if parallel:
            from scripts.run_parallel_transformers import run_parallel_transformers
            report=run_parallel_transformers(args.repo_root,args.manifest,args.output/'transformers',args.plan,args.pretrained_dir,
                       wall_timeout_seconds=plan['transformer']['max_project_training_wall_seconds'],
                       gpu_budget_seconds=plan['transformer']['max_parallel_worker_gpu_hours']*3600,
                       common_patience=plan['transformer']['common_early_stopping_patience'])
        else:
            report=run_transformer_experiment(args.repo_root,args.manifest,args.output/'transformers',args.plan,args.pretrained_dir)
        print(json.dumps({'stage':'transformer-complete','seconds':time.perf_counter()-start}),flush=True)
        if ml is None:
            ml=train_stylometric(args.repo_root,args.manifest,args.output/'stylometric')
        result={'schema_version':1,'status':'completed','transformer':report,'stylometric':ml,
                'models':{method:str(args.output/'transformers'/method) for method in ('ce','custom_asl')},
                'locked_holdout_evaluated':False,'calibration_evaluated':False,'seconds':time.perf_counter()-start}
        result['models']['stylometric']=str(args.output/'stylometric')
        (args.output/'experiment-completion.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        print(json.dumps({'stage':'training-complete','seconds':result['seconds'],'models':result['models']}),flush=True)
        if args.complete_release:
            from nlu_authorship.release import release_models
            release=release_models(args.repo_root,args.manifest,result['models'],args.plan,args.output/'release',device='cuda',batch_size=4)
            print(json.dumps({'stage':'frozen-release-complete','selected_transformer':release['selected_transformer'],
                              'freeze_sha256':release['freeze_sha256'],'seconds':time.perf_counter()-start}),flush=True)
            result['release_completed']=True
            result['locked_holdout_evaluated']=True
            result['calibration_evaluated']=True
            result['release_summary']='release/release-summary.json'
            (args.output/'experiment-completion.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        return 0
    except Exception as error:
        failure={'status':'failed','error_type':type(error).__name__,'error':str(error),'seconds':time.perf_counter()-start,
                 'release_stage_requested':args.complete_release,'final_inference_may_have_started':(args.output/'release/before-final-evaluation.json').is_file()}
        (args.output/'experiment-failure.json').write_text(json.dumps(failure,indent=2)+'\n',encoding='utf8')
        print(json.dumps(failure),file=sys.stderr,flush=True)
        raise


if __name__=='__main__':raise SystemExit(main())
