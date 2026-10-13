"""Bounded, matched fresh RoBERTa-Large experiments on Kaggle GPU only."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from .models import sha256_file, write_artifact_inventory

DEFAULT_PLAN = {
    'schema_version':1,'model_id':'FacebookAI/roberta-large',
    'revision':'722cf37b1afa9454edce342e7895e588b6ff1d59',
    'pretrained_sha256':'047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38',
    'seed':378,'epochs':6,'max_length':512,'microbatch':2,'effective_batch':32,
    'learning_rate':5e-6,'warmup_ratio':.1,'weight_decay':.01,'max_grad_norm':1.,
    'profile_optimizer_steps':40,'training_budget_seconds':32400,
    'common_early_stopping_patience':3,
    'loss_methods':['ce','custom_asl'],'gradient_checkpointing':True,'selection_metric':'macro_f1_ab_threshold_0.5',
    'swap_policy':'sha256(seed:epoch:pair_id) first byte low bit; 1 means swap',
}


def validated_plan(plan_path=None):
    plan=DEFAULT_PLAN.copy()
    if plan_path is not None:
        supplied=json.loads(Path(plan_path).read_text(encoding='utf8'))
        if 'transformer' not in supplied or 'pretrained' not in supplied:
            raise ValueError('Expected the repository experiment-plan.json schema')
        transformer=supplied['transformer'];pretrained=supplied['pretrained']
        declared={'schema_version':supplied['schema_version'],'seed':supplied['seed'],
                  'model_id':pretrained['model_id'],'revision':pretrained['revision'],'pretrained_sha256':pretrained['safetensors_sha256'],
                  **{k:transformer[k] for k in ('epochs','max_length','learning_rate','warmup_ratio','weight_decay','max_grad_norm','gradient_checkpointing')},
                  'microbatch':transformer['micro_batch_size'],'effective_batch':transformer['effective_batch_size'],
                  'profile_optimizer_steps':transformer['profiling_updates'],'training_budget_seconds':transformer['max_project_training_wall_seconds'],
                  'common_early_stopping_patience':transformer['common_early_stopping_patience']}
        for key,value in declared.items():
            if value!=DEFAULT_PLAN[key]:
                raise ValueError('The fixed matched experiment setting cannot be changed: '+key)
        if transformer['losses']!=['cross_entropy','custom_binary_asymmetric_focal'] or transformer['gamma_negative']!=2 or transformer['gamma_positive']!=1:
            raise ValueError('The predeclared CE/custom binary focal comparison is required')
        if not transformer['fp16_cuda'] or not transformer['loss_float32'] or transformer['profile_oom_restarts']!=1:
            raise ValueError('The declared numerical precision and bounded profiling policy are required')
        if transformer['orientation_augmentation']!='seeded_50pct_swap_per_pair_per_epoch_without_doubling_rows' or transformer['checkpoint_selection']!='selection_macro_f1_at_threshold_0.5' or transformer['seed_repeats']!=1:
            raise ValueError('The declared pair schedule and checkpoint selection policy are required')
    return plan


def require_kaggle_gpu():
    import platform
    import torch
    if platform.system()!='Linux' or not Path('/kaggle/working').is_dir() or not (
        os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or Path('/kaggle/input').is_dir()):
        raise RuntimeError('Real training is restricted to an actual Kaggle GPU kernel')
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU must be enabled and visible to PyTorch')


def epoch_schedule(records, seed, epoch):
    order=list(range(len(records)))
    random.Random(f'{seed}:{epoch}:order').shuffle(order)
    result=[]
    for index in order:
        digest=hashlib.sha256(f"{seed}:{epoch}:{records[index]['id']}".encode('utf8')).digest()
        result.append((index,bool(digest[0]&1)))
    return result


def schedule_sha256(records, seed, epoch):
    return hashlib.sha256(json.dumps(epoch_schedule(records,seed,epoch),separators=(',',':')).encode()).hexdigest()


def _seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def _state_sha256(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _prepare_pretrained(plan, pretrained_dir=None):
    if pretrained_dir is None:
        from huggingface_hub import snapshot_download
        pretrained_dir=snapshot_download(plan['model_id'],revision=plan['revision'],token=False,
                                        allow_patterns=['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'])
    root=Path(pretrained_dir).resolve(strict=True)
    required=('model.safetensors','config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')
    for name in required:
        if not (root/name).is_file():
            raise ValueError('Fresh pinned pretrained artifact missing: '+name)
    if sha256_file(root/'model.safetensors')!=plan['pretrained_sha256']:
        raise ValueError('Pretrained weights differ from the declared fresh RoBERTa-Large revision')
    return root,{name:{'bytes':(root/name).stat().st_size,'sha256':sha256_file(root/name)} for name in required}


def _new_model(pretrained, plan):
    from transformers import AutoModelForSequenceClassification
    from transformers.utils import logging as transformer_logging
    transformer_logging.disable_progress_bar()
    _seed_everything(plan['seed'])
    model=AutoModelForSequenceClassification.from_pretrained(pretrained,num_labels=2,local_files_only=True,
                                                            trust_remote_code=False,attn_implementation='eager')
    if plan['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache=False
    fingerprint=_state_sha256(model)
    return model.to('cuda:0'),fingerprint


def _tokenize(records, tokenizer, max_length):
    from .normalization import normalize_roberta_text
    a=[normalize_roberta_text(r['text_1']) for r in records]
    b=[normalize_roberta_text(r['text_2']) for r in records]
    # Both orientations are pre-tokenized independently: truncation of equal
    # length passages can retain different tokens when the pair is reversed.
    return [tokenizer(a,b,max_length=max_length,truncation='longest_first',padding=False),
            tokenizer(b,a,max_length=max_length,truncation='longest_first',padding=False)]


def _batch(encodings, scheduled, tokenizer):
    import torch
    items=[]
    for index,swapped in scheduled:
        orientation=encodings[int(swapped)]
        items.append({key:values[index] for key,values in orientation.items()})
    return {key:value.to('cuda:0') for key,value in tokenizer.pad(items,padding=True,return_tensors='pt').items()}


def _predict(model, records, encodings, tokenizer, batch_size=16, orientation=0):
    import numpy as np
    import torch
    model.eval();values=[]
    with torch.inference_mode():
        for start in range(0,len(records),batch_size):
            inputs=_batch(encodings,[(i,bool(orientation)) for i in range(start,min(start+batch_size,len(records)))],tokenizer)
            output=model(**inputs).logits.float().cpu().numpy()
            if not np.isfinite(output).all():raise RuntimeError('Nonfinite selection logits')
            values.append(output)
    return np.concatenate(values)


def _optimizer(model, plan):
    import torch
    return torch.optim.AdamW(model.parameters(),lr=plan['learning_rate'],weight_decay=plan['weight_decay'],foreach=False)


def _step(model, optimizer, scaler, records, encodings, scheduled, tokenizer, microbatch, method, plan):
    import torch
    from .losses import classification_loss
    model.train();optimizer.zero_grad(set_to_none=True)
    total_loss=0.
    for offset in range(0,len(scheduled),microbatch):
        part=scheduled[offset:offset+microbatch]
        inputs=_batch(encodings,part,tokenizer)
        labels=torch.tensor([int(records[i]['label']) for i,_ in part],dtype=torch.long,device='cuda:0')
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            logits=model(**inputs).logits
        loss=classification_loss(logits,labels,method)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite FP32 classification loss')
        fraction=len(part)/len(scheduled)
        scaler.scale(loss*fraction).backward()
        total_loss+=float(loss.detach())*len(part)
    scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['max_grad_norm'])
    before=scaler.get_scale()
    scaler.step(optimizer);scaler.update()
    skipped=scaler.get_scale()<before
    if not skipped and not torch.isfinite(norm):raise RuntimeError('Nonfinite unscaled gradient')
    return total_loss,len(scheduled),skipped


def _profile(pretrained, plan, records, encodings, tokenizer, *, methods_count=None, deadline=None):
    import statistics
    import torch
    schedule=epoch_schedule(records,plan['seed'],0)
    methods_count=len(plan['loss_methods']) if methods_count is None else methods_count
    if isinstance(methods_count,bool) or methods_count not in (1,2):
        raise ValueError('The bounded profile must cover one worker or both paired methods')
    history=[]
    for attempt,microbatch in enumerate((plan['microbatch'],plan['microbatch']//2)):
        model=None;optimizer=None;scaler=None
        try:
            model,fingerprint=_new_model(pretrained,plan)
            optimizer=_optimizer(model,plan);scaler=torch.amp.GradScaler('cuda')
            torch.cuda.reset_peak_memory_stats()
            timings=[];skipped=0
            for step in range(plan['profile_optimizer_steps']):
                if deadline is not None and time.perf_counter()>=deadline:
                    raise TimeoutError('The bounded GPU timing profile reached its deadline')
                begin=step*plan['effective_batch']
                if begin>=len(schedule):break
                chunk=schedule[begin:begin+plan['effective_batch']]
                torch.cuda.synchronize();started=time.perf_counter()
                _,_,skip=_step(model,optimizer,scaler,records,encodings,chunk,tokenizer,microbatch,'ce',plan)
                torch.cuda.synchronize();timings.append(time.perf_counter()-started);skipped+=int(skip)
                if deadline is not None and time.perf_counter()>deadline:
                    raise TimeoutError('The bounded GPU timing profile reached its deadline')
                if (step+1)%10==0:
                    print(json.dumps({'stage':'gpu-profile-progress','optimizer_steps':step+1,'microbatch':microbatch,
                                      'last_step_seconds':timings[-1]}),flush=True)
            if len(timings)<2:raise ValueError('The real dataset is too small for the declared timing profile')
            seconds=statistics.median(timings[min(5,len(timings)//2):])
            total_steps=math.ceil(len(records)/plan['effective_batch'])*plan['epochs']*methods_count
            estimate=seconds*total_steps*1.3
            result={'microbatch':microbatch,'effective_batch':plan['effective_batch'],'optimizer_steps':len(timings),
                    'median_optimizer_step_seconds':seconds,'projected_training_seconds_including_30pct_margin':estimate,
                    'budget_seconds':plan['training_budget_seconds'],'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_gpu_bytes':torch.cuda.max_memory_reserved(),
                    'gpu_name':torch.cuda.get_device_name(0),'visible_gpu_count':torch.cuda.device_count(),
                    'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
                    'successful_optimizer_updates':len(timings)-skipped,
                    'parameter_count':sum(p.numel() for p in model.parameters()),
                    'within_training_budget':estimate<=plan['training_budget_seconds'],
                    'projected_method_count':methods_count,
                    'initial_state_sha256':fingerprint,'overflow_skipped_steps':skipped,'oom_attempts':history,
                    'profile_weights_discarded':True}
            if skipped>max(2,len(timings)//10):raise RuntimeError('Too many numerical overflow steps in the timing profile')
            return result
        except torch.cuda.OutOfMemoryError:
            history.append({'attempt':attempt+1,'microbatch':microbatch,'error':'CUDA out of memory'})
            if attempt==1:raise RuntimeError('Both predeclared microbatch attempts ran out of memory') from None
        finally:
            del model,optimizer,scaler
            gc.collect();torch.cuda.empty_cache()
    raise RuntimeError('GPU profiling failed')


def _source_snapshot(repo_root):
    root=Path(repo_root)
    paths=sorted((root/'nlu_authorship').glob('*.py'))+sorted(root.glob('requirements*.txt'))
    if (root/'pyproject.toml').is_file():paths.append(root/'pyproject.toml')
    return {p.relative_to(root).as_posix():sha256_file(p) for p in paths}


def run_throughput_profile(repo_root, manifest_path, output_dir, plan_path, pretrained_dir=None):
    """Choose batch execution from timing and memory alone, never task scores."""
    require_kaggle_gpu()
    from transformers import AutoTokenizer
    from .data import load_records
    plan=validated_plan(plan_path)
    output=Path(output_dir)
    if output.exists():
        raise FileExistsError('Preserve prior timing evidence; use a new output')
    output.mkdir(parents=True)
    source=_source_snapshot(repo_root)
    manifest_hash=sha256_file(manifest_path)
    plan_hash=sha256_file(plan_path)
    records=load_records(repo_root,manifest_path,'train')
    pretrained,pretrained_inventory=_prepare_pretrained(plan,pretrained_dir)
    tokenizer=AutoTokenizer.from_pretrained(pretrained,local_files_only=True,trust_remote_code=False)
    encodings=_tokenize(records,tokenizer,plan['max_length'])
    started=time.perf_counter()
    candidates=[]
    declared={'candidate_max_microbatches':[2,8,32],'optimizer_attempts_per_candidate':20,
              'oom_fallback':'one half-size retry per candidate','time_budget_seconds':1200,
              'maximum_reserved_fraction_for_adoption':.9,'minimum_timing_speedup':1.05,
              'same_effective_batch':32,'timing_only':True,'performance_metrics_used':False}
    initial=None
    for microbatch in declared['candidate_max_microbatches']:
        if time.perf_counter()-started>=declared['time_budget_seconds']:
            candidates.append({'candidate_max_microbatch':microbatch,'status':'not_run_budget'})
            break
        candidate_plan={**plan,'microbatch':microbatch,'profile_optimizer_steps':20}
        print(json.dumps({'stage':'throughput-candidate-start','candidate_max_microbatch':microbatch}),flush=True)
        try:
            result=_profile(pretrained,candidate_plan,records,encodings,tokenizer,methods_count=1,
                            deadline=started+declared['time_budget_seconds'])
            if initial is None:
                initial=result['initial_state_sha256']
            if result['initial_state_sha256']!=initial:
                raise ValueError('Timing candidates must share identical fresh initial weights')
            candidates.append({'candidate_max_microbatch':microbatch,'status':'completed','profile':result})
        except TimeoutError as error:
            candidates.append({'candidate_max_microbatch':microbatch,'status':'failed_time_budget','error':str(error)})
            break
        except RuntimeError as error:
            if 'out of memory' not in str(error).lower():
                raise
            candidates.append({'candidate_max_microbatch':microbatch,'status':'failed_oom','error':str(error)})
        print(json.dumps({'stage':'throughput-candidate-complete','candidate':candidates[-1]}),flush=True)
    valid=[candidate for candidate in candidates if candidate['status']=='completed' and
           candidate['profile']['peak_reserved_gpu_bytes']<=declared['maximum_reserved_fraction_for_adoption']*candidate['profile']['gpu_total_bytes']]
    reference=next((candidate for candidate in candidates if candidate['candidate_max_microbatch']==2 and candidate['status']=='completed'),None)
    if reference is None:
        raise RuntimeError('Required microbatch2 timing reference failed; no execution policy can be selected')
    if not valid:
        raise RuntimeError('No timing candidate retained the declared GPU memory margin')
    fastest=min(valid,key=lambda candidate:(candidate['profile']['median_optimizer_step_seconds'],candidate['profile']['microbatch']))
    speedup=reference['profile']['median_optimizer_step_seconds']/fastest['profile']['median_optimizer_step_seconds']
    if speedup>=declared['minimum_timing_speedup']:
        selected=fastest
    elif reference in valid:
        selected=reference
    else:
        raise RuntimeError('No timing candidate satisfies both the memory margin and the predeclared speedup gate')
    if _source_snapshot(repo_root)!=source or sha256_file(manifest_path)!=manifest_hash or sha256_file(plan_path)!=plan_hash:
        raise RuntimeError('Timing source, data or plan changed during profiling')
    report={'schema_version':1,'status':'profile_completed','profile_only':True,'scope':'Actual T4 timing and memory only',
            'declared':declared,'plan':plan,'plan_sha256':plan_hash,'manifest_file_sha256':manifest_hash,
            'source_snapshot':source,'pretrained_inventory':pretrained_inventory,'train_count':len(records),
            'initial_state_sha256':initial,'candidates':candidates,'selected_microbatch':selected['profile']['microbatch'],
            'selected_optimizer_median_seconds':selected['profile']['median_optimizer_step_seconds'],
            'selected_peak_reserved_bytes':selected['profile']['peak_reserved_gpu_bytes'],
            'speedup_vs_microbatch2':reference['profile']['median_optimizer_step_seconds']/selected['profile']['median_optimizer_step_seconds'],
            'selection_inference_performed':False,'calibration_evaluated':False,'locked_holdout_evaluated':False,
            'profile_weights_discarded':True,'retained_task_models_trained':False,
            'all_declared_candidates_completed':sum(c['status']=='completed' for c in candidates)==len(declared['candidate_max_microbatches']),
            'elapsed_profile_seconds':time.perf_counter()-started}
    (output/'throughput-profile.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
    return report


def _coordinate_epoch(directory, method, item, report, initial_state_sha256, deadline):
    """Wait for the shared selection-only decision after both equal epochs."""
    directory=Path(directory).resolve(strict=True)
    if method not in ('ce','custom_asl'):
        raise ValueError('Unknown paired worker method')
    epoch=int(item['epoch'])
    target=directory/f'epoch-{epoch:04d}-{method}.json'
    if target.exists():
        raise FileExistsError('Do not overwrite an existing paired epoch report')
    payload={'method':method,'epoch':epoch,'item':item,'initial_state_sha256':initial_state_sha256,
             'manifest_file_sha256':report['manifest_file_sha256'],'plan_sha256':report['plan_sha256'],
             'microbatch':report['profile']['microbatch']}
    temporary=directory/f'.epoch-{epoch:04d}-{method}-{os.getpid()}.tmp'
    with temporary.open('x',encoding='utf8',newline='\n') as stream:
        stream.write(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    temporary.replace(target)
    decision=directory/f'epoch-{epoch:04d}-decision.json'
    while True:
        if (directory/'abort.json').exists():
            raise RuntimeError('The paired experiment coordinator aborted; preserve partial evidence')
        if time.perf_counter()>deadline:
            raise RuntimeError('The paired epoch barrier reached the predeclared worker deadline')
        if decision.is_file():
            value=json.loads(decision.read_text(encoding='utf8'))
            if value.get('epoch')!=epoch or value.get('action') not in ('continue','stop'):
                raise ValueError('Invalid shared paired epoch decision')
            return value
        time.sleep(.25)


def run_transformer_experiment(repo_root, manifest_path, output_dir, plan_path=None, pretrained_dir=None, *, profile_only=False, method=None, epoch_coordinator_dir=None):
    require_kaggle_gpu()
    import numpy as np
    import torch
    from transformers import AutoTokenizer,get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score
    from .data import load_manifest,load_records
    plan=validated_plan(plan_path)
    if method is not None and method not in plan['loss_methods']:
        raise ValueError('A worker must run one of the predeclared paired losses')
    methods=plan['loss_methods'] if method is None else [method]
    if epoch_coordinator_dir is not None and (profile_only or method is None):
        raise ValueError('Shared epoch coordination requires one actual training method')
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if any((out/m).exists() for m in methods):
        raise ValueError('Existing training checkpoints cannot be silently overwritten or resumed')
    source=_source_snapshot(repo_root)
    manifest=load_manifest(manifest_path)
    train=load_records(repo_root,manifest,'train')
    selection=load_records(repo_root,manifest,'selection')
    if not train or not selection:raise ValueError('Train and selection splits must be nonempty')
    # The trainer deliberately never loads calibration or locked_holdout rows.
    pretrained,pretrained_inventory=_prepare_pretrained(plan,pretrained_dir)
    tokenizer=AutoTokenizer.from_pretrained(pretrained,local_files_only=True,trust_remote_code=False)
    train_enc=_tokenize(train,tokenizer,plan['max_length'])
    sel_enc=_tokenize(selection,tokenizer,plan['max_length'])
    global_start=time.perf_counter()
    profile=_profile(pretrained,plan,train,train_enc,tokenizer,methods_count=len(methods))
    (out/'gpu-profile.json').write_text(json.dumps(profile,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'stage':'gpu-profile','result':profile}),flush=True)
    report={'schema_version':1,'plan':plan,'plan_sha256':sha256_file(plan_path) if plan_path else None,
            'manifest_file_sha256':sha256_file(manifest_path),'source_snapshot':source,
            'pretrained_inventory':pretrained_inventory,'profile':profile,'train_count':len(train),'selection_count':len(selection),
            'calibration_evaluated':False,'locked_holdout_evaluated':False,'seed_count':1,
            'epoch_schedule_sha256':[schedule_sha256(train,plan['seed'],e) for e in range(plan['epochs'])],
            'requested_methods':list(methods),'methods':{}}
    if profile_only:
        if _source_snapshot(repo_root)!=source or sha256_file(manifest_path)!=report['manifest_file_sha256'] or (
            plan_path and sha256_file(plan_path)!=report['plan_sha256']):
            raise RuntimeError('Source, split or plan changed during capability profiling')
        report.update({'status':'profile_completed','profile_only':True,
                       'profile_weights_discarded':True,'measured_models_trained':False,
                       'elapsed_profile_seconds':time.perf_counter()-global_start})
        (out/'profile-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        return report
    if time.perf_counter()-global_start+profile['projected_training_seconds_including_30pct_margin']>plan['training_budget_seconds']:
        raise RuntimeError('Profile plus projected matched training exceeds the predeclared combined budget')
    initial=None
    for method in methods:
        if _source_snapshot(repo_root)!=source or sha256_file(manifest_path)!=report['manifest_file_sha256'] or (
            plan_path and sha256_file(plan_path)!=report['plan_sha256']):
            raise RuntimeError('Training source or split manifest changed during the experiment')
        model,fingerprint=_new_model(pretrained,plan)
        if initial is None:initial=fingerprint
        if fingerprint!=initial or fingerprint!=profile['initial_state_sha256']:
            raise RuntimeError('Matched model initialization differs')
        optimizer=_optimizer(model,plan)
        steps=math.ceil(len(train)/plan['effective_batch'])*plan['epochs']
        scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=math.ceil(plan['warmup_ratio']*steps),num_training_steps=steps)
        scaler=torch.amp.GradScaler('cuda')
        torch.cuda.reset_peak_memory_stats()
        history=[];best=-1.;best_epoch=None;skipped_total=0;method_start=time.perf_counter();stop_decision=None
        destination=out/method;destination.mkdir()
        for epoch in range(plan['epochs']):
            schedule=epoch_schedule(train,plan['seed'],epoch)
            loss_sum=0.;seen=0;skipped=0;training_start=time.perf_counter()
            for start in range(0,len(schedule),plan['effective_batch']):
                if time.perf_counter()-global_start>plan['training_budget_seconds']:
                    raise RuntimeError('The fixed experiment reached its predeclared training time budget')
                loss,count,skip=_step(model,optimizer,scaler,train,train_enc,schedule[start:start+plan['effective_batch']],tokenizer,profile['microbatch'],method,plan)
                loss_sum+=loss;seen+=count;skipped+=int(skip)
                if not skip:scheduler.step()
                updates=start//plan['effective_batch']+1
                if updates%100==0:
                    print(json.dumps({'stage':'training-progress','method':method,'epoch':epoch+1,'optimizer_attempts':updates,
                                      'training_loss_so_far':loss_sum/seen}),flush=True)
            torch.cuda.synchronize();train_seconds=time.perf_counter()-training_start
            eval_start=time.perf_counter()
            logits=_predict(model,selection,sel_enc,tokenizer,batch_size=4)
            labels=np.array([int(r['label']) for r in selection])
            metric=float(f1_score(labels,logits[:,1]>=logits[:,0],average='macro',labels=[0,1],zero_division=0))
            ce=float(torch.nn.functional.cross_entropy(torch.tensor(logits),torch.tensor(labels)).item())
            item={'epoch':epoch+1,'training_loss':loss_sum/seen,'training_loss_method':method,'training_seconds':train_seconds,
                  'selection_macro_f1_ab_threshold_0_5':metric,'selection_cross_entropy':ce,
                  'selection_seconds':time.perf_counter()-eval_start,'examples_seen':seen,'optimizer_attempts':math.ceil(len(train)/plan['effective_batch']),
                  'overflow_skipped_steps':skipped,'schedule_sha256':report['epoch_schedule_sha256'][epoch]}
            skipped_total+=skipped
            history.append(item);print(json.dumps({'stage':'training-epoch','method':method,**item}),flush=True)
            if metric>best:
                best=metric;best_epoch=epoch+1
                model.save_pretrained(destination,safe_serialization=True)
                tokenizer.save_pretrained(destination)
            if skipped_total>max(3,steps//20):raise RuntimeError('Too many skipped numerical overflow optimizer steps')
            if epoch_coordinator_dir is not None:
                shared=_coordinate_epoch(epoch_coordinator_dir,method,item,report,initial,
                                         global_start+plan['training_budget_seconds'])
                if shared['action']=='stop':
                    stop_decision=shared
                    break
        metadata={**plan,'format':'transformer-fp32','loss_method':method,'microbatch':profile['microbatch'],
                  'parameter_count':sum(p.numel() for p in model.parameters()),'saved_weight_dtype':'float32',
                  'initial_state_sha256':initial,'manifest_file_sha256':report['manifest_file_sha256'],
                  'best_epoch':best_epoch,'selection_macro_f1':best,'selection_threshold':.5,
                  'epochs_completed':len(history),'seed_count':1,'shared_stop_decision':stop_decision}
        (destination/'training-config.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
        checkpoint={'method':method,'best_epoch':best_epoch,'selection_macro_f1':best,'initial_state_sha256':initial,
                    'epochs':history,'training_seconds':time.perf_counter()-method_start,
                    'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),'overflow_skipped_steps':skipped_total,
                    'model_sha256':sha256_file(destination/'model.safetensors'),'model_bytes':(destination/'model.safetensors').stat().st_size,
                    'epochs_completed':len(history),'shared_stop_decision':stop_decision}
        (destination/'checkpoint-report.json').write_text(json.dumps(checkpoint,indent=2)+'\n',encoding='utf8')
        del model,optimizer,scheduler,scaler;gc.collect();torch.cuda.empty_cache()
        from .models import TransformerRunner
        runner=TransformerRunner(destination,device='cuda',verify=False)
        ab=runner.predict_logits(selection,batch_size=4,orientation='ab')
        ba=runner.predict_logits(selection,batch_size=4,orientation='ba')
        (out/f'selection-{method}-logits.jsonl').write_text(''.join(json.dumps({'id':r['id'],'label':int(r['label']),
                    'logits_ab':[float(v) for v in aa],'logits_ba':[float(v) for v in bb]})+'\n' for r,aa,bb in zip(selection,ab,ba)),encoding='utf8')
        del runner;gc.collect();torch.cuda.empty_cache()
        write_artifact_inventory(destination)
        report['methods'][method]=checkpoint
        (out/'training-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if _source_snapshot(repo_root)!=source or sha256_file(manifest_path)!=report['manifest_file_sha256'] or (
        plan_path and sha256_file(plan_path)!=report['plan_sha256']):
        raise RuntimeError('Training source, split, or plan changed before completion')
    report['initial_state_sha256']=initial
    report['status']='completed'
    report['training_seconds']=time.perf_counter()-global_start
    (out/'training-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    return report
