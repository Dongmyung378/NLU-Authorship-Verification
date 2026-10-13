"""Original joint stylometric architecture with a bounded train/selection study.

The pair features and XGBoost/LightGBM ensemble originate from the joint work
of Dongmyung Park and Juho Kim. Task fitting is CPU-only on any supported host.
"""
from __future__ import annotations

import json
import hashlib
import gc
import importlib.metadata
import platform
import re
import sys
import time
from pathlib import Path

from .models import verify_artifact_inventory, write_artifact_inventory, sha256_file

FUNCTION_WORDS = [
    'the','a','an','and','or','but','if','in','on','at','to','for','of','with','by','from','as','is','was','are','were','be','been',
    'have','has','had','do','does','did','will','would','could','should','may','might','shall','that','which','who','whom','this','these','those',
    'i','we','you','he','she','they','it','my','your','his','her','however','therefore','moreover','furthermore','although','though','because','since',
    'while','when','where','what','how','why','just','also','very','really','quite','rather','so','too','even','still','already','not','only','then','here',
    'there','now','never','always','well','me','us','him','them','its','our','their','about','into','through','over','under','before','after','cannot','without',
    'of the','in the','to the','on the','for the','at the','by the','with the','from the','and the','of a','in a','to a','for a','with a',
    'it is','it was','there is','there are','there was','there were','to be','will be','would be','could be','should be','may be','is a','is the','was a','was the','are the','were the',
    'do not','does not','did not','will not','would not','could not','should not','have not','has not','had not','is not','are not','was not','were not',
    'i am','i was','i have','i had','i will','i would','i think','i know','i do','i can','i could','i feel','i want','i need','we are','we have','we will','we were','we had',
    'you are','you have','you will','you were','you can','he was','he is','he had','he would','he said','she was','she is','she had','she would','she said',
    'they are','they were','they have','they had','they will','to me','for me','of it','and i','but i','so i','that i','as well','as a','such as','as the','up to','out of',
    'in order','in fact','in addition','at least','at all','of course','as if','even if','even though','so that','in that','now that','given that','due to','based on','rather than',
]


def style_features(text):
    import numpy as np
    words = text.split()
    count = len(words)
    if count < 5:
        return np.zeros(24, dtype=np.float32)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    lengths = [len(w) for w in words]
    slengths = [len(s.split()) for s in sentences] or [count]
    return np.array([count, np.mean(lengths), np.mean(slengths), len(set(words))/count,
                     np.std(lengths), np.std(slengths), np.max(lengths), np.median(slengths),
                     *[text.count(c)/count for c in [',','.','!','?',';',':','"',"'"]],
                     sum(w.isupper() for w in words)/count, sum(len(w)>6 for w in words)/count,
                     sum(len(w)<=3 for w in words)/count, sum(w[0].isupper() for w in words)/count,
                     sum(w.endswith('ly') for w in words)/count, sum(w.endswith(('tion','sion')) for w in words)/count,
                     sum(w.endswith('ing') for w in words)/count, sum(w.endswith('ed') for w in words)/count], dtype=np.float32)


class PairFeatureExtractor:
    def __init__(self, word=None, char=None, function=None, feature_mode='full'):
        if feature_mode not in ('full','tfidf_only','style_only'):
            raise ValueError('Unknown feature mode')
        self.word, self.char, self.function = word, char, function
        self.feature_mode = feature_mode

    def fit(self, records):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from .normalization import normalize_stylometric_text
        # Only caller-provided TRAIN documents determine vocabulary and IDF.
        texts = [normalize_stylometric_text(r[k]) for r in records for k in ('text_1','text_2')]
        self.word = TfidfVectorizer(ngram_range=(1,2), max_features=7000, dtype=np.float32).fit(texts)
        self.char = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=7000, dtype=np.float32).fit(texts)
        self.function = TfidfVectorizer(vocabulary=FUNCTION_WORDS, ngram_range=(1,2), dtype=np.float32).fit(texts)
        return self

    def transform(self, records):
        import numpy as np
        import scipy.sparse as sp
        from sklearn.metrics.pairwise import paired_cosine_distances
        from .normalization import normalize_stylometric_text
        if not records:
            width = 72 if self.feature_mode=='style_only' else 2*(len(self.word.vocabulary_)+len(self.char.vocabulary_)+len(self.function.vocabulary_))+2
            if self.feature_mode=='full':width+=72
            return sp.csr_matrix((0,width),dtype=np.float32)
        a = [normalize_stylometric_text(r['text_1']) for r in records]
        b = [normalize_stylometric_text(r['text_2']) for r in records]
        sa, sb = np.array([style_features(t) for t in a]), np.array([style_features(t) for t in b])
        pair = sp.csr_matrix(np.hstack([abs(sa-sb),np.minimum(sa,sb)/(np.maximum(sa,sb)+1e-9),(sa+sb)/2]))
        if self.feature_mode=='style_only':return pair
        x1 = sp.hstack([self.word.transform(a),self.char.transform(a)],format='csr')
        x2 = sp.hstack([self.word.transform(b),self.char.transform(b)],format='csr')
        f1, f2 = self.function.transform(a), self.function.transform(b)
        cos = paired_cosine_distances(x1,x2).reshape(-1,1).astype(np.float32)
        fcos = paired_cosine_distances(f1,f2).reshape(-1,1).astype(np.float32)
        blocks=[abs(x1-x2),x1.multiply(x2),abs(f1-f2),f1.multiply(f2)]
        if self.feature_mode=='full':blocks.append(pair)
        return sp.hstack([*blocks,cos,fcos],format='csr',dtype=np.float32)


class StylometricRunner:
    def __init__(self, model_dir):
        import joblib
        import xgboost as xgb
        self.model_dir = Path(model_dir)
        self.inventory = verify_artifact_inventory(model_dir,('ensemble_config.json','xgb_model.json','lgb_model.pkl','tfidf_word.pkl','tfidf_char.pkl','tfidf_fw.pkl'))
        # Pickle decoding occurs only AFTER the complete trusted release hashes match.
        self.metadata = json.loads((self.model_dir/'ensemble_config.json').read_text(encoding='utf8'))
        self.features = PairFeatureExtractor(*[joblib.load(self.model_dir/name) for name in ('tfidf_word.pkl','tfidf_char.pkl','tfidf_fw.pkl')],feature_mode=self.metadata.get('feature_mode','full'))
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(self.model_dir/'xgb_model.json')
        self.xgb.set_params(device='cpu',n_jobs=1)
        self.lgb = joblib.load(self.model_dir/'lgb_model.pkl')
        self.lgb.set_params(n_jobs=1)

    def predict_proba(self, records, batch_size=256, orientation='ab'):
        import numpy as np
        import warnings
        if orientation not in ('ab','ba') or not 1 <= int(batch_size) <= 4096:
            raise ValueError('Invalid orientation or batch size')
        predictions=[]
        for offset in range(0,len(records),batch_size):
            batch=records[offset:offset+batch_size]
            if orientation=='ba':
                batch=[{**r,'text_1':r['text_2'],'text_2':r['text_1']} for r in batch]
            x=self.features.transform(batch)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='X does not have valid feature names',category=UserWarning)
                p=self.xgb.predict_proba(x)[:,1]*self.metadata['xgb_weight']+self.lgb.predict_proba(x)[:,1]*self.metadata['lgb_weight']
            if not np.isfinite(p).all():
                raise RuntimeError('Stylometric model produced a nonfinite score')
            predictions.append(p.reshape(-1,1))
        return np.concatenate(predictions) if predictions else np.empty((0,1),dtype=np.float64)


def load_stylometric(model_dir):
    return StylometricRunner(model_dir)


STUDY_PLAN = {
    'schema_version':1,'protocol':'nlu-stylometric-bounded-v1','seed':378,'cpu_threads':2,
    'total_budget_seconds':7200,'promotion_macro_f1_gain':.002,
    'reported_xgb_weights':[0.,.25,.5,.75,1.], 'final_xgb_weights':[.25,.5,.75],
    'decision_threshold':.5,'reference_recipe':'full-pilot','reference_xgb_weight':.5,
    'component_checkpoint_metric':'selection logloss; native best iteration',
    'recipes':[
        {'name':name,'feature_mode':mode,'xgb_cap':500,'lgb_cap':500,'learning_rate':.03,
         'max_depth':6,'num_leaves':63,'xgb_patience':30,'lgb_patience':30,'fit_budget_seconds':600}
        for name,mode in [('full-pilot','full'),('tfidf-pilot','tfidf_only'),('style-pilot','style_only')]]
        +[{'name':'full-original-caps','feature_mode':'full','xgb_cap':7000,'lgb_cap':2000,
           'learning_rate':.01,'max_depth':7,'num_leaves':127,'xgb_patience':200,'lgb_patience':100,
           'fit_budget_seconds':1800}],
    'tie_policy':'Prefer full features, then weight closest to0.5, then pilot, then lower XGB weight',
    'singles_are_diagnostics_only':True,'calibration_used':False,'locked_holdout_used':False,
}


def _write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n')


def _records_hash(records):
    from .data import canonical_sha256,text_sha256
    return canonical_sha256([{'id':r['id'],'label':int(r['label']),
                             'text_1':text_sha256(r['text_1']),'text_2':text_sha256(r['text_2'])} for r in records])


def _training_sources(repo_root):
    root=Path(repo_root)
    names=['nlu_authorship/stylometric.py','nlu_authorship/data.py','nlu_authorship/normalization.py',
           'nlu_authorship/models.py','nlu_authorship/metrics.py','pyproject.toml']
    names += [p.name for p in root.glob('requirements*.txt')]
    return {name:sha256_file(root/name) for name in sorted(names)}


def _cpu_provenance():
    return {'device':'cpu','operating_system':platform.platform(),'system':platform.system(),
            'machine':platform.machine(),'processor':platform.processor(),'python':platform.python_version(),
            'isolated':sys.prefix!=sys.base_prefix,'cpu_threads':2,
            'packages':{name:importlib.metadata.version(name) for name in
                        ('numpy','scipy','scikit-learn','joblib','xgboost','lightgbm')}}


def _idf_proof(features,records):
    import numpy as np
    from .normalization import normalize_stylometric_text
    texts=[normalize_stylometric_text(r[k]) for r in records for k in ('text_1','text_2')]
    result={'fitting_split':'train','document_count':len(texts),'train_records_sha256':_records_hash(records),'vectors':{}}
    for name,vector in [('word',features.word),('char',features.char),('function',features.function)]:
        matrix=vector.transform(texts)
        df=np.asarray((matrix>0).sum(axis=0)).ravel().astype(np.int64)
        expected=np.log((1+len(texts))/(1+df))+1
        error=float(np.max(np.abs(expected-vector.idf_)))
        if error>3e-6:raise RuntimeError('Saved IDF does not match train-only document frequencies')
        result['vectors'][name]={'vocabulary_size':len(vector.vocabulary_),'maximum_idf_error':error,
                                 'document_frequency_sha256':hashlib.sha256(df.tobytes()).hexdigest(),
                                 'idf_sha256':hashlib.sha256(vector.idf_.tobytes()).hexdigest(),
                                 'formula':'log((1+n_train_documents)/(1+train_document_frequency))+1'}
        del matrix;gc.collect()
    result['passed']=True
    return result


def selection_weights(labels,px,pl):
    from .metrics import binary_metrics
    scores=[{'xgb_weight':w,'lgb_weight':1-w,'diagnostic_only':w in (0.,1.),
             'metrics':binary_metrics(labels,w*px+(1-w)*pl,.5)} for w in STUDY_PLAN['reported_xgb_weights']]
    eligible=[r for r in scores if not r['diagnostic_only']]
    best=min(eligible,key=lambda r:(-r['metrics']['macro_f1'],abs(r['xgb_weight']-.5),r['xgb_weight']))
    return {'weights':scores,'best_nonzero_weight':best['xgb_weight'],'best_macro_f1':best['metrics']['macro_f1']}


def choose_recipe(reports):
    baseline=next(r for r in reports if r.get('status')=='completed' and r['name']=='full-pilot')
    reference=next(w['metrics']['macro_f1'] for w in baseline['weight_study']['weights'] if w['xgb_weight']==.5)
    completed=[r for r in reports if r.get('status')=='completed']
    order={r['name']:i for i,r in enumerate(STUDY_PLAN['recipes'])}
    candidate=min(completed,key=lambda r:(-r['weight_study']['best_macro_f1'],r['feature_mode']!='full',
                                          abs(r['weight_study']['best_nonzero_weight']-.5),order[r['name']],
                                          r['weight_study']['best_nonzero_weight']))
    gain=candidate['weight_study']['best_macro_f1']-reference
    accepted=gain+1e-15>=STUDY_PLAN['promotion_macro_f1_gain']
    return {'recipe':candidate['name'] if accepted else 'full-pilot',
            'feature_mode':candidate['feature_mode'] if accepted else 'full',
            'xgb_weight':candidate['weight_study']['best_nonzero_weight'] if accepted else .5,
            'reference_macro_f1':reference,'candidate_macro_f1':candidate['weight_study']['best_macro_f1'],
            'candidate_gain':gain,'promotion_passed':accepted,'selection_only':True}


def _fit_pair(xtrain,ytrain,xsel,ysel,recipe,deadline,directory):
    import xgboost as xgb
    import lightgbm as lgb
    import joblib
    directory=Path(directory);directory.mkdir()
    times={}
    class XTime(xgb.callback.TrainingCallback):
        def __init__(self,end):self.end=end;self.stopped=False
        def after_iteration(self,model,epoch,evals_log):
            if time.perf_counter()>=self.end:
                # CallbackContainer uses an unordered set and short-circuiting
                # any(). Preserve the current score even if this callback runs
                # before native EarlyStopping at the deadline iteration.
                score=float(evals_log['validation_0']['logloss'][-1])
                previous=model.attr('best_score')
                if previous is None or score<float(previous):
                    model.set_attr(best_iteration=str(epoch),best_score=str(score))
                self.stopped=True;return True
            return False
    class LTime:
        order=40;before_iteration=False
        def __init__(self,end):self.end=end;self.stopped=False;self.best=float('inf');self.iteration=0;self.scores=None
        def __call__(self,env):
            score=env.evaluation_result_list[0][2]
            if score<self.best:self.best=score;self.iteration=env.iteration;self.scores=list(env.evaluation_result_list)
            if time.perf_counter()>=self.end:
                self.stopped=True;raise lgb.callback.EarlyStopException(self.iteration,self.scores)
    if time.perf_counter()>=deadline:raise TimeoutError('Study fit budget exhausted before XGBoost')
    start=time.perf_counter();xtime=XTime(min(deadline,start+recipe['fit_budget_seconds']))
    xm=xgb.XGBClassifier(n_estimators=recipe['xgb_cap'],learning_rate=recipe['learning_rate'],max_depth=recipe['max_depth'],
                        subsample=.8,colsample_bytree=.8,min_child_weight=2,tree_method='hist',device='cpu',max_bin=128,
                        random_state=378,n_jobs=2,eval_metric='logloss',callbacks=[xtime,xgb.callback.EarlyStopping(
                            rounds=recipe['xgb_patience'],metric_name='logloss',data_name='validation_0',maximize=False,save_best=True)])
    xm.fit(xtrain,ytrain,eval_set=[(xsel,ysel)],verbose=False)
    xm.save_model(directory/'xgb_model.json')
    times['xgb']={'seconds':time.perf_counter()-start,'best_iteration_zero_based':int(xm.best_iteration),
                  'saved_trees':int(xm.get_booster().num_boosted_rounds()),'time_stopped':xtime.stopped,
                  'curve':xm.evals_result()}
    if time.perf_counter()>=deadline:raise TimeoutError('Study fit budget exhausted before LightGBM')
    start=time.perf_counter();ltime=LTime(min(deadline,start+recipe['fit_budget_seconds']))
    lm=lgb.LGBMClassifier(n_estimators=recipe['lgb_cap'],learning_rate=recipe['learning_rate'],num_leaves=recipe['num_leaves'],
                         subsample=.8,subsample_freq=1,colsample_bytree=.8,random_state=378,device='cpu',max_bin=127,
                         n_jobs=2,verbosity=-1)
    lm.fit(xtrain,ytrain,eval_set=[(xsel,ysel)],callbacks=[lgb.early_stopping(recipe['lgb_patience'],verbose=False),ltime])
    joblib.dump(lm,directory/'lgb_model.pkl')
    times['lgb']={'seconds':time.perf_counter()-start,'best_iteration_one_based':int(lm.best_iteration_),
                  'saved_trees':int(lm.booster_.num_trees()),'time_stopped':ltime.stopped,'curve':lm.evals_result_}
    px,pl=xm.predict_proba(xsel)[:,1],lm.predict_proba(xsel)[:,1]
    return xm,lm,px,pl,times


def train_stylometric(repo_root, manifest_path, output_dir):
    import shutil
    import warnings
    import joblib
    import numpy as np
    import scipy.sparse as sp
    from .data import load_manifest,load_records
    start=time.perf_counter();deadline=start+STUDY_PLAN['total_budget_seconds']
    out=Path(output_dir);study=out.parent/(out.name+'-study')
    if (out.exists() and any(out.iterdir())) or study.exists():raise ValueError('New model and sibling study directories are required')
    out.mkdir(parents=True,exist_ok=True);study.mkdir()
    _write_json(study/'predeclared-plan.json',STUDY_PLAN)
    source=_training_sources(repo_root);manifest=load_manifest(manifest_path)
    train=load_records(repo_root,manifest,'train');selection=load_records(repo_root,manifest,'selection')
    source_manifest=sha256_file(manifest_path)
    provenance=_cpu_provenance();_write_json(study/'cpu-environment.json',provenance)
    _write_json(study/'input-binding.json',{'source_snapshot':source,'manifest_file_sha256':source_manifest,
                 'train_count':len(train),'selection_count':len(selection),'train_records_sha256':_records_hash(train),
                 'selection_records_sha256':_records_hash(selection),'calibration_evaluated':False,'locked_holdout_evaluated':False})
    features=PairFeatureExtractor().fit(train)
    proof=_idf_proof(features,train);_write_json(study/'train-only-tfidf-proof.json',proof)
    full_train,full_sel=features.transform(train),features.transform(selection)
    yt=np.array([r['label'] for r in train]);ys=np.array([r['label'] for r in selection])
    reports=[]
    for recipe in STUDY_PLAN['recipes']:
        if _training_sources(repo_root)!=source or sha256_file(manifest_path)!=source_manifest:
            raise RuntimeError('ML study source or split changed during fitting')
        mode=recipe['feature_mode'];name=recipe['name']
        if mode=='full':xt,xs=full_train,full_sel
        elif mode=='style_only':xt,xs=full_train[:,-74:-2],full_sel[:,-74:-2]
        else:xt,xs=(sp.hstack([m[:,:-74],m[:,-2:]],format='csr') for m in (full_train,full_sel))
        print(json.dumps({'stage':'ml-recipe-start','recipe':name,'features':xt.shape[1]}),flush=True)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='X does not have valid feature names',category=UserWarning)
                xm,lm,px,pl,times=_fit_pair(xt,yt,xs,ys,recipe,deadline,study/name)
            scores=selection_weights(ys,px,pl)
            report={'name':name,'status':'completed','feature_mode':mode,'feature_count':int(xt.shape[1]),
                    'recipe':recipe,'components':times,'weight_study':scores}
            _write_json(study/name/'report.json',report)
            (study/name/'selection-scores.jsonl').write_text(''.join(json.dumps({'id':r['id'],'label':int(r['label']),
                    'xgb_probability':float(a),'lgb_probability':float(b)})+'\n' for r,a,b in zip(selection,px,pl)),encoding='utf8',newline='\n')
            del xm,lm
        except TimeoutError as error:
            report={'name':name,'status':'incomplete_budget','feature_mode':mode,'reason':str(error)}
            if name=='full-pilot':raise RuntimeError('Required full-feature pilot did not finish') from error
        reports.append(report);_write_json(study/'selection-study.json',{'protocol':STUDY_PLAN,'recipes':reports,'scope':'selection only'})
        print(json.dumps({'stage':'ml-recipe-complete','recipe':name,'status':report['status'],
                         'best_macro_f1':report.get('weight_study',{}).get('best_macro_f1')}),flush=True)
        del xt,xs;gc.collect()
    chosen=choose_recipe(reports);selected=next(r for r in reports if r['name']==chosen['recipe'])
    for name in ('xgb_model.json','lgb_model.pkl'):shutil.copyfile(study/chosen['recipe']/name,out/name)
    for name,obj in [('tfidf_word.pkl',features.word),('tfidf_char.pkl',features.char),('tfidf_fw.pkl',features.function)]:joblib.dump(obj,out/name)
    config={'schema_version':1,'xgb_weight':chosen['xgb_weight'],'lgb_weight':1-chosen['xgb_weight'],'threshold':.5,
            'seed':378,'feature_mode':chosen['feature_mode'],'feature_count':selected['feature_count'],
            'tfidf_fit_split':'train','validation_split':'selection','selection_macro_f1':chosen['candidate_macro_f1'] if chosen['promotion_passed'] else chosen['reference_macro_f1'],
            'manifest_sha256':source_manifest,'training_device':'cpu','training_environment':provenance,
            'selected_recipe':chosen['recipe'],'selection_decision':chosen,'training_seconds':time.perf_counter()-start,
            'selected_recipe_parameters':selected['recipe'],'training_source_snapshot':source,
            'authors':['Dongmyung Park','Juho Kim'],'study_protocol':STUDY_PLAN['protocol'],
            'xgb_best_iteration':selected['components']['xgb']['best_iteration_zero_based'],
            'lgb_best_iteration':selected['components']['lgb']['best_iteration_one_based']}
    _write_json(out/'ensemble_config.json',config)
    write_artifact_inventory(out)
    runner=StylometricRunner(out)
    ab=runner.predict_proba(selection).ravel();ba=runner.predict_proba(selection,orientation='ba').ravel()
    raw=[json.loads(line) for line in (study/chosen['recipe']/'selection-scores.jsonl').read_text().splitlines()]
    expected=np.array([config['xgb_weight']*r['xgb_probability']+config['lgb_weight']*r['lgb_probability'] for r in raw])
    if not np.allclose(ab,expected,rtol=0,atol=1e-12) or not np.allclose(ab,ba,rtol=0,atol=1e-12):
        raise RuntimeError('Saved ensemble reload or input-swap invariance failed')
    if _training_sources(repo_root)!=source or sha256_file(manifest_path)!=source_manifest:
        raise RuntimeError('ML study source or split changed before completion')
    _write_json(study/'completion.json',{'status':'completed','selection_decision':chosen,'recipes':reports,
                 'source_snapshot':source,'manifest_file_sha256':source_manifest,'elapsed_seconds':time.perf_counter()-start,
                 'saved_artifacts':verify_artifact_inventory(out),'selection_ab_ba_max_gap':float(np.max(abs(ab-ba))),
                 'selection_reload_max_gap':float(np.max(abs(ab-expected))),
                 'calibration_evaluated':False,'locked_holdout_evaluated':False,'training_environment':provenance})
    return config
