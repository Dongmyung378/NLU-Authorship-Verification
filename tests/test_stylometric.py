"""CPU-only representation, bounded fit and saved ensemble contracts."""
from pathlib import Path
import tempfile
import time
import unittest

import joblib
import numpy as np

from nlu_authorship.models import write_artifact_inventory
from nlu_authorship.stylometric import (PairFeatureExtractor,StylometricRunner,STUDY_PLAN,
    _fit_pair,_idf_proof,choose_recipe,selection_weights)


def passages():
    return [{'id':f'pair-{i}','label':i%2,
             'text_1':f'The river was calm and blue today number {i}.',
             'text_2':('The quiet river moved through the garden today.' if i%2 else
                       'WHY DID THE BOOK ARRIVE SO LATE AGAIN!!!')} for i in range(24)]


class RepresentationTests(unittest.TestCase):
    def test_ablation_blocks_equal_original_feature_subsets_and_swaps(self):
        rows=passages();f=PairFeatureExtractor().fit(rows)
        full=f.transform(rows)
        f.feature_mode='style_only';style=f.transform(rows)
        self.assertEqual(style.shape[1],72)
        self.assertEqual((style!=full[:,-74:-2]).nnz,0)
        f.feature_mode='tfidf_only';tf=f.transform(rows)
        self.assertEqual(tf.shape[1],full.shape[1]-72)
        self.assertEqual((tf[:,:-2]!=full[:,:-74]).nnz,0)
        self.assertEqual((tf[:,-2:]!=full[:,-2:]).nnz,0)
        swapped=[{**r,'text_1':r['text_2'],'text_2':r['text_1']} for r in rows]
        for mode in ['full','tfidf_only','style_only']:
            f.feature_mode=mode
            self.assertEqual((f.transform(rows)!=f.transform(swapped)).nnz,0)
            self.assertEqual(f.transform([]).shape[1],f.transform(rows).shape[1])

    def test_train_only_idf_is_independently_verified_and_corruption_rejected(self):
        rows=passages();f=PairFeatureExtractor().fit(rows)
        self.assertTrue(_idf_proof(f,rows)['passed'])
        f.word.idf_=f.word.idf_+.02
        with self.assertRaisesRegex(RuntimeError,'train-only'):_idf_proof(f,rows)

    def test_selection_vocabulary_is_not_added(self):
        f=PairFeatureExtractor().fit(passages())
        before=dict(f.word.vocabulary_)
        f.transform([{'text_1':'unobservedselectiontoken is unique here.','text_2':'A quiet river crosses this garden.'}])
        self.assertEqual(before,f.word.vocabulary_)
        self.assertNotIn('unobservedselectiontoken',f.word.vocabulary_)


class DecisionTests(unittest.TestCase):
    def test_single_components_are_diagnostic_and_exact_ties_prefer_half(self):
        p=np.array([.1,.9,.1,.9]);y=np.array([0,1,0,1])
        result=selection_weights(y,p,p)
        self.assertEqual(result['best_nonzero_weight'],.5)
        self.assertEqual([r['xgb_weight'] for r in result['weights'] if r['diagnostic_only']],[0.,1.])

    def test_gain_gate_and_full_representation_tie_are_predeclared(self):
        def report(name,mode,score,weight=.5):
            return {'name':name,'status':'completed','feature_mode':mode,
                    'weight_study':{'best_macro_f1':score,'best_nonzero_weight':weight,
                                    'weights':[{'xgb_weight':.5,'metrics':{'macro_f1':.7}}]}}
        base=report('full-pilot','full',.7)
        self.assertEqual(choose_recipe([base,report('tfidf-pilot','tfidf_only',.7019)])['recipe'],'full-pilot')
        chosen=choose_recipe([base,report('tfidf-pilot','tfidf_only',.71),report('full-original-caps','full',.71)])
        self.assertEqual(chosen['recipe'],'full-original-caps')
        self.assertTrue(chosen['promotion_passed'])


class SavedFitTests(unittest.TestCase):
    def test_actual_small_cpu_fit_reloads_every_representation(self):
        rows=passages();f=PairFeatureExtractor().fit(rows)
        recipe={**STUDY_PLAN['recipes'][0],'xgb_cap':8,'lgb_cap':8,'max_depth':2,'num_leaves':3,
                'xgb_patience':3,'lgb_patience':3,'fit_budget_seconds':30}
        for mode in ['full','tfidf_only','style_only']:
            f.feature_mode=mode;x=f.transform(rows);y=np.array([r['label'] for r in rows])
            with tempfile.TemporaryDirectory() as d:
                model=Path(d)/'model'
                xm,lm,px,pl,report=_fit_pair(x[:16],y[:16],x[16:].copy(),y[16:],recipe,time.perf_counter()+60,model)
                self.assertEqual(xm.get_params()['n_jobs'],2)
                self.assertEqual(lm.get_params()['n_jobs'],2)
                self.assertEqual(xm.get_booster().num_boosted_rounds(),xm.best_iteration+1)
                for name,value in [('tfidf_word.pkl',f.word),('tfidf_char.pkl',f.char),('tfidf_fw.pkl',f.function)]:joblib.dump(value,model/name)
                (model/'ensemble_config.json').write_text('{"xgb_weight":0.5,"lgb_weight":0.5,"feature_mode":"'+mode+'"}')
                write_artifact_inventory(model);runner=StylometricRunner(model)
                np.testing.assert_allclose(runner.predict_proba(rows[16:]).ravel(),.5*px+.5*pl,rtol=0,atol=1e-12)
                np.testing.assert_allclose(runner.predict_proba(rows),runner.predict_proba(rows,orientation='ba'),rtol=0,atol=1e-12)

    def test_time_stops_keep_reloadable_best_checkpoints(self):
        rows=passages();f=PairFeatureExtractor().fit(rows);x=f.transform(rows);y=np.array([r['label'] for r in rows])
        recipe={**STUDY_PLAN['recipes'][0],'xgb_cap':50,'lgb_cap':50,'max_depth':2,'num_leaves':3,
                'fit_budget_seconds':1e-9}
        with tempfile.TemporaryDirectory() as d:
            xm,lm,px,pl,report=_fit_pair(x[:16],y[:16],x[16:].copy(),y[16:],recipe,time.perf_counter()+60,Path(d)/'fit')
            self.assertTrue(report['xgb']['time_stopped'])
            self.assertTrue(report['lgb']['time_stopped'])
            self.assertEqual(xm.get_booster().num_boosted_rounds(),xm.best_iteration+1)
            self.assertEqual(lm.booster_.num_trees(),lm.best_iteration_)
            self.assertTrue(np.isfinite(px).all() and np.isfinite(pl).all())


if __name__=='__main__':unittest.main()
