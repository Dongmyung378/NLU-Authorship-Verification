"""Requested CPU budgets affect loaded estimators, not frozen artifact bytes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# Match the public CPU bootstrap: Torch's pinned Linux wheel loads its bundled
# OpenMP runtime before LightGBM, including on minimal Ubuntu without libgomp.
import torch
import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb

from nlu_authorship.data import build_split_manifest
from nlu_authorship.evaluation import artifact_inventory, fit_model_calibration, freeze_evaluation
from nlu_authorship.models import write_artifact_inventory
from nlu_authorship.runtime import Runtime
from nlu_authorship.stylometric import PairFeatureExtractor, load_stylometric
from nlu_fixtures import write_dataset


class RuntimeThreadTests(unittest.TestCase):
    def test_fresh_public_cpu_bootstrap_loads_native_lightgbm_without_system_install(self):
        """A fresh subprocess prevents this test's Torch import hiding failures."""
        code = '''
import json
import numpy as np
from nlu_authorship.runtime import configure_cpu
configuration=configure_cpu(1)
import lightgbm as lgb
model=lgb.LGBMClassifier(n_estimators=2,num_leaves=3,min_child_samples=1,n_jobs=1,verbosity=-1)
x=np.array([[0.],[1.],[2.],[3.]])
model.fit(x,np.array([0,0,1,1]))
p=model.predict_proba(x)
assert p.shape==(4,2) and np.isfinite(p).all()
print(json.dumps({'device':configuration['device'],'threads':configuration['threads'],
                  'lightgbm':lgb.__version__,'native_probabilities_finite':True}))
'''
        env = os.environ.copy()
        for key in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV'):
            env.pop(key, None)
        env.update(CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', PYTHONNOUSERSITE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        result = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-c', code],
                                cwd=Path(__file__).resolve().parents[1], env=env,
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        proof = json.loads(result.stdout)
        self.assertEqual(proof['device'], 'cpu')
        self.assertEqual(proof['threads'], 1)
        self.assertEqual(proof['lightgbm'], '4.6.0')
        self.assertTrue(proof['native_probabilities_finite'])

    def test_saved_tiny_ensemble_honors_threads_without_mutating_frozen_files(self):
        """Real tiny CPU fits and reloads provide no project-performance claim."""
        original_threads = torch.get_num_threads()
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                repo = write_dataset(root/'repo')
                manifest = build_split_manifest(repo)
                manifest_path = root/'split-manifest.json'
                manifest_path.write_text(json.dumps(manifest)+'\n')
                model = root/'stylometric'
                model.mkdir()
                fixture = [
                    {'text_1':'The quiet river was blue and calm today.',
                     'text_2':'We walked slowly beside the calm water today.'},
                    {'text_1':'I have a book and read it every morning.',
                     'text_2':'They have a garden with green trees and flowers.'},
                    {'text_1':'The train arrived early and we boarded together.',
                     'text_2':'The railway journey was pleasant and very quiet.'},
                    {'text_1':'She carefully examined the design before the meeting.',
                     'text_2':'He quickly prepared a meal in the small kitchen.'}]
                extractor = PairFeatureExtractor().fit(fixture)
                features = extractor.transform(fixture)
                labels = np.array([1, 0, 1, 0])
                xm = xgb.XGBClassifier(n_estimators=3, max_depth=1, tree_method='hist',
                                       device='cpu', n_jobs=2, random_state=378)
                xm.fit(features, labels)
                xm.save_model(model/'xgb_model.json')
                lm = lgb.LGBMClassifier(n_estimators=3, num_leaves=3, min_child_samples=1,
                                        n_jobs=2, verbosity=-1, random_state=378)
                lm.fit(features, labels)
                joblib.dump(lm, model/'lgb_model.pkl')
                for filename, vector in [('tfidf_word.pkl', extractor.word),
                                         ('tfidf_char.pkl', extractor.char),
                                         ('tfidf_fw.pkl', extractor.function)]:
                    joblib.dump(vector, model/filename)
                (model/'ensemble_config.json').write_text(json.dumps({'xgb_weight': .5,
                    'lgb_weight': .5, 'feature_mode': 'full'})+'\n')
                write_artifact_inventory(model)
                runner = load_stylometric(model)
                calibration = fit_model_calibration(runner, repo, manifest, model)
                freeze = root/'evaluation-freeze.json'
                freeze_evaluation({'stylometric': model}, manifest, {},
                                  {'stylometric': calibration}, freeze)
                before = artifact_inventory(model)
                freeze_before = freeze.read_bytes()
                expected = None
                for threads in (1, 2):
                    runtime = Runtime(model, 'stylometric', threads, model_name='stylometric',
                                      freeze_path=freeze, manifest_path=manifest_path)
                    self.assertEqual(runtime.backend.xgb.get_params()['n_jobs'], threads)
                    self.assertEqual(runtime.backend.lgb.get_params()['n_jobs'], threads)
                    generic = json.loads(runtime.backend.xgb.get_booster().save_config())['learner']['generic_param']
                    self.assertEqual(int(generic['nthread']), threads)
                    with patch.object(runtime.backend.lgb.booster_, 'predict',
                                      wraps=runtime.backend.lgb.booster_.predict) as native_predict:
                        prediction = runtime.predict_pair(fixture[0]['text_1'], fixture[0]['text_2'])
                    self.assertEqual(native_predict.call_args.kwargs['num_threads'], threads)
                    if expected is None:
                        expected = prediction
                    else:
                        self.assertEqual(prediction['prediction'], expected['prediction'])
                        self.assertAlmostEqual(prediction['same_author_score'], expected['same_author_score'], places=12)
                    self.assertEqual(artifact_inventory(model), before)
                    self.assertEqual(freeze.read_bytes(), freeze_before)
        finally:
            torch.set_num_threads(original_threads)
