"""Build train_01_stylometric_ml.ipynb: unified stylometric ML training & evaluation notebook."""
from __future__ import annotations

import json
from pathlib import Path


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


CELLS = [
    md_cell(r'''# 01. Stylometric ML Authorship Verification

전통 문체학(Stylometry) 기반 순수 머신러닝 트랙의 최종 학습·평가·모델 패키징 노트북입니다.
`benchmark_1024.zip`만 업로드하고 위에서 아래로 한 번 실행하면 최종 10-전문가 앙상블 + OOD 레지스터 게이트 보정기 학습, Locked Test 평가, 배포용 `.joblib` 생성까지 끝납니다.

- 권장 환경: CPU 런타임 (8+ vCPU, 32 GB RAM 권장)
- 입력 파일: `benchmark_1024.zip`
- 주요 출력: `model_01_stylometric_ml.joblib`, `results_01_stylometric_ml.zip`
'''),
    md_cell("## 1. Environment & Configuration"),
    code_cell(r'''import os
import subprocess
import sys

REQUIRED_PKGS = [
    "lightgbm>=4.3.0",
    "catboost>=1.2.5",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "nltk>=3.8.1",
    "joblib>=1.3.0",
    "threadpoolctl>=3.1.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
    "psutil>=5.9.0",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *REQUIRED_PKGS])

import copy
import gc
import json
import math
import random
import re
import tempfile
import time
import warnings
import zipfile
import zlib
from collections import Counter
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import seaborn as sns
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
sns.set_theme(style="whitegrid")

for tagger_res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
    try:
        nltk.data.find(f"taggers/{tagger_res}")
    except LookupError:
        nltk.download(tagger_res, quiet=True)

SEED = 378
N_JOBS = max(1, min(os.cpu_count() or 4, 16))
random.seed(SEED)
np.random.seed(SEED)

CONFIG = {
    "run_final_test": True,          # set False during validation-only tuning
    "force_recompute_features": False,
    "pruned_lgb_k": 1500,
    "cb_dense_k": 320,
    "cb_sparse_k": 330,
    "genre_blind_sparse_k": 240,
    "cross_topic_weight": 2.5,
    "max_abstention_pct": 15.0,
    "csls_grid": [0.15, 0.25, 0.35, 0.50, 0.65],
    "lgb_full": {
        "learning_rate": 0.025, "num_leaves": 95, "min_child_samples": 50,
        "colsample_bytree": 0.70, "reg_alpha": 2.0, "reg_lambda": 5.0,
    },
    "lgb_pruned": {
        "learning_rate": 0.025, "num_leaves": 55, "max_depth": 8, "min_child_samples": 90,
        "colsample_bytree": 0.55, "reg_alpha": 3.0, "reg_lambda": 10.0,
    },
    "cat_full": {"learning_rate": 0.035, "depth": 6, "l2_leaf_reg": 6.0},
}

print(f"python={sys.version.split()[0]} cpu={os.cpu_count()} n_jobs={N_JOBS} ram_gb={psutil.virtual_memory().total / 1024**3:.1f}")
'''),
    md_cell("## 2. Data Loading & Pair Expansion"),
    code_cell(r'''REQUIRED_FILES = [
    "train_pairs.csv", "closed_anchors.csv",
    "val_closed_queries.csv", "test_closed_queries.csv",
    "val_open_cross.csv", "test_open_cross.csv",
]

extract_dir = Path.cwd() / "benchmark_1024"
if not all((extract_dir / fn).exists() for fn in REQUIRED_FILES):
    for zp in (Path.cwd() / "benchmark_1024.zip", Path("/content/benchmark_1024.zip"), Path.home() / "benchmark_1024.zip"):
        if zp.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "r") as zf:
                for m in zf.infolist():
                    if not m.is_dir():
                        bn = Path(m.filename).name
                        if bn in REQUIRED_FILES or bn.endswith(".json"):
                            (extract_dir / bn).write_bytes(zf.read(m))
            break

DATA_DIR = next(
    (
        c.resolve()
        for c in (extract_dir, Path.cwd(), Path("/content/benchmark_1024"), Path("../../data/benchmark_1024"))
        if all((c / fn).exists() for fn in REQUIRED_FILES)
    ),
    None,
)
if DATA_DIR is None:
    raise FileNotFoundError("benchmark_1024 files not found. Upload benchmark_1024.zip to the working directory.")

ROOT_OUT = (Path.cwd() / "nlu_outputs").resolve()
OUTPUT_DIR = ROOT_OUT / "01_stylometric_ml"
CACHE_DIR = OUTPUT_DIR / "feature_cache"
FIG_DIR = OUTPUT_DIR / "figures"
for d in (OUTPUT_DIR, CACHE_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(DATA_DIR / "train_pairs.csv").reset_index(drop=True)
anchors_df = pd.read_csv(DATA_DIR / "closed_anchors.csv").reset_index(drop=True)
val_closed_q_df = pd.read_csv(DATA_DIR / "val_closed_queries.csv").reset_index(drop=True)
test_closed_q_df = pd.read_csv(DATA_DIR / "test_closed_queries.csv").reset_index(drop=True)
val_open_df = pd.read_csv(DATA_DIR / "val_open_cross.csv").reset_index(drop=True)
test_open_df = pd.read_csv(DATA_DIR / "test_open_cross.csv").reset_index(drop=True)

# verify zero author overlap across open splits
tr_auths = set(train_df["author_1"].astype(str)) | set(train_df["author_2"].astype(str))
va_auths = set(val_open_df["author_1"].astype(str)) | set(val_open_df["author_2"].astype(str))
te_auths = set(test_open_df["author_1"].astype(str)) | set(test_open_df["author_2"].astype(str))
assert len(tr_auths & va_auths) == 0 and len(tr_auths & te_auths) == 0 and len(va_auths & te_auths) == 0


def expand_closed_pairs(queries_df: pd.DataFrame, anc_df: pd.DataFrame) -> pd.DataFrame:
    anc_by_dom = {dom: grp.reset_index(drop=True) for dom, grp in anc_df.groupby("cohort_domain", sort=False)}
    rows = []
    for q in queries_df.itertuples():
        for a in anc_by_dom[q.cohort_domain].itertuples():
            rows.append({
                "query_id": int(q.query_id),
                "cohort_domain": q.cohort_domain,
                "true_author_id": q.true_author_id,
                "candidate_author_id": a.candidate_author_id,
                "anchor_idx": int(a.anchor_idx),
                "text_1": str(q.text),
                "text_2": str(a.text),
                "label": int(q.true_author_id == a.candidate_author_id),
            })
    return pd.DataFrame(rows)


def build_anchor_hub_pairs(anc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dom, grp in anc_df.groupby("cohort_domain", sort=False):
        items = list(grp.itertuples())
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i].candidate_author_id != items[j].candidate_author_id:
                    rows.append({
                        "cohort_domain": dom,
                        "cand_a": items[i].candidate_author_id,
                        "cand_b": items[j].candidate_author_id,
                        "text_1": str(items[i].text),
                        "text_2": str(items[j].text),
                    })
    return pd.DataFrame(rows)


val_closed_pairs_df = expand_closed_pairs(val_closed_q_df, anchors_df)
test_closed_pairs_df = expand_closed_pairs(test_closed_q_df, anchors_df)
anchor_hub_pairs_df = build_anchor_hub_pairs(anchors_df)

y_train = train_df["label"].to_numpy(dtype=np.int32)
y_val_open = val_open_df["label"].to_numpy(dtype=np.int32)
y_test_open = test_open_df["label"].to_numpy(dtype=np.int32)

print(
    f"data_dir={DATA_DIR} | train={len(train_df):,} val_open={len(val_open_df):,} "
    f"test_open={len(test_open_df):,} val_closed={len(val_closed_pairs_df):,} test_closed={len(test_closed_pairs_df):,}"
)
'''),
    md_cell(r'''## 3. Symmetric Stylometric Feature Extractor

문서 쌍 `(text_1, text_2)`에 대해 순서 대칭(order-invariant) 6,001차원 피처를 추출합니다.
- Stage-1 OOF 선형/SVM 마진 및 뷰 간 괴리도 (Surface 60k / POS-Noise 40k / Distorted 30k, 5-Bucket 양방향 저자 배제 OOF)
- 학습 사칭자 256편 기반 AS-Norm, Top-10 마진, Koppel 64-bag 승률
- zlib 기반 대칭 압축 거리 (NCD, CBC)
- Gram2Vec 품사·형태소·가독성·호흡 변동성 62종 문서 프로필의 `|diff|`, 상대차, 평균
- 기능어 250종 z-score 편차 및 Burrows / Argamon / Wurzburg / JSD 거리
- 문자 및 POS-Noise n-gram 희소 차이(`|v1 - v2|`) 및 교차곱(`v1 * v2`)
'''),
    code_cell(r'''FUNCTION_WORDS_250 = list(dict.fromkeys([
    "i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "we", "us", "our", "ours", "ourselves", "they", "them", "their", "theirs",
    "themselves", "who", "whom", "whose", "which", "what", "that", "this", "these", "those",
    "one", "ones",
    "all", "another", "any", "anybody", "anyone", "anything", "both", "each", "either", "enough",
    "every", "everybody", "everyone", "everything", "few", "fewer", "less", "little", "many", "more",
    "most", "much", "neither", "nobody", "none", "no", "nothing", "other", "others", "several",
    "some", "somebody", "someone", "something", "such", "own",
    "a", "an", "the", "about", "above", "across", "after", "against", "along", "amid",
    "among", "around", "as", "at", "before", "behind", "below", "beneath", "beside", "besides",
    "between", "beyond", "by", "concerning", "considering", "despite", "down", "during", "except", "for",
    "from", "in", "inside", "into", "like", "near", "of", "off", "on", "onto",
    "out", "outside", "over", "past", "regarding", "round", "since", "through", "throughout", "till",
    "to", "toward", "towards", "under", "underneath", "until",
    "unto", "up", "upon", "versus", "via", "with", "within", "without", "and", "but",
    "or", "nor", "yet", "so", "although", "though", "because", "unless", "whether", "while",
    "whilst", "whereas", "whereby", "wherein", "wherever", "where", "when", "whenever", "if", "once",
    "than", "lest",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "having", "do", "does", "did", "doing", "done", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "ought", "need", "dare", "used",
    "get", "gets", "got",
    "not", "very", "too", "quite", "rather", "almost", "just", "only", "even", "still",
    "already", "also", "again", "ever", "never", "always", "often", "sometimes", "usually", "perhaps",
    "maybe", "however", "therefore", "thus", "hence", "moreover", "furthermore", "nevertheless", "nonetheless", "otherwise",
    "instead",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't",
    "won't", "wouldn't", "can't", "cannot", "couldn't", "shouldn't", "i'm", "i've", "i'd", "it's",
]))
assert len(FUNCTION_WORDS_250) == 250
FW_SET = frozenset(FUNCTION_WORDS_250)

BE_VERBS = frozenset(["am", "is", "are", "was", "were", "be", "been", "being"])
FIRST_PERSON_SING = frozenset(["i", "me", "my", "mine", "myself", "i'm", "i've", "i'd"])
UK_MARKERS = frozenset(["colour", "behaviour", "favour", "honour", "labour", "neighbour", "centre", "theatre", "organisation", "realise", "recognise", "whilst", "amongst", "towards"])
US_MARKERS = frozenset(["color", "behavior", "favor", "honor", "labor", "neighbor", "center", "theater", "organization", "realize", "recognize", "while", "among", "toward"])
SUBORD_WORDS = frozenset([
    "although", "though", "because", "unless", "whether", "while", "whilst",
    "whereas", "whereby", "wherein", "wherever", "if", "once", "since", "until", "lest",
])
MORPH_SUFFIXES = (
    "tion", "sion", "ment", "ness", "able", "ible", "ingly", "edly",
    "ity", "ive", "ous", "ing", "ize", "ist", "est", "ed", "ly", "al", "er",
)
NOMINAL_SUFFIXES = ("tion", "sion", "ment", "ness", "ity")

TOKEN_SCAN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")
WORD_ONLY_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
VOWEL_GROUP_RE = re.compile(r"[aeiouyAEIOUY]+")
DIGIT_RE = re.compile(r"\d")
SENT_SPLIT_RE = re.compile(r"[.!?]+")

DOC_PROFILE_62_NAMES = [
    "g2v_pos_nn", "g2v_pos_nnp", "g2v_pos_vb_pres", "g2v_pos_vbd_vbn", "g2v_pos_vbg", "g2v_pos_md",
    "g2v_pos_jj", "g2v_pos_rb", "g2v_pos_prp", "g2v_pos_in_to", "g2v_pos_dt", "g2v_pos_cc_part",
    "g2v_morph_ing", "g2v_morph_ed", "g2v_morph_ly", "g2v_morph_nominal",
    "g2v_passive_rate", "g2v_subord_rate", "g2v_sent_init_conj_adv", "g2v_contraction_rate",
    "g2v_pos_bigram_entropy", "g2v_bg_jj_nn", "g2v_bg_rb_vbjj", "g2v_bg_nn_nn",
    "g2v_bg_in_dt", "g2v_bg_prp_vb", "g2v_oxford_clause_comma", "g2v_newline_per100w",
    "lex_mattr_50", "lex_yules_k", "lex_simpsons_d", "lex_honores_r",
    "surf_char_len", "surf_word_len", "surf_sent_count",
    "surf_mean_word_len", "surf_std_word_len", "surf_short_word_ratio", "surf_long_word_ratio",
    "surf_mean_sent_words", "surf_std_sent_words", "surf_char_entropy",
    "surf_upper_ratio", "surf_digit_ratio", "surf_allcaps_ratio",
    "punct_comma_per1k", "punct_period_per1k", "punct_excl_per1k", "punct_quest_per1k",
    "punct_semi_colon_per1k", "punct_quote_per1k", "punct_dash_paren_per1k",
    "bio_sent_burstiness", "bio_word_burstiness",
    "bio_syllables_per_word", "bio_polysyllabic_ratio", "bio_coleman_liau_idx",
    "bio_mid_sent_cap_ratio", "bio_first_person_sing_ratio",
    "bio_saxon_genitive_pref", "bio_uk_vs_us_orthography", "bio_double_space_blankline",
]
assert len(DOC_PROFILE_62_NAMES) == 62

_WORKER_TAGGER = None


def _get_worker_tagger():
    global _WORKER_TAGGER
    if _WORKER_TAGGER is None:
        from nltk.tag.perceptron import PerceptronTagger
        _WORKER_TAGGER = PerceptronTagger()
    return _WORKER_TAGGER


def _count_syllables(word: str) -> int:
    w = word.lower()
    if len(w) <= 3:
        return 1
    groups = len(VOWEL_GROUP_RE.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(groups, 1)


def _compute_mattr(lower_words: list[str], window: int = 50) -> float:
    n = len(lower_words)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(lower_words)) / n
    counts: dict[str, int] = {}
    for w in lower_words[:window]:
        counts[w] = counts.get(w, 0) + 1
    unique_sum = len(counts)
    for i in range(window, n):
        w_out, w_in = lower_words[i - window], lower_words[i]
        c_out = counts[w_out] - 1
        if c_out == 0:
            del counts[w_out]
        else:
            counts[w_out] = c_out
        counts[w_in] = counts.get(w_in, 0) + 1
        unique_sum += len(counts)
    return unique_sum / ((n - window + 1) * window)


def _extract_doc_views_chunk(texts: list[str]) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    tagger = _get_worker_tagger()
    pos_noise_list: list[str] = []
    distort_list: list[str] = []
    comp_lens = np.zeros((len(texts), 2), dtype=np.float32)
    profiles = np.zeros((len(texts), 62), dtype=np.float32)

    def _distort_repl(m: re.Match) -> str:
        tok = m.group(0)
        low = tok.lower()
        if low in FW_SET:
            return low
        return f"{'C' if tok[0].isupper() else 'w'}{min(len(tok), 14)}"

    for idx, text in enumerate(texts):
        c_len = len(text)
        if c_len == 0:
            pos_noise_list.append("")
            distort_list.append("")
            continue

        distort_list.append(WORD_ONLY_RE.sub(_distort_repl, DIGIT_RE.sub("0", text)))

        raw_spans = TOKEN_SCAN_RE.findall(text)
        word_positions, word_tokens = [], []
        for pos_i, tok in enumerate(raw_spans):
            if tok[0].isalpha():
                word_positions.append(pos_i)
                word_tokens.append(tok)

        tagged_words = tagger.tag(word_tokens) if word_tokens else []
        n_words = len(word_tokens)

        pos_lookup = {}
        pos_seq = []
        c_nn = c_nnp = c_vbp = c_vbd = c_vbg = c_md = 0
        c_jj = c_rb = c_prp = c_into = c_dt = c_cc = 0
        m_ing = m_ed = m_ly = m_nom = c_contract = c_subord = 0
        c_fp1 = c_saxon = c_of = c_uk = c_us = 0

        for w_pos, (w_tok, p_tag) in zip(word_positions, tagged_words):
            pos_lookup[w_pos] = p_tag
            pos_seq.append(p_tag)
            low = w_tok.lower()

            if "'" in low:
                c_contract += 1
            if low.endswith("'s"):
                c_saxon += 1
            elif low == "of":
                c_of += 1
            if low in FIRST_PERSON_SING:
                c_fp1 += 1
            if low in UK_MARKERS or low.endswith(("our", "ise", "yse")):
                c_uk += 1
            elif low in US_MARKERS or low.endswith(("ize", "yze")):
                c_us += 1
            if low in SUBORD_WORDS:
                c_subord += 1

            if p_tag in ("NN", "NNS"):
                c_nn += 1
            elif p_tag in ("NNP", "NNPS"):
                c_nnp += 1
            elif p_tag in ("VB", "VBP", "VBZ"):
                c_vbp += 1
            elif p_tag in ("VBD", "VBN"):
                c_vbd += 1
            elif p_tag == "VBG":
                c_vbg += 1
            elif p_tag == "MD":
                c_md += 1
            elif p_tag.startswith("JJ"):
                c_jj += 1
            elif p_tag.startswith("RB") or p_tag == "WRB":
                c_rb += 1
            elif p_tag.startswith("PRP") or p_tag.startswith("WP"):
                c_prp += 1
            elif p_tag in ("IN", "TO"):
                c_into += 1
            elif p_tag in ("DT", "PDT", "WDT"):
                c_dt += 1
            elif p_tag in ("CC", "UH", "RP", "EX"):
                c_cc += 1

            if len(low) >= 5:
                if low.endswith("ing"):
                    m_ing += 1
                elif low.endswith("ed"):
                    m_ed += 1
                elif low.endswith("ly"):
                    m_ly += 1
                if low.endswith(NOMINAL_SUFFIXES):
                    m_nom += 1

        pnoise_tokens = []
        oxford_comma_cnt = 0
        for s_i, tok in enumerate(raw_spans):
            if tok[0].isalpha():
                low = tok.lower()
                if low in FW_SET:
                    pnoise_tokens.append(low)
                else:
                    ptag = pos_lookup[s_i]
                    suf_found = ""
                    if len(low) >= 6:
                        for suf in MORPH_SUFFIXES:
                            if low.endswith(suf) and len(low) >= len(suf) + 3:
                                suf_found = f"_{suf}"
                                break
                    pnoise_tokens.append(f"{ptag}{suf_found}")
            elif tok[0].isdigit():
                pnoise_tokens.append("<NUM>")
            else:
                pnoise_tokens.append(f"P_{tok}")
                if tok == "," and s_i + 1 < len(raw_spans) and raw_spans[s_i + 1].lower() in ("and", "but", "or", "yet", "so"):
                    oxford_comma_cnt += 1

        pos_noise_str = " ".join(pnoise_tokens)
        pos_noise_list.append(pos_noise_str)

        comp_lens[idx, 0] = len(zlib.compress(text.encode("utf-8"), level=6))
        comp_lens[idx, 1] = len(zlib.compress(pos_noise_str.encode("utf-8"), level=6))

        passive_cnt = 0
        bg_jj_nn = bg_rb_vj = bg_nn_nn = bg_in_dt = bg_prp_vb = 0
        lower_words = [w.lower() for w in word_tokens]
        if n_words >= 2:
            for i in range(n_words - 1):
                p1, p2 = pos_seq[i], pos_seq[i + 1]
                if p1.startswith("JJ") and p2.startswith("NN"):
                    bg_jj_nn += 1
                if p1.startswith("RB") and (p2.startswith("VB") or p2.startswith("JJ")):
                    bg_rb_vj += 1
                if p1.startswith("NN") and p2.startswith("NN"):
                    bg_nn_nn += 1
                if p1 == "IN" and p2 == "DT":
                    bg_in_dt += 1
                if p1 == "PRP" and (p2.startswith("VB") or p2 == "MD"):
                    bg_prp_vb += 1
                if lower_words[i] in BE_VERBS:
                    if p2 == "VBN" or (i + 2 < n_words and p2.startswith("RB") and pos_seq[i + 2] == "VBN"):
                        passive_cnt += 1
            bg_counts = Counter(zip(pos_seq[:-1], pos_seq[1:]))
            bg_probs = np.fromiter(bg_counts.values(), dtype=np.float32) / (n_words - 1)
            pos_bg_ent = float(-np.sum(bg_probs * np.log2(bg_probs + 1e-12)))
        else:
            pos_bg_ent = 0.0

        sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
        n_sents = max(len(sents), 1)
        sent_w_lens, sent_init_ca, mid_sent_caps = [], 0, 0
        for s in sents:
            sw = WORD_ONLY_RE.findall(s)
            sent_w_lens.append(len(sw))
            if sw:
                if sw[0].lower() in ("and", "but", "so", "yet", "or", "however", "therefore", "moreover", "furthermore", "meanwhile", "nevertheless", "thus"):
                    sent_init_ca += 1
                if len(sw) > 1:
                    mid_sent_caps += sum(1 for wm in sw[1:] if wm[0].isupper() and wm != "I" and not wm.startswith("I'"))

        sent_arr = np.asarray(sent_w_lens, dtype=np.float32) if sent_w_lens else np.zeros(1, dtype=np.float32)
        mean_s, std_s = float(sent_arr.mean()), float(sent_arr.std())
        sent_burst = (std_s - mean_s) / (std_s + mean_s + 1e-6)

        if n_words > 0:
            mattr50 = _compute_mattr(lower_words, window=50)
            vocab_c = Counter(lower_words)
            fof = Counter(vocab_c.values())
            m2 = sum((f * f) * v_i for f, v_i in fof.items())
            yules_k = 1e4 * (m2 - n_words) / (n_words * n_words)
            simpsons_d = sum(f * (f - 1) * v_i for f, v_i in fof.items()) / (n_words * (n_words - 1)) if n_words > 1 else 0.0
            v1 = fof.get(1, 0)
            honores_r = 100.0 * math.log(max(n_words, 2)) / max(1.0 - (v1 / max(len(vocab_c), 1)), 0.05)

            w_lens = np.fromiter((len(w) for w in word_tokens), dtype=np.float32, count=n_words)
            mean_w, std_w = float(w_lens.mean()), float(w_lens.std())
            word_burst = (std_w - mean_w) / (std_w + mean_w + 1e-6)
            short_w = float(np.mean(w_lens <= 3))
            long_w = float(np.mean(w_lens >= 8))
            allcaps_w = sum(1 for w in word_tokens if len(w) >= 2 and w.isupper()) / n_words

            syls = np.fromiter((_count_syllables(w) for w in word_tokens), dtype=np.float32, count=n_words)
            mean_syl = float(syls.mean())
            poly_syl = float(np.mean(syls >= 3))
            coleman_liau = float(np.clip(0.0588 * (float(w_lens.sum()) / n_words * 100.0) - 0.296 * (n_sents / n_words * 100.0) - 15.8, -5.0, 25.0))
        else:
            mattr50 = yules_k = simpsons_d = honores_r = 0.0
            mean_w = std_w = word_burst = short_w = long_w = allcaps_w = 0.0
            mean_syl = poly_syl = coleman_liau = 0.0

        inv_w, inv_s, inv_c1k = 1.0 / max(n_words, 1), 1.0 / n_sents, 1000.0 / c_len
        char_c = Counter(text)
        ch_p = np.fromiter(char_c.values(), dtype=np.float32) / c_len
        ch_ent = float(-np.sum(ch_p * np.log2(ch_p + 1e-12)))

        n_upper = sum(1 for ch in text if ch.isupper())
        n_digit = sum(1 for ch in text if ch.isdigit())
        saxon_pref = c_saxon / max(c_saxon + c_of, 1)
        uk_us_bias = (c_uk - c_us) / max(c_uk + c_us, 1)
        dbl_space_blank = (text.count("  ") + text.count("\n\n")) * 100.0 * inv_w

        profiles[idx] = [
            c_nn * inv_w, c_nnp * inv_w, c_vbp * inv_w, c_vbd * inv_w, c_vbg * inv_w, c_md * inv_w,
            c_jj * inv_w, c_rb * inv_w, c_prp * inv_w, c_into * inv_w, c_dt * inv_w, c_cc * inv_w,
            m_ing * inv_w, m_ed * inv_w, m_ly * inv_w, m_nom * inv_w,
            passive_cnt * inv_s, c_subord * inv_s, sent_init_ca * inv_s, c_contract * inv_w,
            pos_bg_ent, bg_jj_nn * inv_w, bg_rb_vj * inv_w, bg_nn_nn * inv_w,
            bg_in_dt * inv_w, bg_prp_vb * inv_w, oxford_comma_cnt * inv_s, text.count("\n") * 100.0 * inv_w,
            mattr50, yules_k, simpsons_d, honores_r,
            c_len, n_words, n_sents,
            mean_w, std_w, short_w, long_w,
            mean_s, std_s, ch_ent,
            n_upper / c_len, n_digit / c_len, allcaps_w,
            text.count(",") * inv_c1k, text.count(".") * inv_c1k, text.count("!") * inv_c1k, text.count("?") * inv_c1k,
            (text.count(";") + text.count(":")) * inv_c1k,
            (text.count('"') + text.count("'")) * inv_c1k,
            (text.count("-") + text.count("(") + text.count(")")) * inv_c1k,
            sent_burst, word_burst, mean_syl, poly_syl, coleman_liau,
            mid_sent_caps * inv_w, c_fp1 * inv_w, saxon_pref, uk_us_bias, dbl_space_blank,
        ]

    return pos_noise_list, distort_list, comp_lens, profiles


def _pair_ncd_chunk(
    t1_l: list[str], t2_l: list[str],
    p1_l: list[str], p2_l: list[str],
    c1_arr: np.ndarray, c2_arr: np.ndarray,
) -> np.ndarray:
    n = len(t1_l)
    out = np.zeros((n, 6), dtype=np.float32)
    for i in range(n):
        b1, b2 = t1_l[i].encode("utf-8"), t2_l[i].encode("utf-8")
        cr1, cr2 = float(c1_arr[i, 0]), float(c2_arr[i, 0])
        c12 = 0.5 * (len(zlib.compress(b1 + b"\n" + b2, level=6)) + len(zlib.compress(b2 + b"\n" + b1, level=6)))
        ncd_r = (c12 - min(cr1, cr2)) / max(cr1, cr2, 1.0)
        cbc_r = 1.0 - ((cr1 + cr2 - c12) / math.sqrt(max(cr1 * cr2, 1.0)))
        rd_r = abs(cr1 / max(len(b1), 1) - cr2 / max(len(b2), 1))

        bp1, bp2 = p1_l[i].encode("utf-8"), p2_l[i].encode("utf-8")
        cp1, cp2 = float(c1_arr[i, 1]), float(c2_arr[i, 1])
        cp12 = 0.5 * (len(zlib.compress(bp1 + b"\n" + bp2, level=6)) + len(zlib.compress(bp2 + b"\n" + bp1, level=6)))
        ncd_p = (cp12 - min(cp1, cp2)) / max(cp1, cp2, 1.0)
        cbc_p = 1.0 - ((cp1 + cp2 - cp12) / math.sqrt(max(cp1 * cp2, 1.0)))
        rd_p = abs(cp1 / max(len(bp1), 1) - cp2 / max(len(bp2), 1))
        out[i] = [ncd_r, cbc_r, rd_r, ncd_p, cbc_p, rd_p]
    return out


def _sym_sparse_pair(m1: sp.csr_matrix, m2: sp.csr_matrix) -> sp.csr_matrix:
    diff = (m1 - m2).tocsr()
    diff.data = np.abs(diff.data).astype(np.float32)
    prod = m1.multiply(m2).tocsr().astype(np.float32)
    return sp.hstack([diff, prod], format="csr", dtype=np.float32)


def _augment_s1_interactions(s1_10col: np.ndarray) -> np.ndarray:
    m_slr, p_slr = s1_10col[:, 0:1], s1_10col[:, 1:2]
    m_ssvm = s1_10col[:, 2:3]
    m_plr, p_plr = s1_10col[:, 4:5], s1_10col[:, 5:6]
    m_psvm = s1_10col[:, 6:7]
    m_dlr = s1_10col[:, 8:9]
    all_m = np.hstack([m_slr, m_ssvm, m_plr, m_psvm, m_dlr])
    return np.hstack([
        s1_10col,
        m_plr - m_slr, m_psvm - m_ssvm, m_dlr - m_slr, p_plr * p_slr,
        np.mean(all_m, axis=1, keepdims=True), np.std(all_m, axis=1, keepdims=True),
        np.min(all_m, axis=1, keepdims=True), np.max(all_m, axis=1, keepdims=True),
    ]).astype(np.float32)


class StylometricPairExtractor:
    """Symmetric 6,001-dim stylometric feature extractor (501 dense + 5,500 sparse)."""

    def __init__(
        self,
        surf_hi_dim: int = 60000,
        pnoise_hi_dim: int = 40000,
        dist_hi_dim: int = 30000,
        tree_char_dim: int = 3500,
        tree_pnoise_dim: int = 1500,
        tree_char_prod_dim: int = 300,
        tree_pnoise_prod_dim: int = 200,
        n_impostors: int = 256,
        n_koppel_bags: int = 64,
        n_oof_folds: int = 5,
        n_jobs: int = 8,
        seed: int = 378,
    ):
        self.surf_hi_dim = surf_hi_dim
        self.pnoise_hi_dim = pnoise_hi_dim
        self.dist_hi_dim = dist_hi_dim
        self.tree_char_dim = tree_char_dim
        self.tree_pnoise_dim = tree_pnoise_dim
        self.tree_char_prod_dim = tree_char_prod_dim
        self.tree_pnoise_prod_dim = tree_pnoise_prod_dim
        self.n_impostors = n_impostors
        self.n_koppel_bags = n_koppel_bags
        self.n_oof_folds = n_oof_folds
        self.n_jobs = n_jobs
        self.seed = seed

        self.fw_vec = CountVectorizer(lowercase=True, token_pattern=r"[a-z]+(?:'[a-z]+)?", vocabulary=FUNCTION_WORDS_250, dtype=np.float32)
        self.vec_surf_hi = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=surf_hi_dim, sublinear_tf=True, min_df=5, lowercase=False, dtype=np.float32)
        self.vec_pnoise_hi = TfidfVectorizer(analyzer="word", ngram_range=(1, 4), max_features=pnoise_hi_dim, token_pattern=r"\S+", sublinear_tf=True, min_df=5, lowercase=False, dtype=np.float32)
        self.vec_dist_hi = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), max_features=dist_hi_dim, sublinear_tf=True, min_df=5, lowercase=False, dtype=np.float32)
        self.vec_tree_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=tree_char_dim, sublinear_tf=True, min_df=5, lowercase=False, dtype=np.float32)
        self.vec_tree_pnoise = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), max_features=tree_pnoise_dim, token_pattern=r"\S+", sublinear_tf=True, min_df=5, lowercase=False, dtype=np.float32)

        self.s1_models_: dict[str, tuple[object, object | None]] = {}
        self.fw_mean_: np.ndarray | None = None
        self.fw_std_: np.ndarray | None = None
        self.imp_surf_: sp.csr_matrix | None = None
        self.imp_pnoise_: sp.csr_matrix | None = None
        self.imp_fw_unit_: np.ndarray | None = None
        self.koppel_proj_pnoise_: np.ndarray | None = None
        self.koppel_imp_pnoise_: np.ndarray | None = None
        self.koppel_proj_fw_: np.ndarray | None = None
        self.koppel_imp_fw_: np.ndarray | None = None
        self.tree_char_order_: np.ndarray | None = None
        self.tree_pnoise_order_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.group_slices_: dict[str, tuple[int, int]] = {}
        self.n_dense_: int = 0

    def _parallel_views(self, unique_texts: list[str]):
        n_tot = len(unique_texts)
        if n_tot < 800 or self.n_jobs == 1:
            return _extract_doc_views_chunk(unique_texts)
        n_ch = max(self.n_jobs * 2, 2)
        csz = (n_tot + n_ch - 1) // n_ch
        chunks = [unique_texts[i : i + csz] for i in range(0, n_tot, csz)]
        parts = Parallel(n_jobs=self.n_jobs)(delayed(_extract_doc_views_chunk)(ch) for ch in chunks if ch)
        return (
            [x for r in parts for x in r[0]],
            [x for r in parts for x in r[1]],
            np.vstack([r[2] for r in parts]),
            np.vstack([r[3] for r in parts]),
        )

    def _parallel_ncd(self, t1, t2, p1, p2, c1, c2) -> np.ndarray:
        n_tot = len(t1)
        if n_tot < 1200 or self.n_jobs == 1:
            return _pair_ncd_chunk(t1, t2, p1, p2, c1, c2)
        n_ch = max(self.n_jobs * 2, 2)
        csz = (n_tot + n_ch - 1) // n_ch
        tasks = [(t1[i:i+csz], t2[i:i+csz], p1[i:i+csz], p2[i:i+csz], c1[i:i+csz], c2[i:i+csz]) for i in range(0, n_tot, csz)]
        return np.vstack(Parallel(n_jobs=self.n_jobs)(delayed(_pair_ncd_chunk)(*tk) for tk in tasks))

    @staticmethod
    def _fit_lr(X_tr: sp.csr_matrix, y_tr: np.ndarray, seed: int):
        clf = LogisticRegression(C=2.5, solver="lbfgs", max_iter=250, random_state=seed)
        clf.fit(X_tr, y_tr)
        return clf, None

    @staticmethod
    def _fit_svm_with_platt(X_tr: sp.csr_matrix, y_tr: np.ndarray, seed: int):
        svm = LinearSVC(C=0.45, max_iter=1500, random_state=seed)
        svm.fit(X_tr, y_tr)
        platt = LogisticRegression(C=1.0, solver="lbfgs")
        platt.fit(svm.decision_function(X_tr).reshape(-1, 1), y_tr)
        return svm, platt

    @staticmethod
    def _pred_2col(model_pair, X_mat: sp.csr_matrix) -> np.ndarray:
        clf, platt = model_pair
        margins = clf.decision_function(X_mat).astype(np.float32)
        if platt is not None:
            probs = platt.predict_proba(margins.reshape(-1, 1))[:, 1].astype(np.float32)
        else:
            probs = expit(np.clip(margins, -25.0, 25.0)).astype(np.float32)
        return np.column_stack([margins, probs])

    def _compute_impostor_features(self, ms_u, mp_u, fw_z_u, idx1, idx2) -> np.ndarray:
        def _cohort_3col(sim_pair, sim_u_imp):
            masked = np.where(sim_u_imp > 0.9995, np.nan, sim_u_imp)
            mu_u = np.nanmean(masked, axis=1).astype(np.float32)
            sd_u = np.maximum(np.nanstd(masked, axis=1), 1e-5).astype(np.float32)
            sorted_imp = np.sort(np.where(np.isnan(masked), -1.0, masked), axis=1)
            top10_u = sorted_imp[:, -10:].mean(axis=1).astype(np.float32)
            s12 = sim_pair.ravel()
            as_norm = 0.5 * (((s12 - mu_u[idx1]) / sd_u[idx1]) + ((s12 - mu_u[idx2]) / sd_u[idx2]))
            top10_margin = s12 - 0.5 * (top10_u[idx1] + top10_u[idx2])
            imp1 = np.where(np.isnan(masked[idx1]), -1.0, masked[idx1])
            imp2 = np.where(np.isnan(masked[idx2]), -1.0, masked[idx2])
            mut_rank = 0.5 * (np.mean(s12[:, None] > imp1, axis=1) + np.mean(s12[:, None] > imp2, axis=1))
            return np.column_stack([as_norm, top10_margin, mut_rank]).astype(np.float32)

        c_surf = _cohort_3col(np.asarray(ms_u[idx1].multiply(ms_u[idx2]).sum(axis=1), dtype=np.float32), (ms_u @ self.imp_surf_.T).toarray().astype(np.float32))
        c_pn = _cohort_3col(np.asarray(mp_u[idx1].multiply(mp_u[idx2]).sum(axis=1), dtype=np.float32), (mp_u @ self.imp_pnoise_.T).toarray().astype(np.float32))
        fw_unit = (fw_z_u / (np.linalg.norm(fw_z_u, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        c_fw = _cohort_3col(np.sum(fw_unit[idx1] * fw_unit[idx2], axis=1, dtype=np.float32), fw_unit @ self.imp_fw_unit_.T)

        def _koppel_winrate(u_proj, imp_proj):
            u_n = (u_proj / (np.linalg.norm(u_proj, axis=2, keepdims=True) + 1e-8)).astype(np.float32)
            n_u, n_b, _ = u_n.shape
            max_imp = np.empty((n_u, n_b), dtype=np.float32)
            for st in range(0, n_u, 4096):
                ed = min(st + 4096, n_u)
                blk = np.einsum("nbd,mbd->nbm", u_n[st:ed], imp_proj, optimize=True)
                blk[blk > 0.9995] = -1.0
                max_imp[st:ed] = np.max(blk, axis=2)
            pair_sim = np.sum(u_n[idx1] * u_n[idx2], axis=2)
            w1 = (pair_sim > max_imp[idx1]).astype(np.float32)
            w2 = (pair_sim > max_imp[idx2]).astype(np.float32)
            return (0.5 * (w1.mean(axis=1) + w2.mean(axis=1))).reshape(-1, 1)

        kop_pn = _koppel_winrate((mp_u @ self.koppel_proj_pnoise_).reshape(mp_u.shape[0], self.n_koppel_bags, -1), self.koppel_imp_pnoise_)
        kop_fw = _koppel_winrate(fw_z_u[:, self.koppel_proj_fw_], self.koppel_imp_fw_)
        kop_mean = 0.5 * (kop_pn + kop_fw)
        imp_gap = c_pn[:, 0:1] - c_surf[:, 0:1]
        imp_mean = (c_surf[:, 0:1] + c_pn[:, 0:1] + c_fw[:, 0:1]) / 3.0
        return np.hstack([c_surf, c_pn, c_fw, kop_pn, kop_fw, kop_mean, imp_gap, imp_mean]).astype(np.float32)

    def fit_transform_train(self, df: pd.DataFrame) -> sp.csr_matrix:
        t0 = time.perf_counter()
        n_pairs = len(df)
        y = df["label"].to_numpy(dtype=np.int32)

        combined = pd.concat([df["text_1"], df["text_2"]], ignore_index=True)
        inv_idx, unique_vals = pd.factorize(combined, sort=False)
        unique_texts = [str(x) for x in unique_vals]
        idx1, idx2 = inv_idx[:n_pairs], inv_idx[n_pairs:]

        pnoise_u, dist_u, comp_u, prof_u = self._parallel_views(unique_texts)

        # Dual-author-disjoint 5-bucket OOF for stage-1 linear/SVM views
        all_auths = sorted(set(df["author_1"].astype(str)) | set(df["author_2"].astype(str)))
        rng_b = np.random.default_rng(self.seed)
        perm = list(all_auths)
        rng_b.shuffle(perm)
        a2b = {a: i % self.n_oof_folds for i, a in enumerate(perm)}
        b1 = np.fromiter((a2b[a] for a in df["author_1"].astype(str)), dtype=np.int32, count=n_pairs)
        b2 = np.fromiter((a2b[a] for a in df["author_2"].astype(str)), dtype=np.int32, count=n_pairs)

        oof_acc = np.zeros((n_pairs, 10), dtype=np.float32)
        oof_cnt = np.zeros((n_pairs, 1), dtype=np.float32)
        for k in range(self.n_oof_folds):
            tr_i = np.flatnonzero((b1 != k) & (b2 != k))
            va_i = np.flatnonzero((b1 == k) | (b2 == k))
            tr_docs = np.unique(np.concatenate([idx1[tr_i], idx2[tr_i]]))
            va_docs = np.unique(np.concatenate([idx1[va_i], idx2[va_i]]))
            needed = np.unique(np.concatenate([tr_docs, va_docs]))
            lmap = {int(g): li for li, g in enumerate(needed)}
            l_tr1 = np.fromiter((lmap[int(g)] for g in idx1[tr_i]), dtype=np.int32, count=len(tr_i))
            l_tr2 = np.fromiter((lmap[int(g)] for g in idx2[tr_i]), dtype=np.int32, count=len(tr_i))
            l_va1 = np.fromiter((lmap[int(g)] for g in idx1[va_i]), dtype=np.int32, count=len(va_i))
            l_va2 = np.fromiter((lmap[int(g)] for g in idx2[va_i]), dtype=np.int32, count=len(va_i))

            fp = np.zeros((len(va_i), 10), dtype=np.float32)
            vs = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), max_features=self.surf_hi_dim, sublinear_tf=True, min_df=5, lowercase=False, dtype=np.float32)
            vs.fit([unique_texts[g] for g in tr_docs])
            ms = vs.transform([unique_texts[g] for g in needed])
            X_trs, X_vas = _sym_sparse_pair(ms[l_tr1], ms[l_tr2]), _sym_sparse_pair(ms[l_va1], ms[l_va2])
            fp[:, 0:2] = self._pred_2col(self._fit_lr(X_trs, y[tr_i], self.seed + k), X_vas)
            fp[:, 2:4] = self._pred_2col(self._fit_svm_with_platt(X_trs, y[tr_i], self.seed + k), X_vas)

            vp = TfidfVectorizer(analyzer="word", ngram_range=(1, 4), max_features=self.pnoise_hi_dim, token_pattern=r"\S+", sublinear_tf=True, min_df=5, lowercase=False, dtype=np.float32)
            vp.fit([pnoise_u[g] for g in tr_docs])
            mp = vp.transform([pnoise_u[g] for g in needed])
            X_trp, X_vap = _sym_sparse_pair(mp[l_tr1], mp[l_tr2]), _sym_sparse_pair(mp[l_va1], mp[l_va2])
            fp[:, 4:6] = self._pred_2col(self._fit_lr(X_trp, y[tr_i], self.seed + k), X_vap)
            fp[:, 6:8] = self._pred_2col(self._fit_svm_with_platt(X_trp, y[tr_i], self.seed + k), X_vap)

            vd = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), max_features=self.dist_hi_dim, sublinear_tf=True, min_df=5, lowercase=False, dtype=np.float32)
            vd.fit([dist_u[g] for g in tr_docs])
            md = vd.transform([dist_u[g] for g in needed])
            fp[:, 8:10] = self._pred_2col(self._fit_lr(_sym_sparse_pair(md[l_tr1], md[l_tr2]), y[tr_i], self.seed + k), _sym_sparse_pair(md[l_va1], md[l_va2]))

            oof_acc[va_i] += fp
            oof_cnt[va_i, 0] += 1.0
            print(f"[stage1] bucket={k+1}/{self.n_oof_folds} surf_svm_auc={roc_auc_score(y[va_i], fp[:, 3]):.4f} pnoise_lr_auc={roc_auc_score(y[va_i], fp[:, 5]):.4f}")

        oof_s1_18 = _augment_s1_interactions(oof_acc / oof_cnt)

        ms_full = self.vec_surf_hi.fit_transform(unique_texts)
        X_as = _sym_sparse_pair(ms_full[idx1], ms_full[idx2])
        self.s1_models_["surf_lr"] = self._fit_lr(X_as, y, self.seed)
        self.s1_models_["surf_svm"] = self._fit_svm_with_platt(X_as, y, self.seed)
        del X_as

        mp_full = self.vec_pnoise_hi.fit_transform(pnoise_u)
        X_ap = _sym_sparse_pair(mp_full[idx1], mp_full[idx2])
        self.s1_models_["pnoise_lr"] = self._fit_lr(X_ap, y, self.seed)
        self.s1_models_["pnoise_svm"] = self._fit_svm_with_platt(X_ap, y, self.seed)
        del X_ap

        md_full = self.vec_dist_hi.fit_transform(dist_u)
        self.s1_models_["dist_lr"] = self._fit_lr(_sym_sparse_pair(md_full[idx1], md_full[idx2]), y, self.seed)

        fw_rates = self.fw_vec.transform(unique_texts).toarray().astype(np.float32) / np.maximum(prof_u[:, 33:34], 1.0)
        self.fw_mean_ = fw_rates.mean(axis=0).astype(np.float32)
        self.fw_std_ = np.maximum(fw_rates.std(axis=0), 1e-6).astype(np.float32)
        fw_z = (fw_rates - self.fw_mean_) / self.fw_std_

        rng = np.random.default_rng(self.seed)
        imp_ids = rng.choice(len(unique_texts), size=min(self.n_impostors, len(unique_texts)), replace=False)
        self.imp_surf_ = ms_full[imp_ids].tocsr()
        self.imp_pnoise_ = mp_full[imp_ids].tocsr()
        fw_imp = fw_z[imp_ids]
        self.imp_fw_unit_ = (fw_imp / (np.linalg.norm(fw_imp, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

        proj_d = 24
        self.koppel_proj_pnoise_ = rng.standard_normal((mp_full.shape[1], self.n_koppel_bags * proj_d), dtype=np.float32) / math.sqrt(proj_d)
        imp_pn_p = (self.imp_pnoise_ @ self.koppel_proj_pnoise_).reshape(len(imp_ids), self.n_koppel_bags, proj_d)
        self.koppel_imp_pnoise_ = (imp_pn_p / (np.linalg.norm(imp_pn_p, axis=2, keepdims=True) + 1e-8)).astype(np.float32)

        self.koppel_proj_fw_ = np.vstack([rng.choice(250, size=48, replace=False) for _ in range(self.n_koppel_bags)])
        imp_fw_b = fw_imp[:, self.koppel_proj_fw_]
        self.koppel_imp_fw_ = (imp_fw_b / (np.linalg.norm(imp_fw_b, axis=2, keepdims=True) + 1e-8)).astype(np.float32)

        self.vec_tree_char.fit(unique_texts)
        self.tree_char_order_ = np.argsort(self.vec_tree_char.idf_)
        self.vec_tree_pnoise.fit(pnoise_u)
        self.tree_pnoise_order_ = np.argsort(self.vec_tree_pnoise.idf_)

        self._build_catalog()
        X_tr = self._assemble(df, unique_texts, pnoise_u, comp_u, prof_u, fw_rates, fw_z, ms_full, mp_full, md_full, idx1, idx2, oof_s1_18, "train")
        print(f"[extractor] fit_transform_train shape={X_tr.shape} elapsed={time.perf_counter()-t0:.1f}s")
        return X_tr

    def transform(self, df: pd.DataFrame, split_name: str = "eval") -> sp.csr_matrix:
        n_pairs = len(df)
        combined = pd.concat([df["text_1"], df["text_2"]], ignore_index=True)
        inv_idx, unique_vals = pd.factorize(combined, sort=False)
        unique_texts = [str(x) for x in unique_vals]
        idx1, idx2 = inv_idx[:n_pairs], inv_idx[n_pairs:]

        pnoise_u, dist_u, comp_u, prof_u = self._parallel_views(unique_texts)
        ms = self.vec_surf_hi.transform(unique_texts)
        X_s = _sym_sparse_pair(ms[idx1], ms[idx2])
        s1_slr, s1_ssvm = self._pred_2col(self.s1_models_["surf_lr"], X_s), self._pred_2col(self.s1_models_["surf_svm"], X_s)
        del X_s

        mp = self.vec_pnoise_hi.transform(pnoise_u)
        X_p = _sym_sparse_pair(mp[idx1], mp[idx2])
        s1_plr, s1_psvm = self._pred_2col(self.s1_models_["pnoise_lr"], X_p), self._pred_2col(self.s1_models_["pnoise_svm"], X_p)
        del X_p

        md = self.vec_dist_hi.transform(dist_u)
        s1_dlr = self._pred_2col(self.s1_models_["dist_lr"], _sym_sparse_pair(md[idx1], md[idx2]))

        s1_18 = _augment_s1_interactions(np.hstack([s1_slr, s1_ssvm, s1_plr, s1_psvm, s1_dlr]).astype(np.float32))
        fw_rates = self.fw_vec.transform(unique_texts).toarray().astype(np.float32) / np.maximum(prof_u[:, 33:34], 1.0)
        fw_z = (fw_rates - self.fw_mean_) / self.fw_std_
        return self._assemble(df, unique_texts, pnoise_u, comp_u, prof_u, fw_rates, fw_z, ms, mp, md, idx1, idx2, s1_18, split_name)

    def _build_catalog(self) -> None:
        g1 = [
            "s1_surf60k_lr_margin", "s1_surf60k_lr_prob", "s1_surf60k_svm_margin", "s1_surf60k_svm_prob",
            "s1_pnoise40k_lr_margin", "s1_pnoise40k_lr_prob", "s1_pnoise40k_svm_margin", "s1_pnoise40k_svm_prob",
            "s1_distort30k_lr_margin", "s1_distort30k_lr_prob",
            "s1_gap_pnoise_minus_surf_lr", "s1_gap_pnoise_minus_surf_svm", "s1_gap_distort_minus_surf_lr",
            "s1_agree_pnoise_x_surf_prob", "s1_all5_margin_mean", "s1_all5_margin_std", "s1_all5_margin_min", "s1_all5_margin_max",
        ]
        g2 = [
            "imp_surf_asnorm", "imp_surf_top10_margin", "imp_surf_mut_rank",
            "imp_pnoise_asnorm", "imp_pnoise_top10_margin", "imp_pnoise_mut_rank",
            "imp_fw_asnorm", "imp_fw_top10_margin", "imp_fw_mut_rank",
            "koppel_bag64_pnoise_win", "koppel_bag64_fw_win", "koppel_bag64_mean_win",
            "imp_gap_pnoise_minus_surf_asnorm", "imp_mean_3views_asnorm",
        ]
        g3 = ["ncd_raw_zlib", "cbc_raw_zlib", "comp_ratio_diff_raw", "ncd_pnoise_zlib", "cbc_pnoise_zlib", "comp_ratio_diff_pnoise"]
        g4 = (
            [f"absdiff_{n}" for n in DOC_PROFILE_62_NAMES]
            + [f"reldiff_{n}" for n in DOC_PROFILE_62_NAMES]
            + [f"pairmean_{n}" for n in DOC_PROFILE_62_NAMES]
            + ["pair_min_words", "pair_max_words", "pair_word_len_ratio", "pair_log_total_words"]
        )
        g5 = [f"fw_zdiff_{w.replace(chr(39), '_')}" for w in FUNCTION_WORDS_250] + [
            "fw_burrows_delta", "fw_argamon_l2_delta", "fw_wurzburg_cos_delta",
            "fw_jsd_divergence", "fw_raw_cosine", "fw_raw_l1", "fw_raw_canberra", "fw_hist_inter",
        ]
        g6_d = [
            "kern_surf_cos", "kern_surf_l1", "kern_surf_l2", "kern_surf_hist_inter", "kern_surf_jaccard", "kern_surf_dice",
            "kern_surf_band_top", "kern_surf_band_mid", "kern_surf_band_rare",
            "kern_pnoise_cos", "kern_pnoise_l1", "kern_pnoise_hist_inter", "kern_pnoise_jaccard",
            "kern_distort_cos", "kern_hi_surf60k_cos",
        ]

        inv_tc = np.empty(len(self.vec_tree_char.vocabulary_), dtype=object)
        for tok, ci in self.vec_tree_char.vocabulary_.items():
            inv_tc[ci] = tok
        ord_tc = inv_tc[self.tree_char_order_]
        tc_diff = [f"cngram_diff_{i:04d}_{repr(t)[1:-1]}" for i, t in enumerate(ord_tc)]
        tc_prod = [f"cngram_prod_{i:04d}_{repr(t)[1:-1]}" for i, t in enumerate(ord_tc[: min(self.tree_char_prod_dim, len(ord_tc))])]

        inv_tp = np.empty(len(self.vec_tree_pnoise.vocabulary_), dtype=object)
        for tok, ci in self.vec_tree_pnoise.vocabulary_.items():
            inv_tp[ci] = tok
        ord_tp = inv_tp[self.tree_pnoise_order_]
        tp_diff = [f"pnoise_diff_{i:04d}_{t}" for i, t in enumerate(ord_tp)]
        tp_prod = [f"pnoise_prod_{i:04d}_{t}" for i, t in enumerate(ord_tp[: min(self.tree_pnoise_prod_dim, len(ord_tp))])]

        dense_names = g1 + g2 + g3 + g4 + g5 + g6_d
        sparse_names = tc_diff + tp_diff + tc_prod + tp_prod
        self.n_dense_ = len(dense_names)
        self.feature_names_ = dense_names + sparse_names
        pos = 0
        self.group_slices_ = {}
        for nm, blk in [
            ("G1_Stage1_Margin_Gap", g1), ("G2_Impostor_Cohort", g2), ("G3_NCD_Compression", g3),
            ("G4_Gram2Vec_Biometrics", g4), ("G5_FunctionWords_Delta", g5), ("G6_MultiView_Sparse_Habits", g6_d + sparse_names),
        ]:
            self.group_slices_[nm] = (pos, pos + len(blk))
            pos += len(blk)

    def _assemble(self, df, unique_texts, pnoise_u, comp_u, prof_u, fw_rates, fw_z, ms_hi, mp_hi, md_hi, idx1, idx2, s1_18, split_name) -> sp.csr_matrix:
        n_pairs = len(df)
        feat_imp = self._compute_impostor_features(ms_hi, mp_hi, fw_z, idx1, idx2)
        feat_ncd = self._parallel_ncd(
            [unique_texts[i] for i in idx1], [unique_texts[i] for i in idx2],
            [pnoise_u[i] for i in idx1], [pnoise_u[i] for i in idx2],
            comp_u[idx1], comp_u[idx2],
        )

        p1, p2 = prof_u[idx1], prof_u[idx2]
        p_abs = np.abs(p1 - p2)
        p_rel = p_abs / (np.abs(p1) + np.abs(p2) + 1e-6)
        w1, w2 = p1[:, 33:34], p2[:, 33:34]
        min_w, max_w = np.minimum(w1, w2), np.maximum(w1, w2)
        feat_g4 = np.hstack([p_abs, p_rel, 0.5 * (p1 + p2), min_w, max_w, min_w / np.maximum(max_w, 1.0), np.log1p(w1 + w2)]).astype(np.float32)

        f1, f2 = fw_rates[idx1], fw_rates[idx2]
        z1, z2 = fw_z[idx1], fw_z[idx2]
        z_abs = np.abs(z1 - z2)
        burrows = z_abs.mean(axis=1, keepdims=True)
        argamon = np.sqrt(np.mean((z1 - z2) ** 2, axis=1, keepdims=True))
        wurzburg = 1.0 - np.sum((z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)) * (z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)), axis=1, keepdims=True)
        f1p = (f1 + 1e-7) / np.sum(f1 + 1e-7, axis=1, keepdims=True)
        f2p = (f2 + 1e-7) / np.sum(f2 + 1e-7, axis=1, keepdims=True)
        mp_f = 0.5 * (f1p + f2p)
        jsd = 0.5 * (np.sum(f1p * np.log2(f1p / mp_f), axis=1, keepdims=True) + np.sum(f2p * np.log2(f2p / mp_f), axis=1, keepdims=True))
        fw_cos = np.sum((f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)) * (f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)), axis=1, keepdims=True)
        fw_l1 = np.sum(np.abs(f1 - f2), axis=1, keepdims=True)
        fw_canb = np.mean(np.abs(f1 - f2) / (f1 + f2 + 1e-7), axis=1, keepdims=True)
        fw_hist = np.sum(np.minimum(f1, f2), axis=1, keepdims=True)
        feat_g5 = np.hstack([z_abs, burrows, argamon, wurzburg, jsd, fw_cos, fw_l1, fw_canb, fw_hist]).astype(np.float32)

        tc_u = self.vec_tree_char.transform(unique_texts)[:, self.tree_char_order_].tocsr()
        c1, c2 = tc_u[idx1], tc_u[idx2]
        c_diff = (c1 - c2).tocsr()
        c_diff.data = np.abs(c_diff.data).astype(np.float32)
        n_cp = min(self.tree_char_prod_dim, c1.shape[1])
        c_prod = c1[:, :n_cp].multiply(c2[:, :n_cp]).tocsr().astype(np.float32)

        c_cos = np.asarray(c1.multiply(c2).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c_l1 = np.asarray(c_diff.sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c1_sq = np.asarray(c1.multiply(c1).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c2_sq = np.asarray(c2.multiply(c2).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c_l2 = np.sqrt(np.maximum(c1_sq + c2_sq - 2.0 * c_cos, 0.0))
        c_hist = 0.5 * np.maximum(np.asarray(c1.sum(axis=1), dtype=np.float32).reshape(-1, 1) + np.asarray(c2.sum(axis=1), dtype=np.float32).reshape(-1, 1) - c_l1, 0.0)
        b1, b2 = (c1 > 0).astype(np.float32), (c2 > 0).astype(np.float32)
        inter = np.asarray(b1.multiply(b2).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        bc1, bc2 = np.asarray(b1.sum(axis=1), dtype=np.float32).reshape(-1, 1), np.asarray(b2.sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c_jacc = inter / np.maximum(bc1 + bc2 - inter, 1.0)
        c_dice = (2.0 * inter) / np.maximum(bc1 + bc2, 1.0)

        def _bcos(sc: int, ec: int) -> np.ndarray:
            if ec <= sc:
                return np.zeros((n_pairs, 1), dtype=np.float32)
            s1m, s2m = c1[:, sc:ec], c2[:, sc:ec]
            d = np.asarray(s1m.multiply(s2m).sum(axis=1), dtype=np.float32).reshape(-1, 1)
            n1 = np.sqrt(np.asarray(s1m.multiply(s1m).sum(axis=1), dtype=np.float32).reshape(-1, 1))
            n2 = np.sqrt(np.asarray(s2m.multiply(s2m).sum(axis=1), dtype=np.float32).reshape(-1, 1))
            return d / (n1 * n2 + 1e-8)

        nc = c1.shape[1]
        bt, bm, br = _bcos(0, min(400, nc)), _bcos(min(400, nc), min(1500, nc)), _bcos(min(1500, nc), nc)

        tp_u = self.vec_tree_pnoise.transform(pnoise_u)[:, self.tree_pnoise_order_].tocsr()
        pn1, pn2 = tp_u[idx1], tp_u[idx2]
        pn_diff = (pn1 - pn2).tocsr()
        pn_diff.data = np.abs(pn_diff.data).astype(np.float32)
        n_pp = min(self.tree_pnoise_prod_dim, pn1.shape[1])
        pn_prod = pn1[:, :n_pp].multiply(pn2[:, :n_pp]).tocsr().astype(np.float32)

        pn_cos = np.asarray(pn1.multiply(pn2).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        pn_l1 = np.asarray(pn_diff.sum(axis=1), dtype=np.float32).reshape(-1, 1)
        pn_hist = 0.5 * np.maximum(np.asarray(pn1.sum(axis=1), dtype=np.float32).reshape(-1, 1) + np.asarray(pn2.sum(axis=1), dtype=np.float32).reshape(-1, 1) - pn_l1, 0.0)
        pb1, pb2 = (pn1 > 0).astype(np.float32), (pn2 > 0).astype(np.float32)
        p_inter = np.asarray(pb1.multiply(pb2).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        pn_jacc = p_inter / np.maximum(np.asarray(pb1.sum(axis=1), dtype=np.float32).reshape(-1, 1) + np.asarray(pb2.sum(axis=1), dtype=np.float32).reshape(-1, 1) - p_inter, 1.0)
        dist_cos = np.asarray(md_hi[idx1].multiply(md_hi[idx2]).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        hi_surf_cos = np.asarray(ms_hi[idx1].multiply(ms_hi[idx2]).sum(axis=1), dtype=np.float32).reshape(-1, 1)

        feat_g6 = np.hstack([c_cos, c_l1, c_l2, c_hist, c_jacc, c_dice, bt, bm, br, pn_cos, pn_l1, pn_hist, pn_jacc, dist_cos, hi_surf_cos]).astype(np.float32)
        dense_mat = np.hstack([s1_18, feat_imp, feat_ncd, feat_g4, feat_g5, feat_g6], dtype=np.float32)
        return sp.hstack([sp.csr_matrix(dense_mat), c_diff, pn_diff, c_prod, pn_prod], format="csr", dtype=np.float32)
'''),
    md_cell("## 4. Feature Extraction & Caching"),
    code_cell(r'''cache_meta = CACHE_DIR / "meta.joblib"
cache_ext = CACHE_DIR / "extractor.joblib"
split_paths = {
    k: CACHE_DIR / f"X_{k}.npz"
    for k in ("train", "val_open", "test_open", "val_closed", "test_closed", "anchor_hub")
}

if (not CONFIG["force_recompute_features"]) and cache_meta.exists() and cache_ext.exists() and all(p.exists() for p in split_paths.values()):
    meta = joblib.load(cache_meta)
    extractor = joblib.load(cache_ext)
    X_train = sp.load_npz(split_paths["train"])
    X_val_open = sp.load_npz(split_paths["val_open"])
    X_test_open = sp.load_npz(split_paths["test_open"])
    X_val_closed = sp.load_npz(split_paths["val_closed"])
    X_test_closed = sp.load_npz(split_paths["test_closed"])
    X_anchor_hub = sp.load_npz(split_paths["anchor_hub"])
    print(f"[cache] loaded features train={X_train.shape}")
else:
    extractor = StylometricPairExtractor(n_jobs=N_JOBS, seed=SEED)
    X_train = extractor.fit_transform_train(train_df)
    X_val_open = extractor.transform(val_open_df, "val_open")
    X_test_open = extractor.transform(test_open_df, "test_open")
    X_val_closed = extractor.transform(val_closed_pairs_df, "val_closed")
    X_test_closed = extractor.transform(test_closed_pairs_df, "test_closed")
    X_anchor_hub = extractor.transform(anchor_hub_pairs_df, "anchor_hub")

    for nm, mat in [
        ("train", X_train), ("val_open", X_val_open), ("test_open", X_test_open),
        ("val_closed", X_val_closed), ("test_closed", X_test_closed), ("anchor_hub", X_anchor_hub),
    ]:
        sp.save_npz(split_paths[nm], mat)
    meta = {
        "feature_names": extractor.feature_names_,
        "group_slices": extractor.group_slices_,
        "n_dense": extractor.n_dense_,
    }
    joblib.dump(meta, cache_meta)
    joblib.dump(extractor, cache_ext)

feature_names = meta["feature_names"]
group_slices = meta["group_slices"]
n_dense_cols = int(meta["n_dense"])
name_to_col = {nm: idx for idx, nm in enumerate(feature_names)}
print(f"features={len(feature_names):,} (dense={n_dense_cols}, sparse={len(feature_names)-n_dense_cols:,})")
'''),
    md_cell(r'''## 5. Multi-View 10-Expert Training

전체 6,001 피처 뷰(3종), 중요도 기반 가지치기 + 대칭 교차 피처 12종 뷰(5종), 도메인 편향 억제 뷰(장르 블라인드 순수 문체 편차 CatBoost + 교차 주제 2.5배 가중 LightGBM 2종)까지 총 10개 전문가를 순차 학습합니다.
'''),
    code_cell(r'''def safe_logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def build_interaction_features(X_csr: sp.csr_matrix) -> np.ndarray:
    def _col(nm: str) -> np.ndarray:
        return X_csr[:, name_to_col[nm]].toarray().ravel().astype(np.float32)

    p_surf, p_pn, m_mean = _col("s1_surf60k_svm_prob"), _col("s1_pnoise40k_svm_prob"), _col("s1_all5_margin_mean")
    imp_s_t10, imp_s_rk, imp_s_as = _col("imp_surf_top10_margin"), _col("imp_surf_mut_rank"), _col("imp_surf_asnorm")
    imp_p_t10, imp_p_rk, imp_p_as = _col("imp_pnoise_top10_margin"), _col("imp_pnoise_mut_rank"), _col("imp_pnoise_asnorm")
    cbc_raw, ncd_raw, ncd_pn, min_w = _col("cbc_raw_zlib"), _col("ncd_raw_zlib"), _col("ncd_pnoise_zlib"), _col("pair_min_words")

    return np.column_stack([
        p_pn * imp_s_rk,
        p_surf * imp_p_rk,
        p_pn * imp_p_rk,
        imp_p_t10 - imp_s_t10,
        0.5 * (imp_p_t10 + imp_s_t10),
        np.minimum(imp_s_rk, imp_p_rk),
        np.minimum(p_surf, p_pn),
        (2.0 * p_surf * p_pn) / (p_surf + p_pn + 1e-6),
        (1.0 - cbc_raw) * m_mean,
        ncd_pn - ncd_raw,
        m_mean * np.log1p(np.maximum(min_w, 0.0)),
        np.minimum(imp_s_as, imp_p_as),
    ]).astype(np.float32)


selected_cols = np.arange(X_train.shape[1], dtype=np.int32)
selected_dense_cols = np.arange(n_dense_cols, dtype=np.int32)

X_tr_dense_full = X_train[:, selected_dense_cols].toarray().astype(np.float32)
X_vo_dense_full = X_val_open[:, selected_dense_cols].toarray().astype(np.float32)
X_vc_dense_full = X_val_closed[:, selected_dense_cols].toarray().astype(np.float32)
X_ah_dense_full = X_anchor_hub[:, selected_dense_cols].toarray().astype(np.float32)

# Full-view experts (1-3)
lgb_gbdt = lgb.LGBMClassifier(
    objective="binary", metric=["auc", "binary_logloss"], boosting_type="gbdt",
    subsample=0.80, subsample_freq=1, n_estimators=3500, n_jobs=N_JOBS, random_state=SEED, verbose=-1,
    **CONFIG["lgb_full"],
)
lgb_gbdt.fit(
    X_train, y_train, eval_set=[(X_val_open, y_val_open)],
    callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)],
)

lgb_goss = lgb.LGBMClassifier(
    objective="binary", metric=["auc", "binary_logloss"], boosting_type="goss",
    top_rate=0.20, other_rate=0.10, n_estimators=3500, n_jobs=N_JOBS, random_state=SEED + 1, verbose=-1,
    **CONFIG["lgb_full"],
)
lgb_goss.fit(
    X_train, y_train, eval_set=[(X_val_open, y_val_open)],
    callbacks=[lgb.early_stopping(150, first_metric_only=True, verbose=False)],
)

cat_full = CatBoostClassifier(
    loss_function="Logloss", eval_metric="AUC", iterations=3000,
    random_seed=SEED, thread_count=N_JOBS, verbose=0, **CONFIG["cat_full"],
)
cat_full.fit(X_tr_dense_full, y_train, eval_set=(X_vo_dense_full, y_val_open), early_stopping_rounds=180, use_best_model=True)

# Gain-based column pruning subsets
gain_imp = lgb_gbdt.booster_.feature_importance(importance_type="gain") + lgb_goss.booster_.feature_importance(importance_type="gain")
feat_imp_df = pd.DataFrame({
    "col_idx": np.arange(len(feature_names), dtype=np.int32),
    "feature": feature_names,
    "gain_importance": gain_imp,
}).sort_values("gain_importance", ascending=False).reset_index(drop=True)
feat_imp_df.to_csv(OUTPUT_DIR / "feature_pruning_ranking.csv", index=False)

top_lgb_cols = np.sort(feat_imp_df["col_idx"].iloc[: CONFIG["pruned_lgb_k"]].to_numpy(dtype=np.int32))
top_dense_cols_r2 = np.sort(feat_imp_df[feat_imp_df["col_idx"] < n_dense_cols]["col_idx"].iloc[: CONFIG["cb_dense_k"]].to_numpy(dtype=np.int32))
top_sparse_cols_r2 = np.sort(feat_imp_df[feat_imp_df["col_idx"] >= n_dense_cols]["col_idx"].iloc[: CONFIG["cb_sparse_k"]].to_numpy(dtype=np.int32))
cb_hybrid_cols = np.concatenate([top_dense_cols_r2, top_sparse_cols_r2])

ix_tr, ix_vo = build_interaction_features(X_train), build_interaction_features(X_val_open)
ix_vc, ix_ah = build_interaction_features(X_val_closed), build_interaction_features(X_anchor_hub)

X_tr_r2_lgb = sp.hstack([X_train[:, top_lgb_cols], sp.csr_matrix(ix_tr)], format="csr")
X_vo_r2_lgb = sp.hstack([X_val_open[:, top_lgb_cols], sp.csr_matrix(ix_vo)], format="csr")
X_vc_r2_lgb = sp.hstack([X_val_closed[:, top_lgb_cols], sp.csr_matrix(ix_vc)], format="csr")
X_ah_r2_lgb = sp.hstack([X_anchor_hub[:, top_lgb_cols], sp.csr_matrix(ix_ah)], format="csr")

X_tr_r2_cbh = np.hstack([X_train[:, cb_hybrid_cols].toarray().astype(np.float32), ix_tr])
X_vo_r2_cbh = np.hstack([X_val_open[:, cb_hybrid_cols].toarray().astype(np.float32), ix_vo])
X_vc_r2_cbh = np.hstack([X_val_closed[:, cb_hybrid_cols].toarray().astype(np.float32), ix_vc])
X_ah_r2_cbh = np.hstack([X_anchor_hub[:, cb_hybrid_cols].toarray().astype(np.float32), ix_ah])

X_tr_r2_den = np.hstack([X_train[:, top_dense_cols_r2].toarray().astype(np.float32), ix_tr])
X_vo_r2_den = np.hstack([X_val_open[:, top_dense_cols_r2].toarray().astype(np.float32), ix_vo])
X_vc_r2_den = np.hstack([X_val_closed[:, top_dense_cols_r2].toarray().astype(np.float32), ix_vc])
X_ah_r2_den = np.hstack([X_anchor_hub[:, top_dense_cols_r2].toarray().astype(np.float32), ix_ah])

# Pruned & regularized experts (4-8)
r2_lgb_gbdt = lgb.LGBMClassifier(
    objective="binary", metric=["auc", "binary_logloss"], boosting_type="gbdt",
    subsample=0.80, subsample_freq=1, n_estimators=2500, n_jobs=N_JOBS, random_state=SEED + 10, verbose=-1,
    **CONFIG["lgb_pruned"],
)
r2_lgb_gbdt.fit(X_tr_r2_lgb, y_train, eval_set=[(X_vo_r2_lgb, y_val_open)], callbacks=[lgb.early_stopping(140, first_metric_only=True, verbose=False)])

r2_lgb_goss = lgb.LGBMClassifier(
    objective="binary", metric=["auc", "binary_logloss"], boosting_type="goss",
    top_rate=0.20, other_rate=0.10, n_estimators=2500, n_jobs=N_JOBS, random_state=SEED + 11, verbose=-1,
    **CONFIG["lgb_pruned"],
)
r2_lgb_goss.fit(X_tr_r2_lgb, y_train, eval_set=[(X_vo_r2_lgb, y_val_open)], callbacks=[lgb.early_stopping(140, first_metric_only=True, verbose=False)])

r2_cb_hyb = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", iterations=2500, learning_rate=0.035, depth=6, l2_leaf_reg=8.0, random_seed=SEED + 12, thread_count=N_JOBS, verbose=0)
r2_cb_hyb.fit(X_tr_r2_cbh, y_train, eval_set=(X_vo_r2_cbh, y_val_open), early_stopping_rounds=180, use_best_model=True)

r2_cb_d4 = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", iterations=2500, learning_rate=0.04, depth=4, l2_leaf_reg=12.0, random_seed=SEED + 13, thread_count=N_JOBS, verbose=0)
r2_cb_d4.fit(X_tr_r2_den, y_train, eval_set=(X_vo_r2_den, y_val_open), early_stopping_rounds=180, use_best_model=True)

with threadpool_limits(limits=1):
    r2_scaler = StandardScaler()
    X_tr_r2_sc = np.clip(r2_scaler.fit_transform(np.nan_to_num(X_tr_r2_den, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
    X_vo_r2_sc = np.clip(r2_scaler.transform(np.nan_to_num(X_vo_r2_den, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
    X_vc_r2_sc = np.clip(r2_scaler.transform(np.nan_to_num(X_vc_r2_den, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
    X_ah_r2_sc = np.clip(r2_scaler.transform(np.nan_to_num(X_ah_r2_den, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
    r2_meta_lr = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-4, max_iter=300, random_state=SEED)
    r2_meta_lr.fit(X_tr_r2_sc, y_train)

# Debiased experts (9-10): genre-blind pure delta + cross-topic/domain-balanced weighting
PURE_DELTA_EXACT = {
    "s1_pnoise40k_lr_margin", "s1_pnoise40k_svm_margin", "s1_distort30k_lr_margin",
    "imp_fw_asnorm", "imp_fw_top10_margin", "imp_fw_mut_rank",
    "koppel_bag64_pnoise_win", "koppel_bag64_fw_win", "koppel_bag64_mean_win",
    "ncd_raw_zlib", "cbc_raw_zlib", "comp_ratio_diff_raw",
    "ncd_pnoise_zlib", "cbc_pnoise_zlib", "comp_ratio_diff_pnoise",
    "kern_pnoise_cos", "kern_pnoise_l1", "kern_pnoise_hist_inter", "kern_pnoise_jaccard", "kern_distort_cos",
}
pure_delta_dense = [i for i, nm in enumerate(feature_names[:n_dense_cols]) if (nm in PURE_DELTA_EXACT) or nm.startswith(("absdiff_", "reldiff_", "fw_"))]
top_pnoise_sparse = [int(ci) for ci in feat_imp_df["col_idx"].to_numpy() if ci >= n_dense_cols and feature_names[int(ci)].startswith(("pnoise_diff_", "pnoise_prod_"))][: CONFIG["genre_blind_sparse_k"]]
genre_blind_cols = np.sort(np.asarray(pure_delta_dense + top_pnoise_sparse, dtype=np.int32))

dom_vc = train_df["domain"].value_counts()
w_dom = 1.0 / np.sqrt(train_df["domain"].map(dom_vc).to_numpy(dtype=np.float32))
w_ct = np.where(train_df["is_cross_topic"].to_numpy() == 1, CONFIG["cross_topic_weight"], 1.0).astype(np.float32)
sample_w_r3 = (w_dom * w_ct) / float(np.mean(w_dom * w_ct))

X_tr_gb = X_train[:, genre_blind_cols].toarray().astype(np.float32)
X_vo_gb = X_val_open[:, genre_blind_cols].toarray().astype(np.float32)
X_vc_gb = X_val_closed[:, genre_blind_cols].toarray().astype(np.float32)
X_ah_gb = X_anchor_hub[:, genre_blind_cols].toarray().astype(np.float32)

r3_cb_genre_blind = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", iterations=2600, learning_rate=0.035, depth=6, l2_leaf_reg=10.0, random_seed=SEED + 31, thread_count=N_JOBS, verbose=0)
r3_cb_genre_blind.fit(X_tr_gb, y_train, sample_weight=sample_w_r3, eval_set=(X_vo_gb, y_val_open), early_stopping_rounds=180, use_best_model=True)

r3_lgb_ct = lgb.LGBMClassifier(
    objective="binary", metric=["auc", "binary_logloss"], boosting_type="gbdt",
    learning_rate=0.025, num_leaves=55, max_depth=8, min_child_samples=95,
    colsample_bytree=0.50, subsample=0.80, subsample_freq=1,
    reg_alpha=3.5, reg_lambda=12.0, n_estimators=2500, n_jobs=N_JOBS, random_state=SEED + 32, verbose=-1,
)
r3_lgb_ct.fit(X_tr_r2_lgb, y_train, sample_weight=sample_w_r3, eval_set=[(X_vo_r2_lgb, y_val_open)], callbacks=[lgb.early_stopping(140, first_metric_only=True, verbose=False)])


def score_10_experts(X_full: sp.csr_matrix) -> np.ndarray:
    ix = build_interaction_features(X_full)
    X_r2_l = sp.hstack([X_full[:, top_lgb_cols], sp.csr_matrix(ix)], format="csr")
    X_r2_h = np.hstack([X_full[:, cb_hybrid_cols].toarray().astype(np.float32), ix])
    X_r2_d = np.hstack([X_full[:, top_dense_cols_r2].toarray().astype(np.float32), ix])
    X_gb = X_full[:, genre_blind_cols].toarray().astype(np.float32)
    X_d_full = X_full[:, selected_dense_cols].toarray().astype(np.float32)
    with threadpool_limits(limits=1):
        X_r2_s = np.clip(r2_scaler.transform(np.nan_to_num(X_r2_d, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
        ps = [
            lgb_gbdt.predict_proba(X_full)[:, 1],
            lgb_goss.predict_proba(X_full)[:, 1],
            cat_full.predict_proba(X_d_full)[:, 1],
            r2_lgb_gbdt.predict_proba(X_r2_l)[:, 1],
            r2_lgb_goss.predict_proba(X_r2_l)[:, 1],
            r2_cb_hyb.predict_proba(X_r2_h)[:, 1],
            r2_cb_d4.predict_proba(X_r2_d)[:, 1],
            r2_meta_lr.predict_proba(X_r2_s)[:, 1],
            r3_cb_genre_blind.predict_proba(X_gb)[:, 1],
            r3_lgb_ct.predict_proba(X_r2_l)[:, 1],
        ]
    return np.column_stack([safe_logit(p) for p in ps]).astype(np.float32)


L_vo_10 = score_10_experts(X_val_open)
L_vc_10 = score_10_experts(X_val_closed)
L_ah_10 = score_10_experts(X_anchor_hub)
print(f"[experts] trained 10 experts | genre_blind_cols={len(genre_blind_cols)} | val_auc_range=[{min(roc_auc_score(y_val_open, expit(L_vo_10[:, i])) for i in range(10)):.4f}, {max(roc_auc_score(y_val_open, expit(L_vo_10[:, i])) for i in range(10)):.4f}]")
'''),
    md_cell(r'''## 6. Intrinsic OOD-Register Gated Meta-Calibrator & Evaluation Metrics

도메인 라벨 없이 단일 쌍의 내재적 사칭자 친화도, 전문가 간 불일치(Surface vs Genre-Blind), 문체 격식도 지표만을 입력으로 받아 5-Bucket 양방향 저자 배제 OOF 메타 보정기를 학습합니다.
'''),
    code_cell(r'''def build_ood_gate_matrix(X_csr: sp.csr_matrix, L_10: np.ndarray) -> np.ndarray:
    def _c(nm: str) -> np.ndarray:
        return X_csr[:, name_to_col[nm]].toarray().ravel().astype(np.float32)

    s_cos, p_cos = _c("kern_hi_surf60k_cos"), _c("kern_pnoise_cos")
    ood_surf_aff = s_cos - _c("imp_surf_top10_margin")
    ood_pn_aff = p_cos - _c("imp_pnoise_top10_margin")
    surf_infl = s_cos / (np.maximum(ood_surf_aff, 0.0) + 0.05)

    lg_gb, lg_ct = L_10[:, 8], L_10[:, 9]
    lg_surf = 0.5 * (L_10[:, 0] + L_10[:, 3])
    disagree_gb_surf = lg_gb - lg_surf
    disagree_ct_surf = lg_ct - lg_surf
    tree_std = np.std(L_10[:, [0, 1, 2, 3, 4, 5, 6, 8, 9]], axis=1)
    tree_min = np.min(L_10[:, [3, 5, 8, 9]], axis=1)

    return np.column_stack([
        L_10, disagree_gb_surf, disagree_ct_surf, tree_std, tree_min,
        ood_surf_aff, ood_pn_aff, surf_infl,
        _c("s1_gap_pnoise_minus_surf_svm"), _c("imp_gap_pnoise_minus_surf_asnorm"),
        _c("pairmean_bio_coleman_liau_idx"), _c("pairmean_bio_polysyllabic_ratio"),
        _c("pairmean_g2v_morph_nominal"), _c("pairmean_g2v_passive_rate"),
        _c("pairmean_g2v_contraction_rate"), _c("pairmean_bio_first_person_sing_ratio"),
        _c("pairmean_bio_uk_vs_us_orthography"), _c("pairmean_surf_digit_ratio"),
        _c("pairmean_surf_allcaps_ratio"), _c("pair_log_total_words"),
        surf_infl * lg_surf, _c("pairmean_bio_coleman_liau_idx") * disagree_gb_surf,
    ]).astype(np.float32)


def compute_anchor_hub_priors(anchor_pairs_df: pd.DataFrame, anchor_probs: np.ndarray) -> dict[tuple[str, str], float]:
    ap = anchor_pairs_df[["cohort_domain", "cand_a", "cand_b"]].copy()
    ap["logit"] = safe_logit(anchor_probs)
    both = pd.concat([ap, ap.rename(columns={"cand_a": "cand_b", "cand_b": "cand_a"})], ignore_index=True)
    p_means = both.groupby(["cohort_domain", "cand_a", "cand_b"], as_index=False)["logit"].mean()
    priors = {}
    for (dom, cid), grp in p_means.groupby(["cohort_domain", "cand_a"], sort=False):
        top5 = np.sort(grp["logit"].to_numpy())[-5:]
        priors[(dom, cid)] = float(np.mean(top5)) if len(top5) > 0 else 0.0
    return priors


def evaluate_closed_set_1to30(
    pairs_df: pd.DataFrame,
    pair_probs: np.ndarray,
    hub_priors: dict[tuple[str, str], float] | None = None,
    agg_mode: str = "inductive_csls",
    hub_weight: float = 0.35,
):
    work = pairs_df[["query_id", "cohort_domain", "true_author_id", "candidate_author_id"]].copy()
    work["prob"] = pair_probs
    work["logit"] = safe_logit(pair_probs)
    col = "prob" if agg_mode == "mean_prob" else "logit"
    cand_df = work.groupby(["query_id", "cohort_domain", "true_author_id", "candidate_author_id"], as_index=False)[col].mean()
    cand_df.rename(columns={col: "cand_score"}, inplace=True)

    if agg_mode == "inductive_csls" and hub_priors is not None and hub_weight > 0.0:
        p_vals = np.fromiter(
            (hub_priors.get((d, c), 0.0) for d, c in zip(cand_df["cohort_domain"], cand_df["candidate_author_id"])),
            dtype=np.float64, count=len(cand_df),
        )
        cand_df["cand_score"] = cand_df["cand_score"] - hub_weight * p_vals

    cand_df = cand_df.sort_values(["query_id", "cand_score"], ascending=[True, False])
    top1 = cand_df.groupby("query_id", as_index=False).first().rename(columns={"candidate_author_id": "pred_author_id", "cand_score": "top1_score"})
    top3_sets = cand_df.groupby("query_id")["candidate_author_id"].apply(lambda s: set(s.iloc[:3])).to_dict()
    sec_map = cand_df.groupby("query_id")["cand_score"].apply(lambda s: float(s.iloc[1]) if len(s) > 1 else -999.0).to_dict()

    top1["margin_top1_top2"] = top1["top1_score"] - top1["query_id"].map(sec_map)
    top1["is_top1_correct"] = (top1["pred_author_id"] == top1["true_author_id"]).astype(np.int32)
    top1["is_top3_correct"] = [int(ta in top3_sets[qid]) for qid, ta in zip(top1["query_id"], top1["true_author_id"])]

    dom_rows = []
    for dom, grp in top1.groupby("cohort_domain", sort=False):
        dom_rows.append({
            "Cohort_Domain": dom, "Queries": len(grp),
            "Top1_Accuracy": round(float(accuracy_score(grp["true_author_id"], grp["pred_author_id"])), 5),
            "Top3_Accuracy": round(float(grp["is_top3_correct"].mean()), 5),
            "Macro_F1": round(float(f1_score(grp["true_author_id"], grp["pred_author_id"], labels=sorted(grp["true_author_id"].unique()), average="macro", zero_division=0)), 5),
        })
    dom_df = pd.DataFrame(dom_rows).sort_values("Queries", ascending=False).reset_index(drop=True)
    summary = {
        "agg_mode": agg_mode,
        "overall_top1_acc": round(float(accuracy_score(top1["true_author_id"], top1["pred_author_id"])), 5),
        "overall_top3_acc": round(float(top1["is_top3_correct"].mean()), 5),
        "overall_macro_f1": round(float(f1_score(top1["true_author_id"], top1["pred_author_id"], labels=sorted(top1["true_author_id"].unique()), average="macro", zero_division=0)), 5),
        "domain_mean_top1_acc": round(float(dom_df["Top1_Accuracy"].mean()), 5),
    }
    return summary, dom_df, top1


def compute_open_pan_metrics(y_true: np.ndarray, y_prob: np.ndarray, tau: float = 0.50, delta: float = 0.0) -> dict[str, float]:
    n = len(y_true)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    is_u = (y_prob > tau - delta) & (y_prob < tau + delta) if delta > 0 else np.zeros(n, dtype=bool)
    n_u = int(np.sum(is_u))
    ans = ~is_u
    pred = (y_prob >= tau).astype(np.int32)
    n_c = int(np.sum((pred == y_true) & ans))
    c_at_1 = (n_c + n_u * (n_c / n)) / n

    tp = int(np.sum((pred == 1) & (y_true == 1) & ans))
    fp = int(np.sum((pred == 1) & (y_true == 0) & ans))
    fn_u = int(np.sum((y_true == 1) & ((pred == 0) | is_u)))
    d05 = 1.25 * tp + 0.25 * fn_u + fp
    f_05u = (1.25 * tp / d05) if d05 > 0 else 0.0

    eff_p = np.where(is_u, 0.5, y_prob)
    pan_brier = 1.0 - float(brier_score_loss(y_true, eff_p))
    f1_bin = float(f1_score(y_true, pred, zero_division=0))
    f1_mac = float(f1_score(y_true, pred, average="macro", zero_division=0))
    pan_overall = float(np.mean([roc_auc, c_at_1, f1_bin, f_05u, pan_brier]))
    return {
        "tau": round(float(tau), 4), "delta": round(float(delta), 4),
        "abstain_pct": round(float(n_u / n * 100.0), 2),
        "roc_auc": round(roc_auc, 5), "pan_c_at_1": round(float(c_at_1), 5),
        "pan_f_05u": round(float(f_05u), 5), "pan_brier": round(float(pan_brier), 5),
        "pan_overall": round(pan_overall, 5), "forced_macro_f1": round(f1_mac, 5),
        "answered_acc": round(float(n_c / max(n - n_u, 1)), 5),
    }


M_vo_gate = build_ood_gate_matrix(X_val_open, L_vo_10)
M_vc_gate = build_ood_gate_matrix(X_val_closed, L_vc_10)
M_ah_gate = build_ood_gate_matrix(X_anchor_hub, L_ah_10)

# 5-bucket dual-author-disjoint split stratified by domain
val_auth_dom = {}
for a1, a2, dm in zip(val_open_df["author_1"].astype(str), val_open_df["author_2"].astype(str), val_open_df["domain"].astype(str)):
    val_auth_dom.setdefault(a1, dm)
    val_auth_dom.setdefault(a2, dm)

rng_v = np.random.default_rng(SEED)
val_auth_bkt = {}
for dm in sorted(set(val_auth_dom.values())):
    al = sorted([a for a, d in val_auth_dom.items() if d == dm])
    rng_v.shuffle(al)
    for i, a in enumerate(al):
        val_auth_bkt[a] = i % 5

vb1 = np.fromiter((val_auth_bkt[a] for a in val_open_df["author_1"].astype(str)), dtype=np.int32, count=len(val_open_df))
vb2 = np.fromiter((val_auth_bkt[a] for a in val_open_df["author_2"].astype(str)), dtype=np.int32, count=len(val_open_df))

oof_acc = np.zeros(len(val_open_df), dtype=np.float64)
oof_cnt = np.zeros(len(val_open_df), dtype=np.float64)
gate_models = []

with threadpool_limits(limits=1):
    for k in range(5):
        tr_m = (vb1 != k) & (vb2 != k)
        va_m = (vb1 == k) | (vb2 == k)
        sc_k = StandardScaler()
        X_gk_tr = np.clip(sc_k.fit_transform(np.nan_to_num(M_vo_gate[tr_m], nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
        X_gk_va = np.clip(sc_k.transform(np.nan_to_num(M_vo_gate[va_m], nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)

        lr_k = LogisticRegression(C=0.15, max_iter=300, solver="lbfgs").fit(X_gk_tr, y_val_open[tr_m])
        cb_k = CatBoostClassifier(
            loss_function="Logloss", eval_metric="AUC", iterations=450, learning_rate=0.03,
            depth=3, l2_leaf_reg=25.0, random_seed=SEED + 50 + k, thread_count=N_JOBS, verbose=0,
        ).fit(M_vo_gate[tr_m], y_val_open[tr_m])

        p_va = 0.45 * lr_k.predict_proba(X_gk_va)[:, 1] + 0.55 * cb_k.predict_proba(M_vo_gate[va_m])[:, 1]
        oof_acc[va_m] += p_va
        oof_cnt[va_m] += 1.0
        gate_models.append((sc_k, lr_k, cb_k))

val_open_oof_p = oof_acc / oof_cnt


def predict_gated_meta(M_in: np.ndarray) -> np.ndarray:
    preds = []
    with threadpool_limits(limits=1):
        for sc_k, lr_k, cb_k in gate_models:
            X_sc = np.clip(sc_k.transform(np.nan_to_num(M_in, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
            preds.append(0.45 * lr_k.predict_proba(X_sc)[:, 1] + 0.55 * cb_k.predict_proba(M_in)[:, 1])
    return np.mean(preds, axis=0)


val_closed_p = predict_gated_meta(M_vc_gate)
anchor_hub_p = predict_gated_meta(M_ah_gate)
hub_priors = compute_anchor_hub_priors(anchor_hub_pairs_df, anchor_hub_p)

best_s1 = {"key": (-1.0, -1.0)}
for agg_m in ("inductive_csls", "mean_logit", "mean_prob"):
    hw_list = CONFIG["csls_grid"] if agg_m == "inductive_csls" else [0.0]
    for hw in hw_list:
        cs, cd, cp = evaluate_closed_set_1to30(val_closed_pairs_df, val_closed_p, hub_priors, agg_m, float(hw))
        if (cs["overall_top1_acc"], cs["overall_macro_f1"]) > best_s1["key"]:
            best_s1 = {"key": (cs["overall_top1_acc"], cs["overall_macro_f1"]), "agg": agg_m, "hw": float(hw), "sum": cs, "dom": cd, "preds": cp}

gkf_cal = GroupKFold(n_splits=2)
cal_groups = val_open_df["author_1"].astype(str).to_numpy()
tau_grid = np.linspace(0.30, 0.70, 161)
fold_taus, fold_deltas = [], []
for c_tr, _ in gkf_cal.split(val_open_df, y_val_open, groups=cal_groups):
    tf = float(tau_grid[int(np.argmax([f1_score(y_val_open[c_tr], (val_open_oof_p[c_tr] >= t).astype(int), average="macro") for t in tau_grid]))])
    db, c1b = 0.0, -1.0
    for dv in np.linspace(0.0, 0.14, 71):
        mv = compute_open_pan_metrics(y_val_open[c_tr], val_open_oof_p[c_tr], tau=tf, delta=float(dv))
        if mv["abstain_pct"] <= CONFIG["max_abstention_pct"] and mv["pan_c_at_1"] > c1b:
            c1b, db = mv["pan_c_at_1"], float(dv)
    fold_taus.append(tf)
    fold_deltas.append(db)

best_tau, best_delta = float(np.mean(fold_taus)), float(np.mean(fold_deltas))
val_op_forced = compute_open_pan_metrics(y_val_open, val_open_oof_p, tau=best_tau, delta=0.0)
val_op_abstain = compute_open_pan_metrics(y_val_open, val_open_oof_p, tau=best_tau, delta=best_delta)

val_summary_df = pd.DataFrame([{
    "Model": "10-Expert OOD-Register Gated Meta",
    "Val_Closed_Top1_Acc": best_s1["sum"]["overall_top1_acc"],
    "Val_Closed_Macro_F1": best_s1["sum"]["overall_macro_f1"],
    "Val_Open_ROC_AUC": val_op_abstain["roc_auc"],
    "Val_Open_PAN_c@1": val_op_abstain["pan_c_at_1"],
    "Val_Open_PAN_Brier": val_op_abstain["pan_brier"],
    "Val_Open_PAN_Overall": val_op_abstain["pan_overall"],
}])
val_summary_df.to_csv(OUTPUT_DIR / "validation_round3_final_comparison.csv", index=False)

print(
    f"[val] closed_top1={best_s1['sum']['overall_top1_acc']:.4f} ({best_s1['agg']}, hw={best_s1['hw']}) | "
    f"open_auc={val_op_abstain['roc_auc']:.4f} pan_overall={val_op_abstain['pan_overall']:.4f} "
    f"forced_f1={val_op_forced['forced_macro_f1']:.4f} tau={best_tau:.4f} delta={best_delta:.4f}"
)
display(best_s1["dom"])
'''),
    md_cell("## 7. Locked Test Evaluation"),
    code_cell(r'''if not CONFIG["run_final_test"]:
    print("[test] skipped (set CONFIG['run_final_test'] = True to evaluate locked test split)")
else:
    L_to_10 = score_10_experts(X_test_open)
    L_tc_10 = score_10_experts(X_test_closed)
    test_open_p = predict_gated_meta(build_ood_gate_matrix(X_test_open, L_to_10))
    test_closed_p = predict_gated_meta(build_ood_gate_matrix(X_test_closed, L_tc_10))

    test_cl_sum, test_cl_dom, test_cl_preds = evaluate_closed_set_1to30(
        test_closed_pairs_df, test_closed_p, hub_priors, best_s1["agg"], best_s1["hw"]
    )
    test_op_forced = compute_open_pan_metrics(y_test_open, test_open_p, tau=best_tau, delta=0.0)
    test_op_abstain = compute_open_pan_metrics(y_test_open, test_open_p, tau=best_tau, delta=best_delta)

    st_test_df = pd.DataFrame([
        {"Eval_Stratum": st_nm, "Pairs": len(grp), **compute_open_pan_metrics(y_test_open[grp.index.to_numpy()], test_open_p[grp.index.to_numpy()], best_tau, best_delta)}
        for st_nm, grp in test_open_df.groupby("eval_stratum", sort=False)
    ]).sort_values("Pairs", ascending=False).reset_index(drop=True)

    dom_rows = []
    for dm_nm, grp in test_open_df.groupby("domain", sort=False):
        idx_d = grp.index.to_numpy()
        if len(idx_d) < 20:
            continue
        yd, pf = y_test_open[idx_d], test_open_p[idx_d]
        tn, fp, fn, tp = confusion_matrix(yd, (pf >= best_tau).astype(int), labels=[0, 1]).ravel()
        dm_m = compute_open_pan_metrics(yd, pf, best_tau, best_delta)
        dom_rows.append({
            "Domain": dm_nm, "Pairs": len(idx_d),
            "Test_FPR_%": round(fp / max(fp + tn, 1) * 100.0, 1),
            "Test_FNR_%": round(fn / max(fn + tp, 1) * 100.0, 1),
            "ROC_AUC": dm_m["roc_auc"], "PAN_c@1": dm_m["pan_c_at_1"],
            "PAN_Overall": dm_m["pan_overall"], "Forced_MacroF1": dm_m["forced_macro_f1"],
        })
    dom_test_df = pd.DataFrame(dom_rows).sort_values("Pairs", ascending=False).reset_index(drop=True)

    avg_w = 0.5 * (test_open_df["text_1"].str.split().str.len() + test_open_df["text_2"].str.split().str.len())
    w_bins = pd.cut(avg_w, bins=[0, 60, 180, 400], labels=["Ultra-Short (<60w/doc)", "Short (60-180w/doc)", "Medium (180-350w/doc)"])
    len_test_df = pd.DataFrame([
        {"Doc_Length_Bucket": str(wb), "Pairs": len(grp), **compute_open_pan_metrics(y_test_open[grp.index.to_numpy()], test_open_p[grp.index.to_numpy()], best_tau, best_delta)}
        for wb, grp in test_open_df.groupby(w_bins, observed=False) if len(grp) > 10
    ])

    test_cl_dom.to_csv(OUTPUT_DIR / "final_test_closed_by_domain.csv", index=False)
    test_cl_preds.to_csv(OUTPUT_DIR / "test_closed_predictions_ml.csv", index=False)
    st_test_df.to_csv(OUTPUT_DIR / "final_test_open_by_stratum.csv", index=False)
    dom_test_df.to_csv(OUTPUT_DIR / "final_test_open_by_domain.csv", index=False)
    len_test_df.to_csv(OUTPUT_DIR / "final_test_open_by_length.csv", index=False)

    to_pred_bin = (test_open_p >= best_tau).astype(np.int32)
    to_export = test_open_df[["pair_id", "eval_stratum", "domain", "topic_1", "topic_2", "is_cross_topic", "is_cross_genre", "label"]].copy()
    to_export["ml_sota_prob"] = np.round(test_open_p, 6)
    to_export["ml_sota_pred"] = to_pred_bin
    to_export["is_abstained"] = ((test_open_p > best_tau - best_delta) & (test_open_p < best_tau + best_delta)).astype(np.int32)
    to_export["is_correct"] = (to_pred_bin == y_test_open).astype(np.int32)
    to_export.to_csv(OUTPUT_DIR / "test_open_predictions_ml.csv", index=False)

    final_summary = {
        "track": "01_stylometric_ml",
        "closed_set_matching": {"agg_mode": best_s1["agg"], "csls_hub_weight": best_s1["hw"]},
        "open_thresholds": {"tau": round(best_tau, 4), "delta": round(best_delta, 4)},
        "scenario_1_closed_set_test": test_cl_sum,
        "scenario_2_open_cross_test_forced": test_op_forced,
        "scenario_2_open_cross_test_abstention": test_op_abstain,
    }
    (OUTPUT_DIR / "final_locked_test_summary.json").write_text(json.dumps(final_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[test] closed_top1={test_cl_sum['overall_top1_acc']:.4f} closed_top3={test_cl_sum['overall_top3_acc']:.4f} | "
        f"open_auc={test_op_abstain['roc_auc']:.4f} pan_overall={test_op_abstain['pan_overall']:.4f} "
        f"forced_f1={test_op_forced['forced_macro_f1']:.4f}"
    )
    display(test_cl_dom)
    display(dom_test_df)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    x_d = np.arange(len(test_cl_dom))
    axes[0, 0].bar(x_d - 0.18, test_cl_dom["Top1_Accuracy"], width=0.35, color="#1f77b4", label="Top-1")
    axes[0, 0].bar(x_d + 0.18, test_cl_dom["Top3_Accuracy"], width=0.35, color="#6baed6", label="Top-3")
    axes[0, 0].axhline(test_cl_sum["overall_top1_acc"], ls="--", color="#d62728", label=f"Overall Top-1 ({test_cl_sum['overall_top1_acc']*100:.1f}%)")
    axes[0, 0].set_xticks(x_d); axes[0, 0].set_xticklabels(test_cl_dom["Cohort_Domain"]); axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Closed-Set (1:30) Accuracy by Domain"); axes[0, 0].legend(loc="lower right")

    fpr, tpr, _ = roc_curve(y_test_open, test_open_p)
    pt_c, pp_c = calibration_curve(y_test_open, test_open_p, n_bins=10)
    axes[0, 1].plot(fpr, tpr, lw=2.2, color="#2ca02c", label=f"ROC (AUC={test_op_abstain['roc_auc']:.4f})")
    axes[0, 1].plot(pp_c, pt_c, "s-", lw=1.6, color="#ff7f0e", label=f"Calibration (Brier={test_op_abstain['pan_brier']:.4f})")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].set_title("Open-Set ROC & Reliability"); axes[0, 1].legend(loc="lower right")

    ys = np.arange(len(st_test_df))
    axes[1, 0].barh(ys - 0.18, st_test_df["roc_auc"], height=0.35, color="#2ca02c", label="ROC-AUC")
    axes[1, 0].barh(ys + 0.18, st_test_df["pan_overall"], height=0.35, color="#9467bd", label="PAN Overall")
    axes[1, 0].set_yticks(ys); axes[1, 0].set_yticklabels(st_test_df["Eval_Stratum"], fontsize=8.5); axes[1, 0].set_xlim(0.45, 1.03)
    axes[1, 0].set_title("Performance by Difficulty Stratum"); axes[1, 0].legend(loc="lower right")

    yd = np.arange(len(dom_test_df))
    axes[1, 1].barh(yd - 0.18, dom_test_df["Forced_MacroF1"], height=0.35, color="#1f77b4", label="Macro-F1")
    axes[1, 1].barh(yd + 0.18, dom_test_df["Test_FPR_%"] / 100.0, height=0.35, color="#d62728", label="FPR")
    axes[1, 1].set_yticks(yd); axes[1, 1].set_yticklabels(dom_test_df["Domain"], fontsize=8.5); axes[1, 1].set_xlim(0, 1.03)
    axes[1, 1].set_title("Domain Macro-F1 vs FPR"); axes[1, 1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "test_01_stylometric_ml_4panel.png", bbox_inches="tight")
    plt.show()
'''),
    md_cell(r'''## 8. Portable Inference Export & Self-Verification

OS/NumPy/scikit-learn 버전 차이에 영향을 받지 않도록 순수 Python 내장 타입(`dict`, `list`, `bytes`, `str`, `int`, `float`)으로만 직렬화하여 `model_01_stylometric_ml.joblib`을 저장하고, 리로드 일치성 및 순서 대칭성을 검증합니다.
'''),
    code_cell(r'''def _pack_arr(a: np.ndarray) -> dict:
    ac = np.ascontiguousarray(a)
    return {"bytes": ac.tobytes(), "dtype": str(ac.dtype), "shape": [int(x) for x in ac.shape]}


def _unpack_arr(d: dict) -> np.ndarray:
    return np.frombuffer(d["bytes"], dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def _pack_csr(m: sp.csr_matrix) -> dict:
    mc = m.tocsr()
    return {"data": _pack_arr(mc.data), "indices": _pack_arr(mc.indices), "indptr": _pack_arr(mc.indptr), "shape": [int(x) for x in mc.shape]}


def _unpack_csr(d: dict) -> sp.csr_matrix:
    return sp.csr_matrix((_unpack_arr(d["data"]), _unpack_arr(d["indices"]), _unpack_arr(d["indptr"])), shape=tuple(d["shape"]))


def _pack_tfidf(v: TfidfVectorizer) -> dict:
    return {
        "analyzer": str(v.analyzer), "ngram_range": [int(x) for x in v.ngram_range],
        "token_pattern": str(v.token_pattern) if v.token_pattern is not None else None,
        "sublinear_tf": bool(v.sublinear_tf), "lowercase": bool(v.lowercase),
        "vocab": {str(k): int(i) for k, i in v.vocabulary_.items()}, "idf": _pack_arr(v.idf_.astype(np.float32)),
    }


def _unpack_tfidf(d: dict) -> TfidfVectorizer:
    kw = dict(analyzer=d["analyzer"], ngram_range=tuple(d["ngram_range"]), sublinear_tf=bool(d["sublinear_tf"]), lowercase=bool(d["lowercase"]), vocabulary=d["vocab"], dtype=np.float32)
    if d["analyzer"] == "word" and d["token_pattern"] is not None:
        kw["token_pattern"] = d["token_pattern"]
    v = TfidfVectorizer(**kw)
    v.fit(["placeholder vocabulary calibration document"])
    v.idf_ = _unpack_arr(d["idf"]).astype(np.float32)
    return v


def _pack_linear(clf) -> dict:
    return {"coef": _pack_arr(np.asarray(clf.coef_, dtype=np.float32).ravel()), "intercept": float(np.asarray(clf.intercept_, dtype=np.float64).ravel()[0])}


def _pack_scaler(sc: StandardScaler) -> dict:
    return {"mean": _pack_arr(np.asarray(sc.mean_, dtype=np.float32)), "scale": _pack_arr(np.asarray(sc.scale_, dtype=np.float32))}


def _pack_cb_bytes(cb_model: CatBoostClassifier) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tf:
        tmp_p = Path(tf.name)
    try:
        cb_model.save_model(str(tmp_p), format="cbm")
        return tmp_p.read_bytes()
    finally:
        tmp_p.unlink(missing_ok=True)


def _unpack_cb_bytes(cb_bytes: bytes) -> CatBoostClassifier:
    m = CatBoostClassifier()
    try:
        m.load_model(blob=cb_bytes)
    except TypeError:
        with tempfile.NamedTemporaryFile(suffix=".cbm", delete=False) as tf:
            tf.write(cb_bytes)
            tmp_p = Path(tf.name)
        try:
            m.load_model(str(tmp_p))
        finally:
            tmp_p.unlink(missing_ok=True)
    return m


# Compress Koppel random projection matrix via its PCG64 generator state
n_u_tr = len(pd.factorize(pd.concat([train_df["text_1"], train_df["text_2"]], ignore_index=True), sort=False)[1])
rng_k = np.random.default_rng(int(extractor.seed))
_ = rng_k.choice(n_u_tr, size=min(int(extractor.n_impostors), n_u_tr), replace=False)
koppel_rng_state = copy.deepcopy(rng_k.bit_generator.state)
replayed = rng_k.standard_normal(extractor.koppel_proj_pnoise_.shape, dtype=np.float32) / math.sqrt(24.0)
if np.allclose(replayed, extractor.koppel_proj_pnoise_, atol=1e-6):
    koppel_proj_spec = {"mode": "pcg64_rng", "rng_state": koppel_rng_state, "shape": [int(x) for x in extractor.koppel_proj_pnoise_.shape]}
else:
    koppel_proj_spec = {"mode": "fp16_array", "arr": _pack_arr(extractor.koppel_proj_pnoise_.astype(np.float16))}
del replayed

artifact = {
    "artifact_version": "stylometric_ml_v1_portable",
    "thresholds": {
        "tau": float(best_tau), "delta": float(best_delta),
        "closed_agg_mode": str(best_s1["agg"]), "closed_hub_weight": float(best_s1["hw"]),
    },
    "extractor": {
        "params": {
            "surf_hi_dim": int(extractor.surf_hi_dim), "pnoise_hi_dim": int(extractor.pnoise_hi_dim),
            "dist_hi_dim": int(extractor.dist_hi_dim), "tree_char_dim": int(extractor.tree_char_dim),
            "tree_pnoise_dim": int(extractor.tree_pnoise_dim), "tree_char_prod_dim": int(extractor.tree_char_prod_dim),
            "tree_pnoise_prod_dim": int(extractor.tree_pnoise_prod_dim), "n_impostors": int(extractor.n_impostors),
            "n_koppel_bags": int(extractor.n_koppel_bags), "n_oof_folds": int(extractor.n_oof_folds), "seed": int(extractor.seed),
        },
        "function_words_250": list(FUNCTION_WORDS_250),
        "doc_profile_62_names": list(DOC_PROFILE_62_NAMES),
        "vec_surf_hi": _pack_tfidf(extractor.vec_surf_hi),
        "vec_pnoise_hi": _pack_tfidf(extractor.vec_pnoise_hi),
        "vec_dist_hi": _pack_tfidf(extractor.vec_dist_hi),
        "vec_tree_char": _pack_tfidf(extractor.vec_tree_char),
        "vec_tree_pnoise": _pack_tfidf(extractor.vec_tree_pnoise),
        "s1_models": {str(k): {"clf": _pack_linear(v[0]), "platt": _pack_linear(v[1]) if v[1] is not None else None} for k, v in extractor.s1_models_.items()},
        "fw_mean": _pack_arr(extractor.fw_mean_), "fw_std": _pack_arr(extractor.fw_std_),
        "imp_surf": _pack_csr(extractor.imp_surf_), "imp_pnoise": _pack_csr(extractor.imp_pnoise_),
        "imp_fw_unit": _pack_arr(extractor.imp_fw_unit_),
        "koppel_proj_pnoise": koppel_proj_spec, "koppel_imp_pnoise": _pack_arr(extractor.koppel_imp_pnoise_),
        "koppel_proj_fw": _pack_arr(extractor.koppel_proj_fw_), "koppel_imp_fw": _pack_arr(extractor.koppel_imp_fw_),
        "tree_char_order": _pack_arr(extractor.tree_char_order_), "tree_pnoise_order": _pack_arr(extractor.tree_pnoise_order_),
        "feature_names": [str(x) for x in extractor.feature_names_],
        "group_slices": {str(k): [int(v[0]), int(v[1])] for k, v in extractor.group_slices_.items()},
        "n_dense": int(extractor.n_dense_),
    },
    "col_subsets": {
        "selected_cols": _pack_arr(selected_cols), "selected_dense_cols": _pack_arr(selected_dense_cols),
        "top_lgb_cols": _pack_arr(top_lgb_cols), "cb_hybrid_cols": _pack_arr(cb_hybrid_cols),
        "top_dense_cols_r2": _pack_arr(top_dense_cols_r2), "genre_blind_cols": _pack_arr(genre_blind_cols),
    },
    "experts_10": {
        "r1_lgb_gbdt_str": lgb_gbdt.booster_.model_to_string(),
        "r1_lgb_goss_str": lgb_goss.booster_.model_to_string(),
        "r1_cat_cbm": _pack_cb_bytes(cat_full),
        "r2_lgb_gbdt_str": r2_lgb_gbdt.booster_.model_to_string(),
        "r2_lgb_goss_str": r2_lgb_goss.booster_.model_to_string(),
        "r2_cb_hyb_cbm": _pack_cb_bytes(r2_cb_hyb),
        "r2_cb_d4_cbm": _pack_cb_bytes(r2_cb_d4),
        "r2_scaler": _pack_scaler(r2_scaler),
        "r2_meta_lr": _pack_linear(r2_meta_lr),
        "r3_cb_genre_blind_cbm": _pack_cb_bytes(r3_cb_genre_blind),
        "r3_lgb_ct_str": r3_lgb_ct.booster_.model_to_string(),
    },
    "ood_gate_folds": [
        {"scaler": _pack_scaler(sc_k), "lr": _pack_linear(lr_k), "cb_cbm": _pack_cb_bytes(cb_k), "w_lr": 0.45, "w_cb": 0.55}
        for sc_k, lr_k, cb_k in gate_models
    ],
    "closed_hub_priors": {f"{d}|||{c}": float(v) for (d, c), v in hub_priors.items()},
    "closed_anchors": anchors_df[["cohort_domain", "candidate_author_id", "anchor_idx", "text"]].to_dict(orient="records"),
}


def _check_pure_builtins(obj, path: str = "root") -> None:
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return
    if isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _check_pure_builtins(item, f"{path}[{i}]")
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), f"non-str key at {path}: {type(k)}"
            _check_pure_builtins(v, f"{path}.{k}")
        return
    raise TypeError(f"non-builtin object at {path}: {type(obj)}")


_check_pure_builtins(artifact)
model_out = Path.cwd() / "model_01_stylometric_ml.joblib"
joblib.dump(artifact, model_out, compress=3)

# Round-trip symmetry check on 8 pairs
loaded = joblib.load(model_out)


class _LinearView:
    def __init__(self, d: dict):
        self.coef_ = _unpack_arr(d["coef"]).astype(np.float32)
        self.intercept_ = float(d["intercept"])

    def decision_function(self, X):
        return np.asarray(X @ self.coef_, dtype=np.float32).ravel() + self.intercept_

    def predict_proba(self, X):
        p1 = expit(np.clip(self.decision_function(X), -25.0, 25.0)).astype(np.float32)
        return np.column_stack([1.0 - p1, p1])


ep = loaded["extractor"]
re_ext = StylometricPairExtractor(**ep["params"], n_jobs=1)
re_ext.vec_surf_hi = _unpack_tfidf(ep["vec_surf_hi"])
re_ext.vec_pnoise_hi = _unpack_tfidf(ep["vec_pnoise_hi"])
re_ext.vec_dist_hi = _unpack_tfidf(ep["vec_dist_hi"])
re_ext.vec_tree_char = _unpack_tfidf(ep["vec_tree_char"])
re_ext.vec_tree_pnoise = _unpack_tfidf(ep["vec_tree_pnoise"])
re_ext.s1_models_ = {k: (_LinearView(v["clf"]), _LinearView(v["platt"]) if v["platt"] else None) for k, v in ep["s1_models"].items()}
re_ext.fw_mean_, re_ext.fw_std_ = _unpack_arr(ep["fw_mean"]), _unpack_arr(ep["fw_std"])
re_ext.imp_surf_, re_ext.imp_pnoise_, re_ext.imp_fw_unit_ = _unpack_csr(ep["imp_surf"]), _unpack_csr(ep["imp_pnoise"]), _unpack_arr(ep["imp_fw_unit"])
if ep["koppel_proj_pnoise"]["mode"] == "pcg64_rng":
    rng_r = np.random.default_rng()
    rng_r.bit_generator.state = ep["koppel_proj_pnoise"]["rng_state"]
    re_ext.koppel_proj_pnoise_ = rng_r.standard_normal(tuple(ep["koppel_proj_pnoise"]["shape"]), dtype=np.float32) / math.sqrt(24.0)
else:
    re_ext.koppel_proj_pnoise_ = _unpack_arr(ep["koppel_proj_pnoise"]["arr"]).astype(np.float32)
re_ext.koppel_imp_pnoise_ = _unpack_arr(ep["koppel_imp_pnoise"])
re_ext.koppel_proj_fw_, re_ext.koppel_imp_fw_ = _unpack_arr(ep["koppel_proj_fw"]), _unpack_arr(ep["koppel_imp_fw"])
re_ext.tree_char_order_, re_ext.tree_pnoise_order_ = _unpack_arr(ep["tree_char_order"]), _unpack_arr(ep["tree_pnoise_order"])
re_ext.feature_names_ = ep["feature_names"]
re_ext.group_slices_ = {k: tuple(v) for k, v in ep["group_slices"].items()}
re_ext.n_dense_ = int(ep["n_dense"])

ex = loaded["experts_10"]
re_boosters = [lgb.Booster(model_str=ex[k]) for k in ("r1_lgb_gbdt_str", "r1_lgb_goss_str", "r2_lgb_gbdt_str", "r2_lgb_goss_str", "r3_lgb_ct_str")]
re_cbs = [_unpack_cb_bytes(ex[k]) for k in ("r1_cat_cbm", "r2_cb_hyb_cbm", "r2_cb_d4_cbm", "r3_cb_genre_blind_cbm")]
re_sc_m, re_sc_s, re_lin = _unpack_arr(ex["r2_scaler"]["mean"]), _unpack_arr(ex["r2_scaler"]["scale"]), _LinearView(ex["r2_meta_lr"])
re_gates = [(_unpack_arr(g["scaler"]["mean"]), _unpack_arr(g["scaler"]["scale"]), _LinearView(g["lr"]), _unpack_cb_bytes(g["cb_cbm"])) for g in loaded["ood_gate_folds"]]
cs = {k: _unpack_arr(v) for k, v in loaded["col_subsets"].items()}


def _infer_reloaded(df_in: pd.DataFrame) -> np.ndarray:
    Xc = re_ext.transform(df_in, "verify")
    ixc = build_interaction_features(Xc)
    Xr2l = sp.hstack([Xc[:, cs["top_lgb_cols"]], sp.csr_matrix(ixc)], format="csr")
    Xr2h = np.hstack([Xc[:, cs["cb_hybrid_cols"]].toarray().astype(np.float32), ixc])
    Xr2d = np.hstack([Xc[:, cs["top_dense_cols_r2"]].toarray().astype(np.float32), ixc])
    Xr2s = np.clip((np.nan_to_num(Xr2d, nan=0.0, posinf=0.0, neginf=0.0) - re_sc_m) / re_sc_s, -8.0, 8.0)
    Xgbd = Xc[:, cs["genre_blind_cols"]].toarray().astype(np.float32)
    L10 = np.column_stack([
        safe_logit(re_boosters[0].predict(Xc[:, cs["selected_cols"]])),
        safe_logit(re_boosters[1].predict(Xc[:, cs["selected_cols"]])),
        safe_logit(re_cbs[0].predict_proba(Xc[:, cs["selected_dense_cols"]].toarray().astype(np.float32))[:, 1]),
        safe_logit(re_boosters[2].predict(Xr2l)),
        safe_logit(re_boosters[3].predict(Xr2l)),
        safe_logit(re_cbs[1].predict_proba(Xr2h)[:, 1]),
        safe_logit(re_cbs[2].predict_proba(Xr2d)[:, 1]),
        safe_logit(re_lin.predict_proba(Xr2s)[:, 1]),
        safe_logit(re_cbs[3].predict_proba(Xgbd)[:, 1]),
        safe_logit(re_boosters[4].predict(Xr2l)),
    ])
    Mg = build_ood_gate_matrix(Xc, L10)
    return np.mean([0.45 * glr.predict_proba(np.clip((np.nan_to_num(Mg, nan=0.0, posinf=0.0, neginf=0.0) - gm) / gs, -8.0, 8.0))[:, 1] + 0.55 * gcb.predict_proba(Mg)[:, 1] for gm, gs, glr, gcb in re_gates], axis=0)


probe = val_open_df[["text_1", "text_2"]].iloc[:8].copy().reset_index(drop=True)
t0 = time.perf_counter()
p_fwd = _infer_reloaded(probe)
ms_per_pair = (time.perf_counter() - t0) * 1000.0 / len(probe)
p_bwd = _infer_reloaded(probe.rename(columns={"text_1": "text_2", "text_2": "text_1"}))
sym_err = float(np.max(np.abs(p_fwd - p_bwd)))
assert sym_err < 1e-4

bundle_out = Path.cwd() / "results_01_stylometric_ml.zip"
with zipfile.ZipFile(bundle_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fp in sorted(OUTPUT_DIR.rglob("*")):
        if fp.is_file() and ("feature_cache" not in fp.parts) and fp.suffix.lower() in (".csv", ".json", ".png"):
            zf.write(fp, arcname=str(fp.relative_to(ROOT_OUT)))

print(f"[export] model={model_out.name} ({model_out.stat().st_size/1024**2:.2f} MB) | sym_err={sym_err:.2e} | latency={ms_per_pair:.1f} ms/pair")
print(f"[export] bundle={bundle_out.name} ({bundle_out.stat().st_size/1024:.1f} KB)")
'''),
]


def build_notebook(out_path: Path) -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "colab": {"name": out_path.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({len(CELLS)} cells, {out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_notebook(Path(r"D:\Projects\NLU-Authorship-Verification\notebooks\colab\train_01_stylometric_ml.ipynb"))
