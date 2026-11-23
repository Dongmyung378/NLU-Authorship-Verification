"""Standalone CPU inference engine for Track 01: 6-Layer Stylometric ML + OOD-Register Gate."""
from __future__ import annotations

import math
import re
import tempfile
import time
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoostClassifier
from scipy.special import expit
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nlu_authorship.artifacts import find_artifact

FUNCTION_WORDS_250 = list(dict.fromkeys([
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


def _safe_logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(pc / (1.0 - pc))


def _unpack_arr(d: dict) -> np.ndarray:
    return np.frombuffer(d["bytes"], dtype=np.dtype(d["dtype"])).reshape(tuple(d["shape"])).copy()


def _unpack_csr(d: dict) -> sp.csr_matrix:
    return sp.csr_matrix((_unpack_arr(d["data"]), _unpack_arr(d["indices"]), _unpack_arr(d["indptr"])), shape=tuple(d["shape"]))


def _unpack_tfidf(d: dict) -> TfidfVectorizer:
    kw: dict[str, Any] = dict(
        analyzer=d["analyzer"],
        ngram_range=tuple(d["ngram_range"]),
        sublinear_tf=bool(d["sublinear_tf"]),
        lowercase=bool(d["lowercase"]),
        vocabulary=d["vocab"],
        dtype=np.float32,
    )
    if d["analyzer"] == "word" and d["token_pattern"] is not None:
        kw["token_pattern"] = d["token_pattern"]
    v = TfidfVectorizer(**kw)
    v.fit(["placeholder vocabulary calibration document"])
    v.idf_ = _unpack_arr(d["idf"]).astype(np.float32)
    return v


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


class _LinearView:
    def __init__(self, d: dict):
        self.coef_ = _unpack_arr(d["coef"]).astype(np.float32)
        self.intercept_ = float(d["intercept"])

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X @ self.coef_, dtype=np.float32).ravel() + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        p1 = expit(np.clip(self.decision_function(X), -25.0, 25.0)).astype(np.float32)
        return np.column_stack([1.0 - p1, p1])


def _ensure_nltk_tagger():
    for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"):
        try:
            nltk.data.find(f"taggers/{res}")
            break
        except LookupError:
            try:
                nltk.download(res, quiet=True)
                break
            except Exception:
                continue
    from nltk.tag.perceptron import PerceptronTagger
    return PerceptronTagger()


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


class StylometricMLEngine:
    """Loads pure-builtin model_01_stylometric_ml.joblib and scores English passage pairs on CPU."""

    def __init__(self, artifact_path: Path | None = None):
        path = artifact_path or find_artifact("ml")
        if path is None or not path.is_file():
            raise FileNotFoundError("model_01_stylometric_ml.joblib not found. Run `python verify_author.py --download ml` first.")
        self.artifact_path = path
        self.tagger = _ensure_nltk_tagger()

        loaded = joblib.load(path)
        self.tau = float(loaded["thresholds"]["tau"])
        self.delta = float(loaded["thresholds"]["delta"])

        ep = loaded["extractor"]
        self.params = ep["params"]
        self.n_koppel_bags = int(self.params["n_koppel_bags"])
        self.tree_char_prod_dim = int(self.params["tree_char_prod_dim"])
        self.tree_pnoise_prod_dim = int(self.params["tree_pnoise_prod_dim"])

        self.fw_vec = CountVectorizer(lowercase=True, token_pattern=r"[a-z]+(?:'[a-z]+)?", vocabulary=FUNCTION_WORDS_250, dtype=np.float32)
        self.vec_surf_hi = _unpack_tfidf(ep["vec_surf_hi"])
        self.vec_pnoise_hi = _unpack_tfidf(ep["vec_pnoise_hi"])
        self.vec_dist_hi = _unpack_tfidf(ep["vec_dist_hi"])
        self.vec_tree_char = _unpack_tfidf(ep["vec_tree_char"])
        self.vec_tree_pnoise = _unpack_tfidf(ep["vec_tree_pnoise"])

        self.s1_models = {k: (_LinearView(v["clf"]), _LinearView(v["platt"]) if v["platt"] else None) for k, v in ep["s1_models"].items()}
        self.fw_mean = _unpack_arr(ep["fw_mean"])
        self.fw_std = _unpack_arr(ep["fw_std"])
        self.imp_surf = _unpack_csr(ep["imp_surf"])
        self.imp_pnoise = _unpack_csr(ep["imp_pnoise"])
        self.imp_fw_unit = _unpack_arr(ep["imp_fw_unit"])

        kp = ep["koppel_proj_pnoise"]
        if kp["mode"] == "pcg64_rng":
            rng_r = np.random.default_rng()
            rng_r.bit_generator.state = kp["rng_state"]
            self.koppel_proj_pnoise = rng_r.standard_normal(tuple(kp["shape"]), dtype=np.float32) / math.sqrt(24.0)
        else:
            self.koppel_proj_pnoise = _unpack_arr(kp["arr"]).astype(np.float32)
        self.koppel_imp_pnoise = _unpack_arr(ep["koppel_imp_pnoise"])
        self.koppel_proj_fw = _unpack_arr(ep["koppel_proj_fw"])
        self.koppel_imp_fw = _unpack_arr(ep["koppel_imp_fw"])
        self.tree_char_order = _unpack_arr(ep["tree_char_order"])
        self.tree_pnoise_order = _unpack_arr(ep["tree_pnoise_order"])
        self.feature_names = list(ep["feature_names"])
        self.name_to_col = {nm: idx for idx, nm in enumerate(self.feature_names)}

        ex = loaded["experts_10"]
        self.boosters = [lgb.Booster(model_str=ex[k]) for k in ("r1_lgb_gbdt_str", "r1_lgb_goss_str", "r2_lgb_gbdt_str", "r2_lgb_goss_str", "r3_lgb_ct_str")]
        self.cbs = [_unpack_cb_bytes(ex[k]) for k in ("r1_cat_cbm", "r2_cb_hyb_cbm", "r2_cb_d4_cbm", "r3_cb_genre_blind_cbm")]
        self.r2_sc_m = _unpack_arr(ex["r2_scaler"]["mean"])
        self.r2_sc_s = _unpack_arr(ex["r2_scaler"]["scale"])
        self.r2_lin = _LinearView(ex["r2_meta_lr"])

        self.gates = [
            (_unpack_arr(g["scaler"]["mean"]), _unpack_arr(g["scaler"]["scale"]), _LinearView(g["lr"]), _unpack_cb_bytes(g["cb_cbm"]), float(g.get("w_lr", 0.45)), float(g.get("w_cb", 0.55)))
            for g in loaded["ood_gate_folds"]
        ]
        self.cs = {k: _unpack_arr(v) for k, v in loaded["col_subsets"].items()}

    def _extract_doc_views(self, texts: list[str]):
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

            tagged_words = self.tagger.tag(word_tokens) if word_tokens else []
            n_words = len(word_tokens)
            pos_lookup, pos_seq = {}, []
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

            pnoise_tokens, oxford_comma_cnt = [], 0
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

            passive_cnt = bg_jj_nn = bg_rb_vj = bg_nn_nn = bg_in_dt = bg_prp_vb = 0
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
                    if lower_words[i] in BE_VERBS and (p2 == "VBN" or (i + 2 < n_words and p2.startswith("RB") and pos_seq[i + 2] == "VBN")):
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
                short_w, long_w = float(np.mean(w_lens <= 3)), float(np.mean(w_lens >= 8))
                allcaps_w = sum(1 for w in word_tokens if len(w) >= 2 and w.isupper()) / n_words

                syls = np.fromiter((_count_syllables(w) for w in word_tokens), dtype=np.float32, count=n_words)
                mean_syl, poly_syl = float(syls.mean()), float(np.mean(syls >= 3))
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
                text.count(",") * inv_c1k, text.count(".") * inv_c1k, text.count(chr(33)) * inv_c1k, text.count("?") * inv_c1k,
                (text.count(";") + text.count(":")) * inv_c1k,
                (text.count('"') + text.count("'")) * inv_c1k,
                (text.count("-") + text.count("(") + text.count(")")) * inv_c1k,
                sent_burst, word_burst, mean_syl, poly_syl, coleman_liau,
                mid_sent_caps * inv_w, c_fp1 * inv_w, saxon_pref, uk_us_bias, dbl_space_blank,
            ]
        return pos_noise_list, distort_list, comp_lens, profiles

    @staticmethod
    def _pred_2col(model_pair, X_mat: sp.csr_matrix) -> np.ndarray:
        clf, platt = model_pair
        margins = clf.decision_function(X_mat).astype(np.float32)
        if platt is not None:
            probs = platt.predict_proba(margins.reshape(-1, 1))[:, 1].astype(np.float32)
        else:
            probs = expit(np.clip(margins, -25.0, 25.0)).astype(np.float32)
        return np.column_stack([margins, probs])

    def _transform_pair(self, text_1: str, text_2: str) -> sp.csr_matrix:
        unique_texts = [text_1] if text_1 == text_2 else [text_1, text_2]
        idx1 = np.array([0], dtype=np.int32)
        idx2 = np.array([0 if text_1 == text_2 else 1], dtype=np.int32)

        pnoise_u, dist_u, comp_u, prof_u = self._extract_doc_views(unique_texts)
        ms = self.vec_surf_hi.transform(unique_texts)
        X_s = _sym_sparse_pair(ms[idx1], ms[idx2])
        s1_slr, s1_ssvm = self._pred_2col(self.s1_models["surf_lr"], X_s), self._pred_2col(self.s1_models["surf_svm"], X_s)

        mp = self.vec_pnoise_hi.transform(pnoise_u)
        X_p = _sym_sparse_pair(mp[idx1], mp[idx2])
        s1_plr, s1_psvm = self._pred_2col(self.s1_models["pnoise_lr"], X_p), self._pred_2col(self.s1_models["pnoise_svm"], X_p)

        md = self.vec_dist_hi.transform(dist_u)
        s1_dlr = self._pred_2col(self.s1_models["dist_lr"], _sym_sparse_pair(md[idx1], md[idx2]))
        s1_18 = _augment_s1_interactions(np.hstack([s1_slr, s1_ssvm, s1_plr, s1_psvm, s1_dlr]).astype(np.float32))

        fw_rates = self.fw_vec.transform(unique_texts).toarray().astype(np.float32) / np.maximum(prof_u[:, 33:34], 1.0)
        fw_z = (fw_rates - self.fw_mean) / self.fw_std

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

        c_surf = _cohort_3col(np.asarray(ms[idx1].multiply(ms[idx2]).sum(axis=1), dtype=np.float32), (ms @ self.imp_surf.T).toarray().astype(np.float32))
        c_pn = _cohort_3col(np.asarray(mp[idx1].multiply(mp[idx2]).sum(axis=1), dtype=np.float32), (mp @ self.imp_pnoise.T).toarray().astype(np.float32))
        fw_unit = (fw_z / (np.linalg.norm(fw_z, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        c_fw = _cohort_3col(np.sum(fw_unit[idx1] * fw_unit[idx2], axis=1, dtype=np.float32), fw_unit @ self.imp_fw_unit.T)

        def _koppel(u_proj, imp_proj):
            u_n = (u_proj / (np.linalg.norm(u_proj, axis=2, keepdims=True) + 1e-8)).astype(np.float32)
            blk = np.einsum("nbd,mbd->nbm", u_n, imp_proj, optimize=True)
            blk[blk > 0.9995] = -1.0
            max_imp = np.max(blk, axis=2)
            pair_sim = np.sum(u_n[idx1] * u_n[idx2], axis=2)
            w1 = (pair_sim > max_imp[idx1]).astype(np.float32)
            w2 = (pair_sim > max_imp[idx2]).astype(np.float32)
            return (0.5 * (w1.mean(axis=1) + w2.mean(axis=1))).reshape(-1, 1)

        kop_pn = _koppel((mp @ self.koppel_proj_pnoise).reshape(mp.shape[0], self.n_koppel_bags, -1), self.koppel_imp_pnoise)
        kop_fw = _koppel(fw_z[:, self.koppel_proj_fw], self.koppel_imp_fw)
        feat_imp = np.hstack([c_surf, c_pn, c_fw, kop_pn, kop_fw, 0.5 * (kop_pn + kop_fw), c_pn[:, 0:1] - c_surf[:, 0:1], (c_surf[:, 0:1] + c_pn[:, 0:1] + c_fw[:, 0:1]) / 3.0]).astype(np.float32)

        t1, t2 = unique_texts[idx1[0]], unique_texts[idx2[0]]
        p1s, p2s = pnoise_u[idx1[0]], pnoise_u[idx2[0]]
        b1, b2 = t1.encode("utf-8"), t2.encode("utf-8")
        cr1, cr2 = float(comp_u[idx1[0], 0]), float(comp_u[idx2[0], 0])
        c12 = 0.5 * (len(zlib.compress(b1 + b"\n" + b2, level=6)) + len(zlib.compress(b2 + b"\n" + b1, level=6)))
        bp1, bp2 = p1s.encode("utf-8"), p2s.encode("utf-8")
        cp1, cp2 = float(comp_u[idx1[0], 1]), float(comp_u[idx2[0], 1])
        cp12 = 0.5 * (len(zlib.compress(bp1 + b"\n" + bp2, level=6)) + len(zlib.compress(bp2 + b"\n" + bp1, level=6)))
        feat_ncd = np.array([[
            (c12 - min(cr1, cr2)) / max(cr1, cr2, 1.0),
            1.0 - ((cr1 + cr2 - c12) / math.sqrt(max(cr1 * cr2, 1.0))),
            abs(cr1 / max(len(b1), 1) - cr2 / max(len(b2), 1)),
            (cp12 - min(cp1, cp2)) / max(cp1, cp2, 1.0),
            1.0 - ((cp1 + cp2 - cp12) / math.sqrt(max(cp1 * cp2, 1.0))),
            abs(cp1 / max(len(bp1), 1) - cp2 / max(len(bp2), 1)),
        ]], dtype=np.float32)

        p1, p2 = prof_u[idx1], prof_u[idx2]
        p_abs = np.abs(p1 - p2)
        p_rel = p_abs / (np.abs(p1) + np.abs(p2) + 1e-6)
        w1, w2 = p1[:, 33:34], p2[:, 33:34]
        min_w, max_w = np.minimum(w1, w2), np.maximum(w1, w2)
        feat_g4 = np.hstack([p_abs, p_rel, 0.5 * (p1 + p2), min_w, max_w, min_w / np.maximum(max_w, 1.0), np.log1p(w1 + w2)]).astype(np.float32)

        f1, f2, z1, z2 = fw_rates[idx1], fw_rates[idx2], fw_z[idx1], fw_z[idx2]
        z_abs = np.abs(z1 - z2)
        f1p = (f1 + 1e-7) / np.sum(f1 + 1e-7, axis=1, keepdims=True)
        f2p = (f2 + 1e-7) / np.sum(f2 + 1e-7, axis=1, keepdims=True)
        mp_f = 0.5 * (f1p + f2p)
        feat_g5 = np.hstack([
            z_abs,
            z_abs.mean(axis=1, keepdims=True),
            np.sqrt(np.mean((z1 - z2) ** 2, axis=1, keepdims=True)),
            1.0 - np.sum((z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)) * (z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)), axis=1, keepdims=True),
            0.5 * (np.sum(f1p * np.log2(f1p / mp_f), axis=1, keepdims=True) + np.sum(f2p * np.log2(f2p / mp_f), axis=1, keepdims=True)),
            np.sum((f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)) * (f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)), axis=1, keepdims=True),
            np.sum(np.abs(f1 - f2), axis=1, keepdims=True),
            np.mean(np.abs(f1 - f2) / (f1 + f2 + 1e-7), axis=1, keepdims=True),
            np.sum(np.minimum(f1, f2), axis=1, keepdims=True),
        ]).astype(np.float32)

        tc_u = self.vec_tree_char.transform(unique_texts)[:, self.tree_char_order].tocsr()
        c1, c2 = tc_u[idx1], tc_u[idx2]
        c_diff = (c1 - c2).tocsr()
        c_diff.data = np.abs(c_diff.data).astype(np.float32)
        c_prod = c1[:, : min(self.tree_char_prod_dim, c1.shape[1])].multiply(c2[:, : min(self.tree_char_prod_dim, c1.shape[1])]).tocsr().astype(np.float32)

        c_cos = np.asarray(c1.multiply(c2).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c_l1 = np.asarray(c_diff.sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c1_sq = np.asarray(c1.multiply(c1).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c2_sq = np.asarray(c2.multiply(c2).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c_l2 = np.sqrt(np.maximum(c1_sq + c2_sq - 2.0 * c_cos, 0.0))
        c_hist = 0.5 * np.maximum(np.asarray(c1.sum(axis=1), dtype=np.float32).reshape(-1, 1) + np.asarray(c2.sum(axis=1), dtype=np.float32).reshape(-1, 1) - c_l1, 0.0)
        b1m, b2m = (c1 > 0).astype(np.float32), (c2 > 0).astype(np.float32)
        inter = np.asarray(b1m.multiply(b2m).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        bc1, bc2 = np.asarray(b1m.sum(axis=1), dtype=np.float32).reshape(-1, 1), np.asarray(b2m.sum(axis=1), dtype=np.float32).reshape(-1, 1)
        c_jacc = inter / np.maximum(bc1 + bc2 - inter, 1.0)
        c_dice = (2.0 * inter) / np.maximum(bc1 + bc2, 1.0)

        def _bcos(sc: int, ec: int) -> np.ndarray:
            if ec <= sc:
                return np.zeros((1, 1), dtype=np.float32)
            s1m, s2m = c1[:, sc:ec], c2[:, sc:ec]
            d = np.asarray(s1m.multiply(s2m).sum(axis=1), dtype=np.float32).reshape(-1, 1)
            n1 = np.sqrt(np.asarray(s1m.multiply(s1m).sum(axis=1), dtype=np.float32).reshape(-1, 1))
            n2 = np.sqrt(np.asarray(s2m.multiply(s2m).sum(axis=1), dtype=np.float32).reshape(-1, 1))
            return d / (n1 * n2 + 1e-8)

        nc = c1.shape[1]
        bt, bm, br = _bcos(0, min(400, nc)), _bcos(min(400, nc), min(1500, nc)), _bcos(min(1500, nc), nc)

        tp_u = self.vec_tree_pnoise.transform(pnoise_u)[:, self.tree_pnoise_order].tocsr()
        pn1, pn2 = tp_u[idx1], tp_u[idx2]
        pn_diff = (pn1 - pn2).tocsr()
        pn_diff.data = np.abs(pn_diff.data).astype(np.float32)
        pn_prod = pn1[:, : min(self.tree_pnoise_prod_dim, pn1.shape[1])].multiply(pn2[:, : min(self.tree_pnoise_prod_dim, pn1.shape[1])]).tocsr().astype(np.float32)

        pn_cos = np.asarray(pn1.multiply(pn2).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        pn_l1 = np.asarray(pn_diff.sum(axis=1), dtype=np.float32).reshape(-1, 1)
        pn_hist = 0.5 * np.maximum(np.asarray(pn1.sum(axis=1), dtype=np.float32).reshape(-1, 1) + np.asarray(pn2.sum(axis=1), dtype=np.float32).reshape(-1, 1) - pn_l1, 0.0)
        pb1, pb2 = (pn1 > 0).astype(np.float32), (pn2 > 0).astype(np.float32)
        p_inter = np.asarray(pb1.multiply(pb2).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        pn_jacc = p_inter / np.maximum(np.asarray(pb1.sum(axis=1), dtype=np.float32).reshape(-1, 1) + np.asarray(pb2.sum(axis=1), dtype=np.float32).reshape(-1, 1) - p_inter, 1.0)
        dist_cos = np.asarray(md[idx1].multiply(md[idx2]).sum(axis=1), dtype=np.float32).reshape(-1, 1)
        hi_surf_cos = np.asarray(ms[idx1].multiply(ms[idx2]).sum(axis=1), dtype=np.float32).reshape(-1, 1)

        feat_g6 = np.hstack([c_cos, c_l1, c_l2, c_hist, c_jacc, c_dice, bt, bm, br, pn_cos, pn_l1, pn_hist, pn_jacc, dist_cos, hi_surf_cos]).astype(np.float32)
        dense_mat = np.hstack([s1_18, feat_imp, feat_ncd, feat_g4, feat_g5, feat_g6], dtype=np.float32)
        return sp.hstack([sp.csr_matrix(dense_mat), c_diff, pn_diff, c_prod, pn_prod], format="csr", dtype=np.float32)

    def _col(self, Xc: sp.csr_matrix, nm: str) -> np.ndarray:
        return Xc[:, self.name_to_col[nm]].toarray().ravel().astype(np.float32)

    def _score_single(self, text_1: str, text_2: str) -> tuple[float, dict[str, Any]]:
        Xc = self._transform_pair(text_1, text_2)
        p_surf, p_pn, m_mean = self._col(Xc, "s1_surf60k_svm_prob"), self._col(Xc, "s1_pnoise40k_svm_prob"), self._col(Xc, "s1_all5_margin_mean")
        imp_s_t10, imp_s_rk, imp_s_as = self._col(Xc, "imp_surf_top10_margin"), self._col(Xc, "imp_surf_mut_rank"), self._col(Xc, "imp_surf_asnorm")
        imp_p_t10, imp_p_rk, imp_p_as = self._col(Xc, "imp_pnoise_top10_margin"), self._col(Xc, "imp_pnoise_mut_rank"), self._col(Xc, "imp_pnoise_asnorm")
        cbc_raw, ncd_raw, ncd_pn, min_w = self._col(Xc, "cbc_raw_zlib"), self._col(Xc, "ncd_raw_zlib"), self._col(Xc, "ncd_pnoise_zlib"), self._col(Xc, "pair_min_words")

        ixc = np.column_stack([
            p_pn * imp_s_rk, p_surf * imp_p_rk, p_pn * imp_p_rk,
            imp_p_t10 - imp_s_t10, 0.5 * (imp_p_t10 + imp_s_t10),
            np.minimum(imp_s_rk, imp_p_rk), np.minimum(p_surf, p_pn),
            (2.0 * p_surf * p_pn) / (p_surf + p_pn + 1e-6),
            (1.0 - cbc_raw) * m_mean, ncd_pn - ncd_raw,
            m_mean * np.log1p(np.maximum(min_w, 0.0)), np.minimum(imp_s_as, imp_p_as),
        ]).astype(np.float32)

        cs = self.cs
        Xr2l = sp.hstack([Xc[:, cs["top_lgb_cols"]], sp.csr_matrix(ixc)], format="csr")
        Xr2h = np.hstack([Xc[:, cs["cb_hybrid_cols"]].toarray().astype(np.float32), ixc])
        Xr2d = np.hstack([Xc[:, cs["top_dense_cols_r2"]].toarray().astype(np.float32), ixc])
        Xr2s = np.clip((np.nan_to_num(Xr2d, nan=0.0, posinf=0.0, neginf=0.0) - self.r2_sc_m) / self.r2_sc_s, -8.0, 8.0)
        Xgbd = Xc[:, cs["genre_blind_cols"]].toarray().astype(np.float32)

        L10 = np.column_stack([
            _safe_logit(self.boosters[0].predict(Xc[:, cs["selected_cols"]])),
            _safe_logit(self.boosters[1].predict(Xc[:, cs["selected_cols"]])),
            _safe_logit(self.cbs[0].predict_proba(Xc[:, cs["selected_dense_cols"]].toarray().astype(np.float32))[:, 1]),
            _safe_logit(self.boosters[2].predict(Xr2l)),
            _safe_logit(self.boosters[3].predict(Xr2l)),
            _safe_logit(self.cbs[1].predict_proba(Xr2h)[:, 1]),
            _safe_logit(self.cbs[2].predict_proba(Xr2d)[:, 1]),
            _safe_logit(self.r2_lin.predict_proba(Xr2s)[:, 1]),
            _safe_logit(self.cbs[3].predict_proba(Xgbd)[:, 1]),
            _safe_logit(self.boosters[4].predict(Xr2l)),
        ]).astype(np.float32)

        s_cos, p_cos = self._col(Xc, "kern_hi_surf60k_cos"), self._col(Xc, "kern_pnoise_cos")
        ood_surf_aff = s_cos - imp_s_t10
        ood_pn_aff = p_cos - imp_p_t10
        surf_infl = s_cos / (np.maximum(ood_surf_aff, 0.0) + 0.05)
        lg_gb, lg_ct = L10[:, 8], L10[:, 9]
        lg_surf = 0.5 * (L10[:, 0] + L10[:, 3])
        disagree_gb_surf = lg_gb - lg_surf
        disagree_ct_surf = lg_ct - lg_surf
        tree_std = np.std(L10[:, [0, 1, 2, 3, 4, 5, 6, 8, 9]], axis=1)
        tree_min = np.min(L10[:, [3, 5, 8, 9]], axis=1)

        Mg = np.column_stack([
            L10, disagree_gb_surf, disagree_ct_surf, tree_std, tree_min,
            ood_surf_aff, ood_pn_aff, surf_infl,
            self._col(Xc, "s1_gap_pnoise_minus_surf_svm"), self._col(Xc, "imp_gap_pnoise_minus_surf_asnorm"),
            self._col(Xc, "pairmean_bio_coleman_liau_idx"), self._col(Xc, "pairmean_bio_polysyllabic_ratio"),
            self._col(Xc, "pairmean_g2v_morph_nominal"), self._col(Xc, "pairmean_g2v_passive_rate"),
            self._col(Xc, "pairmean_g2v_contraction_rate"), self._col(Xc, "pairmean_bio_first_person_sing_ratio"),
            self._col(Xc, "pairmean_bio_uk_vs_us_orthography"), self._col(Xc, "pairmean_surf_digit_ratio"),
            self._col(Xc, "pairmean_surf_allcaps_ratio"), self._col(Xc, "pair_log_total_words"),
            surf_infl * lg_surf, self._col(Xc, "pairmean_bio_coleman_liau_idx") * disagree_gb_surf,
        ]).astype(np.float32)

        prob = float(np.mean([
            w_lr * glr.predict_proba(np.clip((np.nan_to_num(Mg, nan=0.0, posinf=0.0, neginf=0.0) - gm) / gs, -8.0, 8.0))[:, 1] + w_cb * gcb.predict_proba(Mg)[:, 1]
            for gm, gs, glr, gcb, w_lr, w_cb in self.gates
        ]))
        prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))

        signals = {
            "words_a": len(text_1.split()),
            "words_b": len(text_2.split()),
            "pos_noise_svm_prob": round(float(p_pn[0]), 4),
            "surface_svm_prob": round(float(p_surf[0]), 4),
            "genre_blind_prob": round(float(expit(lg_gb[0])), 4),
            "cohort_mutual_rank": round(float(imp_s_rk[0]), 4),
            "cohort_as_norm": round(float(imp_s_as[0]), 4),
            "compression_sim": round(float(1.0 - cbc_raw[0]), 4),
            "burrows_delta": round(float(self._col(Xc, "fw_burrows_delta")[0]), 4),
        }
        return prob, signals

    def predict_pair(self, text_1: str, text_2: str, check_symmetry: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        prob, signals = self._score_single(text_1, text_2)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        sym_err = 0.0
        if check_symmetry:
            prob_rev, _ = self._score_single(text_2, text_1)
            sym_err = abs(prob - prob_rev)

        if abs(prob - self.tau) < self.delta:
            verdict = "uncertain"
        elif prob >= self.tau:
            verdict = "same_author"
        else:
            verdict = "different_author"

        return {
            "model_id": "ml",
            "model_name": "01. Stylometric ML (10-Expert + OOD Gate)",
            "probability": round(prob, 6),
            "verdict": verdict,
            "forced_verdict": "same_author" if prob >= self.tau else "different_author",
            "tau": round(self.tau, 4),
            "delta": round(self.delta, 4),
            "latency_ms": round(latency_ms, 2),
            "symmetry_error": float(f"{sym_err:.2e}"),
            "device": "CPU",
            "signals": signals,
        }
