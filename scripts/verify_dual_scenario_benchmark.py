"""Independent 16-point forensic verifier for data/benchmark_1024/ Dual-Scenario Benchmark."""
from __future__ import annotations

import json
from pathlib import Path
import re
import pandas as pd

NON_SPACE_RE = re.compile(r"\S+")
MULTI_SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def norm_s(s: str) -> str:
    return MULTI_SPACE_RE.sub(" ", s.strip().lower())


def alnum_s(s: str) -> str:
    return NON_ALNUM_RE.sub("", s.lower())


def doc_fingerprints(texts: list[str]) -> dict[str, set[str]]:
    ex, nm, al, p120, s120 = set(), set(), set(), set(), set()
    for t in texts:
        ex.add(t)
        nm.add(norm_s(t))
        a = alnum_s(t)
        al.add(a)
        if len(a) >= 160:
            p120.add(a[:120])
            s120.add(a[-120:])
    return {"exact": ex, "norm": nm, "alnum": al, "pre120": p120, "suf120": s120}


def verify_all(b_dir: Path) -> dict:
    train_df = pd.read_csv(b_dir / "train_pairs.csv")
    anchors_df = pd.read_csv(b_dir / "closed_anchors.csv")
    val_cl_df = pd.read_csv(b_dir / "val_closed_queries.csv")
    test_cl_df = pd.read_csv(b_dir / "test_closed_queries.csv")
    val_op_df = pd.read_csv(b_dir / "val_open_cross.csv")
    test_op_df = pd.read_csv(b_dir / "test_open_cross.csv")

    report: dict[str, object] = {}

    # 1. Null & Word Bound checks
    nulls = sum(int(d.isna().sum().sum()) for d in [train_df, anchors_df, val_cl_df, test_cl_df, val_op_df, test_op_df])
    under15 = 0
    over350 = 0
    for d, cols in [
        (train_df, ["text_1", "text_2"]),
        (anchors_df, ["text"]),
        (val_cl_df, ["text"]),
        (test_cl_df, ["text"]),
        (val_op_df, ["text_1", "text_2"]),
        (test_op_df, ["text_1", "text_2"]),
    ]:
        for c in cols:
            for t in d[c].astype(str):
                wc = len(NON_SPACE_RE.findall(t))
                if wc < 15:
                    under15 += 1
                if wc > 350:
                    over350 += 1

    # 2. Ground-truth author consistency & duplicate pair checks
    label_mismatches = 0
    self_pairs = 0
    dup_pairs = 0
    seen_pairs = set()
    for name, d in [("train", train_df), ("val_open", val_op_df), ("test_open", test_op_df)]:
        for r in d.itertuples():
            a1, a2, lab = r.author_1, r.author_2, int(r.label)
            if (lab == 1 and a1 != a2) or (lab == 0 and a1 == a2):
                label_mismatches += 1
            f1, f2 = alnum_s(r.text_1), alnum_s(r.text_2)
            if f1 == f2:
                self_pairs += 1
            pk = tuple(sorted((f1, f2)))
            if pk in seen_pairs:
                dup_pairs += 1
            seen_pairs.add(pk)

    # 3. Closed-Set 1:N guarantees
    train_auths = set(train_df["author_1"]) | set(train_df["author_2"])
    anchor_auths = set(anchors_df["candidate_author_id"])
    untrained_anchor_auths = len(anchor_auths - train_auths)

    cohort_mismatch = 0
    for d in (val_cl_df, test_cl_df):
        for r in d.itertuples():
            valid_cands = set(anchors_df[anchors_df["cohort_domain"] == r.cohort_domain]["candidate_author_id"])
            if r.true_author_id not in valid_cands:
                cohort_mismatch += 1

    # 4. Author-Disjointness guarantees for Open & Cross-Domain
    val_op_auths = set(val_op_df["author_1"]) | set(val_op_df["author_2"])
    test_op_auths = set(test_op_df["author_1"]) | set(test_op_df["author_2"])

    author_overlaps = {
        "train_vs_val_open_authors": len(train_auths & val_op_auths),
        "train_vs_test_open_authors": len(train_auths & test_op_auths),
        "val_open_vs_test_open_authors": len(val_op_auths & test_op_auths),
        "closed_cands_vs_val_open_authors": len(anchor_auths & val_op_auths),
        "closed_cands_vs_test_open_authors": len(anchor_auths & test_op_auths),
    }

    # 5. Document-Disjointness guarantees across all 5 fingerprint levels
    fp_train = doc_fingerprints(pd.concat([train_df["text_1"], train_df["text_2"]]).tolist())
    fp_anc = doc_fingerprints(anchors_df["text"].tolist())
    fp_vcl = doc_fingerprints(val_cl_df["text"].tolist())
    fp_tcl = doc_fingerprints(test_cl_df["text"].tolist())
    fp_vop = doc_fingerprints(pd.concat([val_op_df["text_1"], val_op_df["text_2"]]).tolist())
    fp_top = doc_fingerprints(pd.concat([test_op_df["text_1"], test_op_df["text_2"]]).tolist())

    doc_overlaps = {}
    for lvl in ("exact", "norm", "alnum", "pre120", "suf120"):
        doc_overlaps[f"train_vs_val_closed_q_{lvl}"] = len(fp_train[lvl] & fp_vcl[lvl])
        doc_overlaps[f"train_vs_test_closed_q_{lvl}"] = len(fp_train[lvl] & fp_tcl[lvl])
        doc_overlaps[f"anchors_vs_val_closed_q_{lvl}"] = len(fp_anc[lvl] & fp_vcl[lvl])
        doc_overlaps[f"anchors_vs_test_closed_q_{lvl}"] = len(fp_anc[lvl] & fp_tcl[lvl])
        doc_overlaps[f"val_closed_q_vs_test_closed_q_{lvl}"] = len(fp_vcl[lvl] & fp_tcl[lvl])
        known_union = fp_train[lvl] | fp_anc[lvl] | fp_vcl[lvl] | fp_tcl[lvl]
        doc_overlaps[f"known_all_vs_val_open_{lvl}"] = len(known_union & fp_vop[lvl])
        doc_overlaps[f"known_all_vs_test_open_{lvl}"] = len(known_union & fp_top[lvl])
        doc_overlaps[f"val_open_vs_test_open_{lvl}"] = len(fp_vop[lvl] & fp_top[lvl])

    report["null_values"] = nulls
    report["under_15_words"] = under15
    report["over_350_words"] = over350
    report["author_label_mismatches"] = label_mismatches
    report["self_pairs"] = self_pairs
    report["duplicate_pairs_global"] = dup_pairs
    report["untrained_closed_candidate_authors"] = untrained_anchor_auths
    report["closed_query_cohort_mismatches"] = cohort_mismatch
    report["author_overlaps"] = author_overlaps
    report["doc_overlaps_max_across_all_40_checks"] = max(doc_overlaps.values())
    report["doc_overlaps_detail"] = doc_overlaps

    print(json.dumps(report, indent=2, ensure_ascii=False))

    assert nulls == 0
    assert under15 == 0
    assert over350 == 0
    assert label_mismatches == 0
    assert self_pairs == 0
    assert dup_pairs == 0
    assert untrained_anchor_auths == 0
    assert cohort_mismatch == 0
    assert all(v == 0 for v in author_overlaps.values()), f"Author overlap! {author_overlaps}"
    assert all(v == 0 for v in doc_overlaps.values()), f"Doc overlap! {doc_overlaps}"

    print("\n[ALL 16 FORENSIC CHECKS (40 SUB-CHECKS) PASSED: 0 AUTHOR LEAKAGE, 0 DOC LEAKAGE, 0 DUPLICATES!]")
    return report


if __name__ == "__main__":
    verify_all(Path(r"D:\Projects\NLU-Authorship-Verification\data\benchmark_1024"))
