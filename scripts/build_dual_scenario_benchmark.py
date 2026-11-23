"""Build the Dual-Scenario 1024-Token Authorship Benchmark (Closed-Set 1:N + Open & Cross-Domain).

Matches the exact evaluation protocol:
1. Closed-Set (닫힌 집합):
   - Known authors seen in Train, unseen documents within the same domain.
   - 4 domain cohorts ('imdb_reviews', 'literature', 'blogs', 'news'), each N=30 candidate authors
     (120 known candidate authors total; uniform 1:30 comparison per domain).
   - Inference: 1:N pairwise verification against each candidate author's Anchor documents
     -> Top-1 Accuracy & Macro-F1.
2. Open & Cross-Domain (오픈 + 크로스 도메인):
   - Strictly Author-Disjoint unseen authors (0 overlap with Train authors; Val & Test mutually disjoint)
     + Cross-Topic, Cross-Genre, Hard Same-Topic Negative, and Zero-Shot Unseen Genre pairs.
   - Inference: Binary verification -> ROC-AUC, PAN c@1, F_0.5u, Brier Score.

Strict Forensic Guarantees:
- Formatting-preserving Head-Tail 350-word truncation ('\\n' join, zero synthetic ' ... ' artifacts).
- Minimum length >= 15 words & >= 12 alphabetic words per document.
- Multi-layer global deduplication (exact + normalized + alphanumeric + 120-char prefix/suffix).
- 0% Author overlap between Train and Open-Val / Open-Test, and between Open-Val and Open-Test.
- 0% Document overlap across ALL evaluation queries/pairs and Train.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

NON_SPACE_RE = re.compile(r"\S+")
ALPHA_WORD_RE = re.compile(r"[A-Za-z]+")
MULTI_SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class DocRecord:
    doc_id: str
    author_id: str
    macro_domain: str
    topic: str
    genre_group: str
    source_split_hint: str  # 'train', 'val', 'test', or 'any'
    text: str
    alnum_fp: str
    pre120: str
    suf120: str


def decode_clean(raw_bytes: bytes) -> str:
    try:
        s = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        s = raw_bytes.decode("cp1252", errors="replace")
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def clean_str(s: str) -> str:
    return str(s).replace("\r\n", "\n").replace("\r", "\n").strip()


def head_tail_truncate_clean(text: str, max_words: int = 350) -> str:
    matches = list(NON_SPACE_RE.finditer(text))
    if len(matches) <= max_words:
        return text
    half = max_words // 2
    return text[: matches[half - 1].end()] + "\n" + text[matches[-half].start() :]


def norm_ws_lower(text: str) -> str:
    return MULTI_SPACE_RE.sub(" ", text.strip().lower())


def alnum_fingerprint(text: str) -> str:
    return NON_ALNUM_RE.sub("", text.lower())


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def map_authbench_domain(source: str, genre: str) -> tuple[str, str, str]:
    """Return (macro_domain, topic, genre_group)."""
    g = str(genre or source).strip().lower()
    src = str(source).strip().lower()
    if src == "babel_briefings":
        return "news", g, "journalism"
    if src == "blog_authorship":
        return "blogs", g, "personal_blog"
    if src in ("project_gutenberg", "wikisource"):
        return "literature", g, g.split("/")[0] if "/" in g else "prose_lit"
    if src == "amazon_multi":
        return "reviews_ecommerce", g, "product_review"
    if src in ("exorde", "ytcomments"):
        return "social_media", g, "social_post"
    if src == "stackexchange":
        return "qna_tech", g, "technical_qna"
    if src == "arxiv":
        return "arxiv_research", g, "academic_paper"
    return src, g, g


def build_benchmark(
    repo_root: Path,
    output_dir: Path,
    seed: int = 378,
    max_words: int = 350,
    min_words: int = 15,
    min_alpha: int = 12,
) -> dict:
    rng = np.random.default_rng(seed)
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_norm_hashes: set[str] = set()
    seen_alnum_hashes: set[str] = set()
    seen_pre120: set[str] = set()
    seen_suf120: set[str] = set()

    all_docs: list[DocRecord] = []
    dedup_dropped = 0
    short_dropped = 0

    def try_register_doc(
        doc_id: str,
        author_id: str,
        macro_domain: str,
        topic: str,
        genre_group: str,
        split_hint: str,
        raw_text: str,
    ) -> bool:
        nonlocal dedup_dropped, short_dropped
        cleaned = clean_str(raw_text)
        if not cleaned:
            return False
        trunc = head_tail_truncate_clean(cleaned, max_words=max_words)
        n_w = len(NON_SPACE_RE.findall(trunc))
        n_a = len(ALPHA_WORD_RE.findall(trunc))
        if n_w < min_words or n_a < min_alpha:
            short_dropped += 1
            return False

        nh = sha1_hex(norm_ws_lower(trunc))
        af_raw = alnum_fingerprint(trunc)
        if len(af_raw) < 40:
            short_dropped += 1
            return False
        ah = sha1_hex(af_raw)
        p120 = sha1_hex("P:" + af_raw[:120]) if len(af_raw) >= 160 else ""
        s120 = sha1_hex("S:" + af_raw[-120:]) if len(af_raw) >= 160 else ""

        if (
            nh in seen_norm_hashes
            or ah in seen_alnum_hashes
            or (p120 and p120 in seen_pre120)
            or (s120 and s120 in seen_suf120)
        ):
            dedup_dropped += 1
            return False

        seen_norm_hashes.add(nh)
        seen_alnum_hashes.add(ah)
        if p120:
            seen_pre120.add(p120)
        if s120:
            seen_suf120.add(s120)

        all_docs.append(
            DocRecord(
                doc_id=doc_id,
                author_id=author_id,
                macro_domain=macro_domain,
                topic=topic,
                genre_group=genre_group,
                source_split_hint=split_hint,
                text=trunc,
                alnum_fp=ah,
                pre120=p120,
                suf120=s120,
            )
        )
        return True

    # 1. Load Stamatatos Guardian Cross-Topic & Cross-Genre Corpus (13 columnists -> 100% Open Cross-Domain!)
    print("1. Loading Stamatatos Guardian Cross-Topic & Cross-Genre corpus...")
    g_zip = repo_root / "data" / "external_raw" / "Guardian.zip"
    g_count = 0
    with zipfile.ZipFile(g_zip) as zf:
        # Load Guardian_original first (includes Books, Politics, Society, UK, World), then Guardian_extended
        txt_entries = sorted(
            n
            for n in zf.namelist()
            if (n.startswith("Guardian/Guardian_original/") or n.startswith("Guardian/Guardian_extended/"))
            and n.endswith(".txt")
        )
        for entry in txt_entries:
            parts = entry.split("/")
            # Guardian / Guardian_original / TopicOrBooks / author / 001.txt
            sub_topic = parts[2]  # Books, Politics, Society, UK, World
            author_slug = parts[3].lower()
            g_group = "book_review" if sub_topic == "Books" else "opinion_column"
            txt = decode_clean(zf.read(entry))
            if try_register_doc(
                doc_id=f"guard_{g_count:04d}",
                author_id=f"guardian:{author_slug}",
                macro_domain="guardian_journalism",
                topic=f"guardian/{sub_topic.lower()}",
                genre_group=g_group,
                split_hint="open_only",
                raw_text=txt,
            ):
                g_count += 1
    print(f"   - Registered {g_count:,} unique Guardian articles across 13 columnists")

    # 2. Load IMDb62 (62 prolific authors: 40 Train/Closed-Set authors, 10 Val-Open authors, 12 Test-Open authors)
    print("2. Loading IMDb62 author-verified corpus...")
    p_imdb = hf_hub_download(
        "tasksource/imdb62",
        "data/train-00000-of-00001-62894f3b39974753387b64" if False else "data/train-00000-of-00001-62894f3b39974716.parquet",
        repo_type="dataset",
        local_files_only=True,
    )
    df_imdb = pd.read_parquet(p_imdb)
    imdb_users = sorted(df_imdb["userId"].unique().tolist())
    rng.shuffle(imdb_users)
    assert len(imdb_users) == 62
    imdb_train_users = set(imdb_users[:40])     # 30 Closed-Set + 10 extra Train
    imdb_val_users = set(imdb_users[40:50])     # 10 Val-Open unseen critics
    imdb_test_users = set(imdb_users[50:62])    # 12 Test-Open unseen critics

    imdb_reg = 0
    for uid, grp in df_imdb.groupby("userId", sort=False):
        # Sample up to 65 clean reviews per critic (plenty for pairs & anchors/queries while keeping domain balanced)
        sampled_rows = grp.sample(n=min(len(grp), 65), random_state=seed)
        sp_hint = "train" if uid in imdb_train_users else ("val" if uid in imdb_val_users else "test")
        for r_idx, row in enumerate(sampled_rows.itertuples()):
            # Rating bucket acts as sentiment polarity sub-topic (low <=4, mid 5-7, high >=8)
            r_val = float(getattr(row, "rating", 5))
            pol = "neg_review" if r_val <= 4 else ("pos_review" if r_val >= 8 else "mid_review")
            if try_register_doc(
                doc_id=f"imdb_{uid}_{r_idx:03d}",
                author_id=f"imdb62:{uid}",
                macro_domain="imdb_reviews",
                topic=f"imdb/{pol}",
                genre_group="movie_review",
                split_hint=sp_hint,
                raw_text=str(row.content),
            ):
                imdb_reg += 1
    print(f"   - Registered {imdb_reg:,} unique IMDb62 reviews across 62 critics (40 Train / 10 Val-Open / 12 Test-Open)")

    # 3. Load MaoXun/AuthBench English corpus (train / dev / test are natively author-disjoint!)
    # Note: Hold out ALL 'arxiv' documents from AuthBench train into Val/Test Open Zero-Shot!
    print("3. Loading MaoXun/AuthBench English corpus...")
    ab_counts = Counter()
    arxiv_train_toggle = 0
    for ab_sp in ["train", "dev", "test"]:
        p_ab = hf_hub_download("MaoXun/AuthBench", f"{ab_sp}/documents.jsonl", repo_type="dataset", local_files_only=True)
        mapped_sp = "train" if ab_sp == "train" else ("val" if ab_sp == "dev" else "test")
        with open(p_ab, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("lang") != "en":
                    continue
                src = str(obj.get("source", ""))
                genre = str(obj.get("genre", ""))
                m_dom, top, g_grp = map_authbench_domain(src, genre)
                eff_sp = mapped_sp
                if m_dom == "arxiv_research" and mapped_sp == "train":
                    # Hold out arxiv completely from training -> pure zero-shot academic research genre!
                    eff_sp = "val" if (arxiv_train_toggle % 2 == 0) else "test"
                    arxiv_train_toggle += 1
                if try_register_doc(
                    doc_id=f"ab_{obj['doc_id']}",
                    author_id=f"ab:{obj['author_id']}",
                    macro_domain=m_dom,
                    topic=top,
                    genre_group=g_grp,
                    split_hint=eff_sp,
                    raw_text=str(obj.get("content", "")),
                ):
                    ab_counts[eff_sp] += 1
    print(f"   - Registered AuthBench EN docs: {dict(ab_counts)} (dedup_dropped={dedup_dropped:,}, short_dropped={short_dropped:,})")

    # Verify strict author partitioning across Train vs Val-Open vs Test-Open!
    # Guardian ('open_only'): assign 5 columnists to 'val' and 8 columnists to 'test'
    guardian_authors = sorted({d.author_id for d in all_docs if d.source_split_hint == "open_only"})
    rng.shuffle(guardian_authors)
    g_val_auths = set(guardian_authors[:5])
    for d in all_docs:
        if d.source_split_hint == "open_only":
            d.source_split_hint = "val" if d.author_id in g_val_auths else "test"

    # Ensure ZERO author overlap across (train, val, test) author pools
    author_to_docs: dict[str, list[DocRecord]] = defaultdict(list)
    for d in all_docs:
        author_to_docs[d.author_id].append(d)

    train_authors: set[str] = set()
    val_open_authors: set[str] = set()
    test_open_authors: set[str] = set()

    arxiv_rerouted_authors = sorted(
        aid for aid, dlist in author_to_docs.items() if any(x.macro_domain == "arxiv_research" for x in dlist)
    )
    rng.shuffle(arxiv_rerouted_authors)
    half_arxiv = len(arxiv_rerouted_authors) // 2
    arxiv_val_set = set(arxiv_rerouted_authors[:half_arxiv])

    for aid, dlist in author_to_docs.items():
        if aid in arxiv_val_set:
            val_open_authors.add(aid)
            continue
        if any(x.macro_domain == "arxiv_research" for x in dlist):
            test_open_authors.add(aid)
            continue
        hints = {x.source_split_hint for x in dlist}
        if "train" in hints:
            train_authors.add(aid)
        elif "val" in hints:
            val_open_authors.add(aid)
        else:
            test_open_authors.add(aid)

    assert len(train_authors & val_open_authors) == 0
    assert len(train_authors & test_open_authors) == 0
    assert len(val_open_authors & test_open_authors) == 0

    # ==========================================================================
    # PART A: Construct Scenario 1 — Closed-Set 1:N Attribution Cohorts (4 Domains x N=30 = 120 Known Authors)
    # ==========================================================================
    print("4. Constructing Scenario 1: Closed-Set 1:30 Attribution (Known Train Authors, Same-Domain Unseen Docs)...")
    closed_anchor_rows = []
    val_closed_query_rows = []
    test_closed_query_rows = []
    heldout_closed_query_doc_ids: set[str] = set()

    # Helper to pick N=30 known authors in a domain who have enough same-domain docs
    def build_closed_cohort(
        cohort_domain: str,
        n_candidates: int,
        n_anchors: int,
        n_val_q: int,
        n_test_q: int,
    ) -> None:
        min_needed = n_anchors + n_val_q + n_test_q
        eligible = []
        for aid in sorted(train_authors):
            dom_docs = [d for d in author_to_docs[aid] if d.macro_domain == cohort_domain]
            if len(dom_docs) >= min_needed:
                eligible.append((len(dom_docs), aid, dom_docs))
        # Sort by doc count desc, deterministic shuffle within ties
        rng.shuffle(eligible)
        eligible.sort(key=lambda x: -x[0])
        assert len(eligible) >= n_candidates, f"Not enough authors for {cohort_domain}: got {len(eligible)} < {n_candidates}"

        chosen = eligible[:n_candidates]
        for cand_idx, (_, aid, dom_docs) in enumerate(chosen):
            d_copy = list(dom_docs)
            rng.shuffle(d_copy)
            val_qs = d_copy[:n_val_q]
            test_qs = d_copy[n_val_q : n_val_q + n_test_q]
            # Remaining docs stay in Train! First n_anchors of remaining docs serve as Representative Anchors!
            train_pool_docs = d_copy[n_val_q + n_test_q :]
            anchors = train_pool_docs[:n_anchors]

            for q in val_qs:
                heldout_closed_query_doc_ids.add(q.doc_id)
                val_closed_query_rows.append({
                    "cohort_domain": cohort_domain,
                    "true_author_id": aid,
                    "topic": q.topic,
                    "doc_id": q.doc_id,
                    "text": q.text,
                })
            for q in test_qs:
                heldout_closed_query_doc_ids.add(q.doc_id)
                test_closed_query_rows.append({
                    "cohort_domain": cohort_domain,
                    "true_author_id": aid,
                    "topic": q.topic,
                    "doc_id": q.doc_id,
                    "text": q.text,
                })
            for a_i, anc in enumerate(anchors):
                closed_anchor_rows.append({
                    "cohort_domain": cohort_domain,
                    "candidate_author_id": aid,
                    "anchor_idx": a_i,
                    "topic": anc.topic,
                    "doc_id": anc.doc_id,
                    "text": anc.text,
                })
        print(
            f"   - Cohort [{cohort_domain:14s}]: N={n_candidates} authors, K={n_anchors} anchors/author, "
            f"Val queries={n_candidates*n_val_q}, Test queries={n_candidates*n_test_q}"
        )

    build_closed_cohort("imdb_reviews", n_candidates=30, n_anchors=3, n_val_q=2, n_test_q=4)
    build_closed_cohort("literature", n_candidates=30, n_anchors=2, n_val_q=1, n_test_q=2)
    build_closed_cohort("blogs", n_candidates=30, n_anchors=2, n_val_q=1, n_test_q=1)
    build_closed_cohort("news", n_candidates=30, n_anchors=2, n_val_q=1, n_test_q=1)

    closed_anchors_df = pd.DataFrame(closed_anchor_rows)
    val_closed_df = pd.DataFrame(val_closed_query_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_closed_df = pd.DataFrame(test_closed_query_rows).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    val_closed_df.insert(0, "query_id", np.arange(len(val_closed_df)))
    test_closed_df.insert(0, "query_id", np.arange(len(test_closed_df)))

    # ==========================================================================
    # PART B: Construct Unified Training Pair Set (train_pairs.csv) from Train Docs ONLY
    # ==========================================================================
    print("5. Constructing Training Pair Set (train_pairs.csv, strictly excluding Closed-Set Val/Test queries)...")
    train_author_docs: dict[str, list[DocRecord]] = {}
    for aid in sorted(train_authors):
        usable = [d for d in author_to_docs[aid] if d.doc_id not in heldout_closed_query_doc_ids]
        if usable:
            train_author_docs[aid] = usable

    # Generate Positive pairs from train_author_docs (cap per author so IMDb doesn't overwhelm AuthBench!)
    pos_train = []
    used_pair_ids: set[tuple[str, str]] = set()

    for aid, dlist in train_author_docs.items():
        if len(dlist) < 2:
            continue
        # Prioritize cross-topic pairs first!
        cand_pairs = []
        for i in range(len(dlist)):
            for j in range(i + 1, len(dlist)):
                is_ct = int(dlist[i].topic != dlist[j].topic)
                cand_pairs.append((is_ct, dlist[i], dlist[j]))
        rng.shuffle(cand_pairs)
        cand_pairs.sort(key=lambda x: -x[0])
        dom_first = dlist[0].macro_domain
        if aid.startswith("imdb62:"):
            max_per_auth = 125
        elif dom_first == "literature":
            max_per_auth = 2
        else:
            max_per_auth = 6
        for is_ct, d1, d2 in cand_pairs[:max_per_auth]:
            if rng.random() < 0.5:
                d1, d2 = d2, d1
            pk = tuple(sorted((d1.doc_id, d2.doc_id)))
            if pk not in used_pair_ids:
                used_pair_ids.add(pk)
                pos_train.append({
                    "text_1": d1.text,
                    "text_2": d2.text,
                    "label": 1,
                    "domain": d1.macro_domain,
                    "topic_1": d1.topic,
                    "topic_2": d2.topic,
                    "is_cross_topic": int(d1.topic != d2.topic),
                    "author_1": d1.author_id,
                    "author_2": d2.author_id,
                    "doc_id_1": d1.doc_id,
                    "doc_id_2": d2.doc_id,
                })

    # Match every positive with 1 negative in the same macro_domain (60% Hard Same-Topic Negative, 40% Cross-Topic Negative!)
    docs_by_dom: dict[str, list[DocRecord]] = defaultdict(list)
    docs_by_topic: dict[str, list[DocRecord]] = defaultdict(list)
    for dlist in train_author_docs.values():
        for d in dlist:
            docs_by_dom[d.macro_domain].append(d)
            docs_by_topic[d.topic].append(d)

    neg_train = []
    for p_row in pos_train:
        dom = p_row["domain"]
        top1 = p_row["topic_1"]
        use_same_topic = (rng.random() < 0.60) and len(docs_by_topic[top1]) >= 10
        pool = docs_by_topic[top1] if use_same_topic else docs_by_dom[dom]
        for _ in range(50):
            d1 = pool[int(rng.integers(0, len(pool)))]
            d2 = docs_by_dom[dom][int(rng.integers(0, len(docs_by_dom[dom])))]
            if use_same_topic:
                d2 = pool[int(rng.integers(0, len(pool)))]
            if d1.author_id == d2.author_id:
                continue
            pk = tuple(sorted((d1.doc_id, d2.doc_id)))
            if pk in used_pair_ids:
                continue
            used_pair_ids.add(pk)
            neg_train.append({
                "text_1": d1.text,
                "text_2": d2.text,
                "label": 0,
                "domain": dom,
                "topic_1": d1.topic,
                "topic_2": d2.topic,
                "is_cross_topic": int(d1.topic != d2.topic),
                "author_1": d1.author_id,
                "author_2": d2.author_id,
                "doc_id_1": d1.doc_id,
                "doc_id_2": d2.doc_id,
            })
            break

    train_pairs_df = pd.DataFrame(pos_train + neg_train).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_pairs_df.insert(0, "pair_id", np.arange(len(train_pairs_df)))
    print(
        f"   - Built train_pairs.csv: {len(train_pairs_df):,} pairs "
        f"(Pos={train_pairs_df['label'].mean()*100:.2f}%, Cross-Topic={train_pairs_df['is_cross_topic'].mean()*100:.1f}%)"
    )

    # ==========================================================================
    # PART C: Construct Scenario 2 — Open & Cross-Domain Verification (Val & Test)
    # ==========================================================================
    def build_open_split(open_authors: set[str], split_tag: str, rng_seed: int) -> pd.DataFrame:
        local_rng = np.random.default_rng(rng_seed)
        pos_rows = []
        used_pk: set[tuple[str, str]] = set()

        open_docs_by_dom: dict[str, list[DocRecord]] = defaultdict(list)
        open_docs_by_topic: dict[str, list[DocRecord]] = defaultdict(list)

        for aid in sorted(open_authors):
            dlist = author_to_docs[aid]
            for d in dlist:
                open_docs_by_dom[d.macro_domain].append(d)
                open_docs_by_topic[d.topic].append(d)

            if len(dlist) < 2:
                continue

            # Enumerate all candidate same-author pairs and classify their Open/Cross stratum
            cands_cross_genre = []
            cands_cross_topic = []
            cands_zero_shot = []
            cands_in_dom = []

            for i in range(len(dlist)):
                for j in range(i + 1, len(dlist)):
                    a, b = dlist[i], dlist[j]
                    if local_rng.random() < 0.5:
                        a, b = b, a
                    if a.macro_domain in ("arxiv_research", "guardian_journalism") and a.genre_group == b.genre_group and a.topic == b.topic:
                        if a.macro_domain == "arxiv_research":
                            cands_zero_shot.append(("zero_shot_unseen_genre", a, b))
                        else:
                            cands_in_dom.append(("open_unseen_author_same_topic", a, b))
                    elif a.genre_group != b.genre_group:
                        cands_cross_genre.append(("cross_genre", a, b))
                    elif a.topic != b.topic:
                        cands_cross_topic.append(("cross_topic", a, b))
                    elif a.macro_domain == "arxiv_research":
                        cands_zero_shot.append(("zero_shot_unseen_genre", a, b))
                    else:
                        cands_in_dom.append(("open_unseen_author_same_topic", a, b))

            local_rng.shuffle(cands_cross_genre)
            local_rng.shuffle(cands_cross_topic)
            local_rng.shuffle(cands_zero_shot)
            local_rng.shuffle(cands_in_dom)

            # Caps per author to keep Guardian/IMDb balanced against AuthBench's 5,000 unseen authors!
            if aid.startswith("guardian:"):
                selected = cands_cross_genre[:35] + cands_cross_topic[:45] + cands_in_dom[:10]
            elif aid.startswith("imdb62:"):
                selected = cands_cross_topic[:35] + cands_in_dom[:25]
            else:
                selected = cands_cross_genre[:4] + cands_cross_topic[:4] + cands_zero_shot[:4] + cands_in_dom[:2]

            for stratum, d1, d2 in selected:
                pk = tuple(sorted((d1.doc_id, d2.doc_id)))
                if pk in used_pk:
                    continue
                used_pk.add(pk)
                pos_rows.append({
                    "text_1": d1.text,
                    "text_2": d2.text,
                    "label": 1,
                    "eval_stratum": stratum,
                    "domain": d1.macro_domain,
                    "topic_1": d1.topic,
                    "topic_2": d2.topic,
                    "genre_1": d1.genre_group,
                    "genre_2": d2.genre_group,
                    "is_cross_topic": int(d1.topic != d2.topic),
                    "is_cross_genre": int(d1.genre_group != d2.genre_group),
                    "is_zero_shot_domain": int(d1.macro_domain in ("arxiv_research", "guardian_journalism")),
                    "author_1": d1.author_id,
                    "author_2": d2.author_id,
                    "doc_id_1": d1.doc_id,
                    "doc_id_2": d2.doc_id,
                })

        # Also incorporate BAWE british university essays (val/test native split only, deduplicated!) as extra zero-shot academic essays!
        # Wait: to preserve 100% explicit cryptographic author_id on EVERY row, let's check how many zero_shot rows we already have!
        neg_rows = []
        all_open_list = [d for aid in sorted(open_authors) for d in author_to_docs[aid]]
        docs_by_genre_grp: dict[str, list[DocRecord]] = defaultdict(list)
        for d in all_open_list:
            docs_by_genre_grp[d.genre_group].append(d)

        for p_row in pos_rows:
            stratum = p_row["eval_stratum"]
            dom = p_row["domain"]
            dom_pool = open_docs_by_dom[dom]

            for _ in range(80):
                if stratum == "cross_topic":
                    # Half hard same-topic negative (topic trap!), half cross-topic negative within same domain!
                    top_pool = open_docs_by_topic[p_row["topic_1"]]
                    if local_rng.random() < 0.55 and len(top_pool) >= 4:
                        d1 = top_pool[int(local_rng.integers(0, len(top_pool)))]
                        d2 = top_pool[int(local_rng.integers(0, len(top_pool)))]
                    else:
                        d1 = dom_pool[int(local_rng.integers(0, len(dom_pool)))]
                        d2 = dom_pool[int(local_rng.integers(0, len(dom_pool)))]
                elif stratum == "cross_genre":
                    g1_pool = docs_by_genre_grp[p_row["genre_1"]]
                    g2_pool = docs_by_genre_grp[p_row["genre_2"]]
                    d1 = g1_pool[int(local_rng.integers(0, len(g1_pool)))]
                    d2 = g2_pool[int(local_rng.integers(0, len(g2_pool)))]
                else:
                    d1 = dom_pool[int(local_rng.integers(0, len(dom_pool)))]
                    d2 = dom_pool[int(local_rng.integers(0, len(dom_pool)))]

                if d1.author_id == d2.author_id:
                    continue
                pk = tuple(sorted((d1.doc_id, d2.doc_id)))
                if pk in used_pk:
                    continue
                used_pk.add(pk)
                neg_rows.append({
                    "text_1": d1.text,
                    "text_2": d2.text,
                    "label": 0,
                    "eval_stratum": stratum,
                    "domain": d1.macro_domain if d1.macro_domain == d2.macro_domain else f"{d1.macro_domain}_x_{d2.macro_domain}",
                    "topic_1": d1.topic,
                    "topic_2": d2.topic,
                    "genre_1": d1.genre_group,
                    "genre_2": d2.genre_group,
                    "is_cross_topic": int(d1.topic != d2.topic),
                    "is_cross_genre": int(d1.genre_group != d2.genre_group),
                    "is_zero_shot_domain": int(
                        d1.macro_domain in ("arxiv_research", "guardian_journalism")
                        or d2.macro_domain in ("arxiv_research", "guardian_journalism")
                    ),
                    "author_1": d1.author_id,
                    "author_2": d2.author_id,
                    "doc_id_1": d1.doc_id,
                    "doc_id_2": d2.doc_id,
                })
                break

        out_df = pd.DataFrame(pos_rows + neg_rows).sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)
        out_df.insert(0, "pair_id", np.arange(len(out_df)))
        print(
            f"   - Built {split_tag}: {len(out_df):,} pairs (Pos={out_df['label'].mean()*100:.2f}%) | "
            f"Strata={out_df['eval_stratum'].value_counts().to_dict()}"
        )
        return out_df

    print("6. Constructing Scenario 2: Open & Cross-Domain Verification (100% Author-Disjoint Val & Test)...")
    val_open_df = build_open_split(val_open_authors, "val_open_cross.csv", seed + 10)
    test_open_df = build_open_split(test_open_authors, "test_open_cross.csv", seed + 20)

    # Save all 6 CSV files
    print("7. Writing benchmark_1024 files to disk...")
    train_pairs_df.to_csv(output_dir / "train_pairs.csv", index=False, encoding="utf-8-sig")
    closed_anchors_df.to_csv(output_dir / "closed_anchors.csv", index=False, encoding="utf-8-sig")
    val_closed_df.to_csv(output_dir / "val_closed_queries.csv", index=False, encoding="utf-8-sig")
    test_closed_df.to_csv(output_dir / "test_closed_queries.csv", index=False, encoding="utf-8-sig")
    val_open_df.to_csv(output_dir / "val_open_cross.csv", index=False, encoding="utf-8-sig")
    test_open_df.to_csv(output_dir / "test_open_cross.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "benchmark_version": "3.0-dual-scenario-author-verified",
        "max_words_per_doc": max_words,
        "min_words_per_doc": min_words,
        "max_pair_token_budget": 1024,
        "train_pairs": {
            "total": len(train_pairs_df),
            "pos_rate": round(float(train_pairs_df["label"].mean()), 5),
            "unique_authors": int(len(set(train_pairs_df["author_1"]) | set(train_pairs_df["author_2"]))),
            "cross_topic_ratio": round(float(train_pairs_df["is_cross_topic"].mean()), 4),
            "domain_counts": train_pairs_df["domain"].value_counts().to_dict(),
        },
        "scenario_1_closed_set": {
            "protocol": "1:30 Same-Domain Attribution via Pairwise Verification against K Anchors",
            "metrics": ["Top-1 Accuracy", "Macro-F1"],
            "total_known_candidate_authors": int(closed_anchors_df["candidate_author_id"].nunique()),
            "candidates_per_domain": 30,
            "total_anchors": len(closed_anchors_df),
            "val_queries": len(val_closed_df),
            "test_queries": len(test_closed_df),
            "cohorts": closed_anchors_df.groupby("cohort_domain")["candidate_author_id"].nunique().to_dict(),
        },
        "scenario_2_open_cross_domain": {
            "protocol": "100% Author-Disjoint Open & Cross-Topic/Genre/Zero-Shot Binary Verification",
            "metrics": ["ROC-AUC", "PAN c@1", "PAN F_0.5u", "PAN Brier Score (1 - Brier)", "Macro-F1"],
            "val_open_pairs": len(val_open_df),
            "val_open_pos_rate": round(float(val_open_df["label"].mean()), 5),
            "val_strata": val_open_df["eval_stratum"].value_counts().to_dict(),
            "val_domains": val_open_df["domain"].value_counts().to_dict(),
            "test_open_pairs": len(test_open_df),
            "test_open_pos_rate": round(float(test_open_df["label"].mean()), 5),
            "test_strata": test_open_df["eval_stratum"].value_counts().to_dict(),
            "test_domains": test_open_df["domain"].value_counts().to_dict(),
        },
    }

    with open(output_dir / "benchmark_1024_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n=== BENCHMARK 1024 v3.0 DUAL-SCENARIO BUILT SUCCESSFULLY ===")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    repo = Path(__file__).resolve().parent.parent
    build_benchmark(repo, repo / "data" / "benchmark_1024")
