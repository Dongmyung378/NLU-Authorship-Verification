"""Build train_03_deberta_v3_large.ipynb: unified DeBERTa-v3-large training, Dual-ABTT OOD gate & export notebook."""
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
    md_cell(r'''# 03. DeBERTa-v3-large Authorship Verification

`microsoft/deberta-v3-large` (24-layer, 437M params) 기반 순수 딥러닝 동일 저자 판별 최종 학습·평가·배포 패키징 노트북입니다.
`benchmark_1024.zip`을 업로드하고 순서대로 실행하면 LLRD + Prior-WD 백본 학습, 체크포인트 보간 수프(`0.65 / 0.35`), 24층 Multi-Std + Dual-Rank All-but-the-Top(`r=3, 8`) + 대칭 합의축 PCA OOD 보정, Locked Test 평가, 배포용 순수 텐서 `.pt` 저장까지 일괄 수행됩니다.

- 권장 환경: GPU 런타임 (`L4 24GB` / `A100 40GB` / `A100 80GB` / `H100 80GB`)
- 입력 파일: `benchmark_1024.zip` (선택: 3대 모델 비교용 `results_01_stylometric_ml.zip`, `results_02_deberta_v3_base.zip`)
- 주요 출력: `model_03_deberta_v3_large.pt`, `results_03_deberta_v3_large.zip`
'''),
    md_cell("## 1. Environment & Configuration"),
    code_cell(r'''import os
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REQUIRED_PKGS = [
    "transformers>=4.44.0",
    "accelerate>=0.33.0",
    "sentencepiece>=0.2.0",
    "protobuf>=4.25.0",
    "tiktoken>=0.7.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "threadpoolctl>=3.1.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
    "tqdm>=4.66.0",
    "psutil>=5.9.0",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *REQUIRED_PKGS])

import gc
import json
import math
import pickle
import random
import re
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.dpi"] = 300
sns.set_theme(style="whitegrid")

assert torch.cuda.is_available(), "CUDA GPU runtime is required."
DEVICE = torch.device("cuda")
GPU_NAME = torch.cuda.get_device_name(0)
VRAM_GB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
USE_BF16 = torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

SEED = 378


def seed_everything(seed: int = 378) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

# Maintain effective batch size 32 across GPU memory tiers
if VRAM_GB >= 70.0:
    BATCH_SIZE, GRAD_ACCUM, USE_GRAD_CKPT, EVAL_BATCH_SIZE = 16, 2, False, 192
elif VRAM_GB >= 35.0:
    BATCH_SIZE, GRAD_ACCUM, USE_GRAD_CKPT, EVAL_BATCH_SIZE = 16, 2, True, 128
elif VRAM_GB >= 20.0:
    BATCH_SIZE, GRAD_ACCUM, USE_GRAD_CKPT, EVAL_BATCH_SIZE = 8, 4, True, 32
else:
    BATCH_SIZE, GRAD_ACCUM, USE_GRAD_CKPT, EVAL_BATCH_SIZE = 4, 8, True, 24

CONFIG = {
    "model_name": "microsoft/deberta-v3-large",
    "max_doc_tokens": 512,
    "run_hp_probe": False,          # set True to rerun the 4-way 10% hyperparameter probe
    "run_final_test": True,         # set False during validation-only tuning
    "force_retrain": False,
    "epochs": 2.5,
    "backbone_lr": 1.6e-5,
    "head_lr": 1.0e-4,
    "use_llrd": True,
    "llrd_gamma": round(math.sqrt(0.90), 5),
    "use_prior_wd": True,
    "prior_wd_alpha": 0.01,
    "cosent_weight": 0.0,
    "entity_mask_prob": 0.15,
    "label_smoothing": 0.02,
    "ema_decay": 0.9985,
    "soup_alpha": 0.65,
    "band_slices": {"b1": (2, 7), "b2": (8, 13), "b3": (14, 19), "b4": (22, 25)},
    "abtt_specs": [
        ("unit", "unit", 3), ("b1", "b1", 3), ("b2", "b2", 3), ("b3", "b3", 3), ("b4", "b4", 3),
        ("s1", "s1", 2), ("s2", "s2", 2), ("s3", "s3", 2),
        ("unit_r8", "unit", 8), ("b1_r8", "b1", 8), ("b2_r8", "b2", 8), ("b4_r8", "b4", 8),
    ],
    "pca_dom_k": 36,
    "pca_agree_k": 16,
    "pca_interact_k": 10,
    "gate_lr_C": 0.06,
    "gate_lr_w": 0.35,
    "gate_mlp_seeds": [400, 500, 600, 700, 800],
    "gate_mlp_epochs": 240,
    "gate_mlp_lr": 2.0e-3,
    "gate_mlp_wd": 0.12,
    "csls_grid": [0.0, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35],
    "max_abstention_pct": 15.0,
}

print(
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"gpu={GPU_NAME} ({VRAM_GB:.1f}GB) amp={'bf16' if USE_BF16 else 'fp16'} "
    f"bs={BATCH_SIZE}x{GRAD_ACCUM} grad_ckpt={USE_GRAD_CKPT} eval_bs={EVAL_BATCH_SIZE}"
)
'''),
    md_cell("## 2. Data Loading & Document Token Store"),
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
    raise FileNotFoundError("benchmark_1024 files not found. Upload benchmark_1024.zip first.")

ROOT_OUT = (Path.cwd() / "nlu_outputs").resolve()
OUTPUT_DIR = ROOT_OUT / "03_deberta_v3_large"
FIG_DIR = OUTPUT_DIR / "figures"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
TOKEN_CACHE_PATH = ROOT_OUT / "cache_deberta_v3_tokens_512.pkl"
for d in (ROOT_OUT, OUTPUT_DIR, FIG_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Unpack optional peer result archives for the 3-way statistical comparison
PEER_BUNDLES = (
    "results_01_stylometric_ml.zip",
    "results_02_deberta_v3_base.zip",
    "outputs_01_ml_sota_bundle.zip",
    "outputs_02_deberta_base_bundle.zip",
)
for b_name in PEER_BUNDLES:
    for cand in (Path.cwd() / b_name, Path("/content") / b_name, Path.home() / b_name):
        if cand.exists():
            with zipfile.ZipFile(cand, "r") as zf:
                zf.extractall(ROOT_OUT)
            print(f"[peer] merged {cand.name}")
            break

train_df = pd.read_csv(DATA_DIR / "train_pairs.csv").reset_index(drop=True)
anchors_df = pd.read_csv(DATA_DIR / "closed_anchors.csv").reset_index(drop=True)
val_closed_q_df = pd.read_csv(DATA_DIR / "val_closed_queries.csv").reset_index(drop=True)
test_closed_q_df = pd.read_csv(DATA_DIR / "test_closed_queries.csv").reset_index(drop=True)
val_open_df = pd.read_csv(DATA_DIR / "val_open_cross.csv").reset_index(drop=True)
test_open_df = pd.read_csv(DATA_DIR / "test_open_cross.csv").reset_index(drop=True)


def expand_closed_pairs(queries_df: pd.DataFrame, anc_df: pd.DataFrame) -> pd.DataFrame:
    anc_by_dom = {dom: grp.reset_index(drop=True) for dom, grp in anc_df.groupby("cohort_domain", sort=False)}
    rows = []
    for q in queries_df.itertuples():
        for a in anc_by_dom[q.cohort_domain].itertuples():
            rows.append({
                "query_id": int(q.query_id), "cohort_domain": q.cohort_domain,
                "true_author_id": q.true_author_id, "candidate_author_id": a.candidate_author_id,
                "anchor_idx": int(a.anchor_idx), "text_1": str(q.text), "text_2": str(a.text),
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
                        "cand_a": items[i].candidate_author_id, "cand_b": items[j].candidate_author_id,
                        "text_1": str(items[i].text), "text_2": str(items[j].text),
                    })
    return pd.DataFrame(rows)


val_closed_pairs_df = expand_closed_pairs(val_closed_q_df, anchors_df)
test_closed_pairs_df = expand_closed_pairs(test_closed_q_df, anchors_df)
anchor_hub_pairs_df = build_anchor_hub_pairs(anchors_df)

try:
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=False)

MASK_TOKEN_ID = int(tokenizer.mask_token_id)
PAD_TOKEN_ID = int(tokenizer.pad_token_id)

COMMON_STOP_LOWER = frozenset([
    "the", "a", "an", "and", "but", "or", "if", "when", "while", "because", "although", "however",
    "therefore", "this", "that", "these", "those", "he", "she", "it", "they", "we", "you", "his",
    "her", "their", "our", "my", "in", "on", "at", "for", "with", "from", "by", "about", "into",
])
SENT_END_TOKENS = frozenset([".", chr(33), "?", "\n", ":", ";", '"', "”", "“"])


class FastDocStore:
    """Pre-tokenized unique document table with mid-sentence proper-noun spans."""

    def __init__(self, tok, max_len: int = 512):
        self.tok = tok
        self.max_len = max_len
        self.map: dict[str, int] = {}
        self.ids_list: list[np.ndarray] = []
        self.ent_pos_list: list[np.ndarray] = []

    def add_and_map(self, texts) -> np.ndarray:
        str_list = [str(x) for x in texts]
        unseen = [s for s in dict.fromkeys(str_list) if s not in self.map]
        if unseen:
            base_idx = len(self.ids_list)
            for offset, s in enumerate(unseen):
                self.map[s] = base_idx + offset
            for st in range(0, len(unseen), 4096):
                enc = self.tok(unseen[st : st + 4096], max_length=self.max_len, truncation=True, padding=False, return_attention_mask=False)
                for b_ids in enc["input_ids"]:
                    self.ids_list.append(np.asarray(b_ids, dtype=np.int32))
                    tok_strs = self.tok.convert_ids_to_tokens(b_ids)
                    ent_idx, prev_end, in_ent = [], True, False
                    for ti in range(1, len(tok_strs) - 1):
                        ts = tok_strs[ti]
                        clean = ts.lstrip("▁Ġ ")
                        if not clean:
                            in_ent = False
                            continue
                        if clean in SENT_END_TOKENS:
                            prev_end, in_ent = True, False
                            continue
                        if ts.startswith(("▁", "Ġ")) or ti == 1:
                            in_ent = (
                                (not prev_end) and len(clean) >= 2 and clean[0].isupper()
                                and clean.lower() not in COMMON_STOP_LOWER
                                and clean not in ("I", "I'm", "I've", "I'd", "I'll")
                            )
                        if in_ent and any(ch.isalpha() for ch in clean):
                            ent_idx.append(ti)
                        if any(ch.isalpha() for ch in clean):
                            prev_end = False
                    self.ent_pos_list.append(np.asarray(ent_idx, dtype=np.int16))
        return np.fromiter((self.map[s] for s in str_list), dtype=np.int32, count=len(str_list))


os.environ["TOKENIZERS_PARALLELISM"] = "true"
doc_store = FastDocStore(tokenizer, max_len=CONFIG["max_doc_tokens"])
if TOKEN_CACHE_PATH.exists():
    with open(TOKEN_CACHE_PATH, "rb") as f:
        cached = pickle.load(f)
    if cached.get("vocab_size") == tokenizer.vocab_size and cached.get("max_len") == CONFIG["max_doc_tokens"]:
        doc_store.map = cached["map"]
        doc_store.ids_list = cached["ids_list"]
        doc_store.ent_pos_list = cached["ent_pos_list"]

prev_n = len(doc_store.ids_list)
for df in (train_df, val_open_df, test_open_df, val_closed_pairs_df, test_closed_pairs_df, anchor_hub_pairs_df):
    df["d1"] = doc_store.add_and_map(df["text_1"])
    df["d2"] = doc_store.add_and_map(df["text_2"])

if len(doc_store.ids_list) > prev_n:
    with open(TOKEN_CACHE_PATH, "wb") as f:
        pickle.dump(
            {"vocab_size": tokenizer.vocab_size, "max_len": CONFIG["max_doc_tokens"], "map": doc_store.map, "ids_list": doc_store.ids_list, "ent_pos_list": doc_store.ent_pos_list},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

rng_imp = np.random.default_rng(SEED)
train_unique_dids = np.unique(np.concatenate([train_df["d1"].to_numpy(), train_df["d2"].to_numpy()]))
IMPOSTOR_DOC_IDS = rng_imp.choice(train_unique_dids, size=min(256, len(train_unique_dids)), replace=False)

y_vo = val_open_df["label"].to_numpy(dtype=np.int32)
y_to = test_open_df["label"].to_numpy(dtype=np.int32)
print(f"unique_docs={len(doc_store.ids_list):,} impostors={len(IMPOSTOR_DOC_IDS)}")
'''),
    md_cell(r'''## 3. Symmetric Verifier, LLRD/Prior-WD & Residual OOD Gate

- `WeightedTopLayersPooling`: 24층 백본 상위 12개 층(`L13-L24`)을 학습 가능한 softmax 가중치로 결합
- `AttentiveStatsPooling`: 토큰별 어텐션 가중 평균(`1024d`)과 표준편차(`1024d`) 결합
- `SymmetricDebertaVerifier`: 대칭 상호작용 특징과 5-branch Multi-Sample Dropout 헤드
- `build_llrd_priorwd_optimizer`: 층별 학습률 감쇠(`gamma = sqrt(0.90) = 0.94868`)와 사전학습 가중치 앵커링(`Prior-WD`)
'''),
    code_cell(r'''class PairDataset(Dataset):
    def __init__(self, d1: np.ndarray, d2: np.ndarray, labels: np.ndarray | None, store: FastDocStore, is_train: bool = False, entity_mask_prob: float = 0.0, seed: int = 378):
        self.d1, self.d2, self.labels, self.store = d1, d2, labels, store
        self.is_train, self.entity_mask_prob = is_train, entity_mask_prob
        self.rng = np.random.default_rng(seed)

    def set_epoch_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.d1)

    def _tokens(self, did: int) -> np.ndarray:
        ids = self.store.ids_list[did]
        if self.is_train and self.entity_mask_prob > 0.0:
            ent = self.store.ent_pos_list[did]
            if len(ent) > 0:
                flips = self.rng.random(len(ent)) < self.entity_mask_prob
                if np.any(flips):
                    ids = ids.copy()
                    ids[ent[flips]] = MASK_TOKEN_ID
        return ids

    def __getitem__(self, idx: int):
        return self._tokens(int(self.d1[idx])), self._tokens(int(self.d2[idx])), (float(self.labels[idx]) if self.labels is not None else 0.0)


def dynamic_pair_collate(batch):
    bs = len(batch)
    l1, l2 = max(len(x[0]) for x in batch), max(len(x[1]) for x in batch)
    ids1, mask1 = np.full((bs, l1), PAD_TOKEN_ID, dtype=np.int64), np.zeros((bs, l1), dtype=np.int64)
    ids2, mask2 = np.full((bs, l2), PAD_TOKEN_ID, dtype=np.int64), np.zeros((bs, l2), dtype=np.int64)
    ys = np.zeros(bs, dtype=np.float32)
    for i, (a1, a2, y) in enumerate(batch):
        ids1[i, : len(a1)], mask1[i, : len(a1)] = a1, 1
        ids2[i, : len(a2)], mask2[i, : len(a2)] = a2, 1
        ys[i] = y
    return torch.from_numpy(ids1), torch.from_numpy(mask1), torch.from_numpy(ids2), torch.from_numpy(mask2), torch.from_numpy(ys)


class WeightedTopLayersPooling(nn.Module):
    def __init__(self, num_total_layers: int, top_k_layers: int):
        super().__init__()
        self.num_total_layers = num_total_layers
        self.top_k = top_k_layers
        self.layer_weights = nn.Parameter(torch.zeros(top_k_layers))

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        selected = torch.stack(hidden_states[-self.top_k :], dim=0)
        w = F.softmax(self.layer_weights.float(), dim=0).to(selected.dtype).view(-1, 1, 1, 1)
        return torch.sum(selected * w, dim=0)


class AttentiveStatsPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(), nn.Linear(hidden_size // 2, 1))

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = hidden.float()
        scores = self.attn(hidden).squeeze(-1).float().masked_fill(~mask.bool(), -1e9)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)
        mean = torch.sum(alpha * h, dim=1)
        var = torch.sum(alpha * ((h - mean.unsqueeze(1)) ** 2), dim=1).clamp(min=1e-6)
        return torch.cat([mean, torch.sqrt(var)], dim=-1)


class SymmetricDebertaVerifier(nn.Module):
    def __init__(
        self,
        model_name_or_config,
        use_weighted_layers: bool = True,
        use_asp_pooling: bool = True,
        use_msd_head: bool = True,
        emb_dim: int = 512,
        grad_ckpt: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        if isinstance(model_name_or_config, str) and pretrained:
            self.config = AutoConfig.from_pretrained(model_name_or_config)
            self.config.update({"output_hidden_states": True})
            # Force fp32 master weights; some transformers builds load DeBERTa-v3 in fp16
            self.backbone = AutoModel.from_pretrained(model_name_or_config, config=self.config).float()
        else:
            self.config = model_name_or_config
            self.config.output_hidden_states = True
            self.backbone = AutoModel.from_config(self.config).float()
        assert next(self.backbone.parameters()).dtype == torch.float32
        if grad_ckpt:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        h_size = self.config.hidden_size
        n_layers = self.config.num_hidden_layers
        self.top_k_layers = max(n_layers // 2, 1)
        self.use_weighted_layers = use_weighted_layers
        self.use_asp_pooling = use_asp_pooling
        self.use_msd_head = use_msd_head
        if use_weighted_layers:
            self.layer_pool = WeightedTopLayersPooling(n_layers, self.top_k_layers)
        if use_asp_pooling:
            self.token_pool = AttentiveStatsPooling(h_size)
            in_dim = h_size * 2
        else:
            in_dim = h_size

        self.proj = nn.Sequential(nn.Linear(in_dim, emb_dim))
        self.pair_mlp = nn.Sequential(nn.Linear(emb_dim * 4 + 1, emb_dim), nn.LayerNorm(emb_dim), nn.GELU())
        self.msd_dropouts = nn.ModuleList([nn.Dropout(p) for p in (0.10, 0.15, 0.20, 0.25, 0.30)])
        self.single_drop = nn.Dropout(0.15)
        self.classifier = nn.Linear(emb_dim, 1)

    def encode_document(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        seq_h = self.layer_pool(out.hidden_states) if self.use_weighted_layers else out.last_hidden_state
        if self.use_asp_pooling:
            pooled = self.token_pool(seq_h, attention_mask)
        else:
            mf = attention_mask.unsqueeze(-1).float()
            pooled = torch.sum(seq_h.float() * mf, dim=1) / torch.clamp(mf.sum(dim=1), min=1.0)
        u = self.proj(pooled.to(self.proj[0].weight.dtype))
        return u, F.normalize(u.float(), p=2, dim=-1)

    def classify_pair_embeddings(self, u: torch.Tensor, u_unit: torch.Tensor, v: torch.Tensor, v_unit: torch.Tensor):
        cos_sim = torch.sum(u_unit * v_unit, dim=-1, keepdim=True)
        uf, vf = u.float(), v.float()
        pair_feat = torch.cat([0.5 * (uf + vf), torch.abs(uf - vf), u_unit * v_unit, (u_unit - v_unit) ** 2, cos_sim], dim=-1)
        h = self.pair_mlp(pair_feat.to(self.pair_mlp[0].weight.dtype))
        if self.use_msd_head:
            logits = torch.mean(torch.stack([self.classifier(dp(h)) for dp in self.msd_dropouts], dim=0), dim=0)
        else:
            logits = self.classifier(self.single_drop(h))
        return logits.squeeze(-1), cos_sim.squeeze(-1)

    def forward(self, ids1, m1, ids2, m2):
        u, u_unit = self.encode_document(ids1, m1)
        v, v_unit = self.encode_document(ids2, m2)
        return self.classify_pair_embeddings(u, u_unit, v, v_unit)


class ResidualOODGate(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48), nn.LayerNorm(48), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(48, 24), nn.LayerNorm(24), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(24, 1),
        )
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, raw_logit: torch.Tensor) -> torch.Tensor:
        return self.scale * raw_logit + self.bias + self.net(x).squeeze(-1)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9985):
        self.decay = decay
        self.shadow = {k: v.detach().float().cpu().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float().cpu(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v.detach().cpu())

    def state_dict_copy(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


def cosent_loss(cos_sim: torch.Tensor, labels: torch.Tensor, scale: float = 20.0) -> torch.Tensor:
    cf = cos_sim.float()
    pos, neg = labels > 0.5, labels <= 0.5
    if (not torch.any(pos)) or (not torch.any(neg)):
        return cf.new_tensor(0.0)
    return F.softplus(torch.logsumexp(scale * cf[neg], dim=0) + torch.logsumexp(-scale * cf[pos], dim=0))


def build_llrd_priorwd_optimizer(
    model: SymmetricDebertaVerifier,
    backbone_lr: float = 1.6e-5,
    head_lr: float = 1.0e-4,
    use_llrd: bool = True,
    llrd_gamma: float = 0.94868,
    use_prior_wd: bool = True,
    weight_decay: float = 0.01,
):
    n_layers = model.config.num_hidden_layers
    no_decay = ("bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight")
    grouped: dict[tuple[float, float], list[nn.Parameter]] = {}
    prior_anchors: list[tuple[nn.Parameter, torch.Tensor]] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bb = name.startswith("backbone.")
        is_nd = any(k in name for k in no_decay) or param.ndim < 2
        if not is_bb:
            lr, wd = head_lr, (0.0 if is_nd else weight_decay)
        else:
            if use_llrd:
                m = re.search(r"encoder\.layer\.(\d+)\.", name)
                if m:
                    depth = (n_layers - 1) - int(m.group(1))
                elif "embeddings" in name:
                    depth = n_layers
                else:
                    depth = 0
                lr = backbone_lr * (llrd_gamma ** depth)
            else:
                lr = backbone_lr
            if use_prior_wd and (not is_nd):
                wd = 0.0
                prior_anchors.append((param, param.detach().clone()))
            else:
                wd = 0.0 if is_nd else weight_decay
        grouped.setdefault((lr, wd), []).append(param)

    param_groups = [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in grouped.items()]
    return torch.optim.AdamW(param_groups, eps=1e-6), prior_anchors


@torch.no_grad()
def apply_prior_wd_step(optimizer, prior_anchors: list[tuple[nn.Parameter, torch.Tensor]], alpha: float = 0.01) -> None:
    if not prior_anchors:
        return
    p_lr = {}
    for g in optimizer.param_groups:
        lr = g["lr"]
        for p in g["params"]:
            p_lr[id(p)] = lr
    for p, p0 in prior_anchors:
        lr = p_lr.get(id(p), 0.0)
        if lr > 0.0:
            p.data.add_(p0 - p.data, alpha=lr * alpha)
'''),
    md_cell("## 4. Dual-Rank ABTT + Multi-Std Feature Extraction & Evaluation Helpers"),
    code_cell(r'''def fit_abtt(imp_mat: np.ndarray, remove_top_k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    mu = imp_mat.mean(axis=0, keepdims=True).astype(np.float32)
    _, _, vt = np.linalg.svd((imp_mat - mu).astype(np.float32), full_matrices=False)
    return mu, vt[:remove_top_k].astype(np.float32)


def apply_abtt(arr: np.ndarray, mu: np.ndarray, top_v: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32) - mu
    x = x - (x @ top_v.T) @ top_v
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def impostor_stats(mat: np.ndarray, imp_mat: np.ndarray):
    sim = mat @ imp_mat.T
    ms = np.where(sim > 0.9995, np.nan, sim)
    mu = np.nanmean(ms, axis=1).astype(np.float32)
    sd = np.maximum(np.nanstd(ms, axis=1), 1e-4).astype(np.float32)
    sr = np.sort(np.where(np.isnan(ms), -1.0, ms), axis=1)
    return mu, sd, sr[:, -10:].mean(axis=1).astype(np.float32), sr[:, -1].astype(np.float32)


@torch.no_grad()
def fast_infer_logits(model: SymmetricDebertaVerifier, pair_dfs: list[pd.DataFrame], store: FastDocStore, batch_size: int = 32) -> list[np.ndarray]:
    model.eval()
    uniq = np.unique(np.concatenate([np.concatenate([p["d1"].to_numpy(), p["d2"].to_numpy()]) for p in pair_dfs]))
    d2l = {int(d): i for i, d in enumerate(uniq)}
    lens = np.fromiter((len(store.ids_list[int(d)]) for d in uniq), dtype=np.int32, count=len(uniq))
    order = np.argsort(lens)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    ordered = uniq[order]

    u_l, un_l = [], []
    for st in range(0, len(ordered), batch_size):
        bd = ordered[st : st + batch_size]
        ml = max(len(store.ids_list[int(d)]) for d in bd)
        ids = np.full((len(bd), ml), PAD_TOKEN_ID, dtype=np.int64)
        msk = np.zeros((len(bd), ml), dtype=np.int64)
        for i, d in enumerate(bd):
            arr = store.ids_list[int(d)]
            ids[i, : len(arr)], msk[i, : len(arr)] = arr, 1
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            u, un = model.encode_document(torch.from_numpy(ids).to(DEVICE, non_blocking=True), torch.from_numpy(msk).to(DEVICE, non_blocking=True))
        u_l.append(u.float().cpu())
        un_l.append(un.float().cpu())

    all_u, all_un = torch.cat(u_l, dim=0)[inv], torch.cat(un_l, dim=0)[inv]
    out_l = []
    for pdf in pair_dfs:
        loc1 = np.fromiter((d2l[int(d)] for d in pdf["d1"]), dtype=np.int32, count=len(pdf))
        loc2 = np.fromiter((d2l[int(d)] for d in pdf["d2"]), dtype=np.int32, count=len(pdf))
        parts = []
        for pst in range(0, len(pdf), 4096):
            s1, s2 = loc1[pst : pst + 4096], loc2[pst : pst + 4096]
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                lg, _ = model.classify_pair_embeddings(all_u[s1].to(DEVICE), all_un[s1].to(DEVICE), all_u[s2].to(DEVICE), all_un[s2].to(DEVICE))
            parts.append(lg.float().cpu().numpy())
        out_l.append(np.concatenate(parts, axis=0).astype(np.float32))
    return out_l


@torch.no_grad()
def encode_docs_rich(model: SymmetricDebertaVerifier, doc_ids: np.ndarray, store: FastDocStore, batch_size: int = 32) -> dict[str, np.ndarray]:
    """Encode unique documents across 4 linguistic bands (mean + token-std on L2-L18, mean on L22-L24)."""
    model.eval()
    lens = np.fromiter((len(store.ids_list[int(d)]) for d in doc_ids), dtype=np.int32, count=len(doc_ids))
    order = np.argsort(lens)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    ordered = doc_ids[order]

    bsl = CONFIG["band_slices"]
    cols = {k: [] for k in ("u", "unit", "b1", "s1", "b2", "s2", "b3", "s3", "b4")}
    for st in range(0, len(ordered), batch_size):
        bd = ordered[st : st + batch_size]
        ml = max(len(store.ids_list[int(d)]) for d in bd)
        ids = np.full((len(bd), ml), PAD_TOKEN_ID, dtype=np.int64)
        msk = np.zeros((len(bd), ml), dtype=np.int64)
        for i, d in enumerate(bd):
            arr = store.ids_list[int(d)]
            ids[i, : len(arr)], msk[i, : len(arr)] = arr, 1
        ids_g, msk_g = torch.from_numpy(ids).to(DEVICE, non_blocking=True), torch.from_numpy(msk).to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            out = model.backbone(input_ids=ids_g, attention_mask=msk_g, output_hidden_states=True)
            seq_h = model.layer_pool(out.hidden_states) if model.use_weighted_layers else out.last_hidden_state
            mf = msk_g.unsqueeze(-1).float()
            denom = torch.clamp(mf.sum(dim=1), min=1.0)
            pooled = model.token_pool(seq_h, msk_g) if model.use_asp_pooling else (torch.sum(seq_h.float() * mf, dim=1) / denom)
            u = model.proj(pooled.to(model.proj[0].weight.dtype))
            un = F.normalize(u.float(), p=2, dim=-1)

            def _ms(hs_slice):
                h = torch.stack(hs_slice, dim=0).float().mean(dim=0)
                mu = torch.sum(h * mf, dim=1) / denom
                vr = torch.sum(((h - mu.unsqueeze(1)) ** 2) * mf, dim=1) / denom
                return F.normalize(mu, p=2, dim=-1), F.normalize(torch.sqrt(vr.clamp(min=1e-6)), p=2, dim=-1)

            b1, s1 = _ms(out.hidden_states[bsl["b1"][0] : bsl["b1"][1]])
            b2, s2 = _ms(out.hidden_states[bsl["b2"][0] : bsl["b2"][1]])
            b3, s3 = _ms(out.hidden_states[bsl["b3"][0] : bsl["b3"][1]])
            h4 = torch.stack(out.hidden_states[bsl["b4"][0] : bsl["b4"][1]], dim=0).float().mean(dim=0)
            b4 = F.normalize(torch.sum(h4 * mf, dim=1) / denom, p=2, dim=-1)

        for k, t in [("u", u), ("unit", un), ("b1", b1), ("s1", s1), ("b2", b2), ("s2", s2), ("b3", b3), ("s3", s3), ("b4", b4)]:
            cols[k].append(t.float().cpu())

    out_d = {k: torch.cat(v, dim=0)[inv].numpy().astype(np.float32) for k, v in cols.items()}
    out_d["tok_len"] = lens.astype(np.float32)
    return out_d


@torch.no_grad()
def extract_dual_abtt_features(
    model: SymmetricDebertaVerifier,
    pair_dfs: list[pd.DataFrame],
    store: FastDocStore,
    impostor_doc_ids: np.ndarray,
    batch_size: int = 32,
    abtt_bundle: dict | None = None,
):
    """Extract Dual-Rank ABTT (r=3 and r=8), multi-depth std, centroid, and symmetric agreement vectors."""
    all_dids = [] if abtt_bundle is not None else [impostor_doc_ids]
    for pdf in pair_dfs:
        all_dids.extend([pdf["d1"].to_numpy(), pdf["d2"].to_numpy()])
    uniq = np.unique(np.concatenate(all_dids))
    d2l = {int(d): i for i, d in enumerate(uniq)}

    enc = encode_docs_rich(model, uniq, store, batch_size=batch_size)
    u_np, unit_np, doc_lens = enc["u"], enc["unit"], enc["tok_len"]
    u_norm = np.linalg.norm(u_np, axis=1).astype(np.float32)
    all_u_t, all_un_t = torch.from_numpy(u_np), torch.from_numpy(unit_np)

    if abtt_bundle is None:
        imp_locs = np.fromiter((d2l[int(d)] for d in impostor_doc_ids), dtype=np.int32, count=len(impostor_doc_ids))
        abtt_bundle = {"imp_raw_unit": unit_np[imp_locs].copy(), "bands": {}}
        for bname, src, rk in CONFIG["abtt_specs"]:
            mu, tv = fit_abtt(enc[src][imp_locs], remove_top_k=rk)
            abtt_bundle["bands"][bname] = {"src": src, "mu": mu, "top_v": tv, "imp_abtt": apply_abtt(enc[src][imp_locs], mu, tv)}

    abtt_mats = {}
    stats_d = {"unit_raw": impostor_stats(unit_np, abtt_bundle["imp_raw_unit"])}
    for bk, bp in abtt_bundle["bands"].items():
        abtt_mats[bk] = apply_abtt(enc[bp["src"]], bp["mu"], bp["top_v"])
        stats_d[bk] = impostor_stats(abtt_mats[bk], bp["imp_abtt"])

    sc_l, cdom_l, cagr_l, rl_l = [], [], [], []
    for pdf in pair_dfs:
        loc1 = np.fromiter((d2l[int(d)] for d in pdf["d1"]), dtype=np.int32, count=len(pdf))
        loc2 = np.fromiter((d2l[int(d)] for d in pdf["d2"]), dtype=np.int32, count=len(pdf))
        parts = []
        for pst in range(0, len(pdf), 4096):
            s1, s2 = loc1[pst : pst + 4096], loc2[pst : pst + 4096]
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                lg, _ = model.classify_pair_embeddings(all_u_t[s1].to(DEVICE), all_un_t[s1].to(DEVICE), all_u_t[s2].to(DEVICE), all_un_t[s2].to(DEVICE))
            parts.append(lg.float().cpu().numpy())
        raw_lg = np.concatenate(parts, axis=0).astype(np.float32)

        def _pasn(mat, st_tup):
            mu, sd, t10, _ = st_tup
            rc = np.sum(mat[loc1] * mat[loc2], axis=1).astype(np.float32)
            an = (0.5 * (((rc - mu[loc1]) / sd[loc1]) + ((rc - mu[loc2]) / sd[loc2]))).astype(np.float32)
            t10m = 0.5 * (t10[loc1] + t10[loc2])
            return rc, an, t10m, rc - t10m

        rc_u, an_u, t10_u, mar_u = _pasn(unit_np, stats_d["unit_raw"])
        rc_ua, an_ua, t10_ua, mar_ua = _pasn(abtt_mats["unit"], stats_d["unit"])
        rc_b1, an_b1, t10_b1, mar_b1 = _pasn(abtt_mats["b1"], stats_d["b1"])
        rc_b2, an_b2, t10_b2, mar_b2 = _pasn(abtt_mats["b2"], stats_d["b2"])
        rc_b3, an_b3, t10_b3, mar_b3 = _pasn(abtt_mats["b3"], stats_d["b3"])
        rc_b4, an_b4, t10_b4, mar_b4 = _pasn(abtt_mats["b4"], stats_d["b4"])
        rc_s1, an_s1, _, _ = _pasn(abtt_mats["s1"], stats_d["s1"])
        rc_s2, an_s2, _, _ = _pasn(abtt_mats["s2"], stats_d["s2"])
        rc_s3, an_s3, _, _ = _pasn(abtt_mats["s3"], stats_d["s3"])
        rc_u8, an_u8, t10_u8, mar_u8 = _pasn(abtt_mats["unit_r8"], stats_d["unit_r8"])
        rc_b1_8, an_b1_8, _, _ = _pasn(abtt_mats["b1_r8"], stats_d["b1_r8"])
        rc_b2_8, an_b2_8, _, _ = _pasn(abtt_mats["b2_r8"], stats_d["b2_r8"])
        rc_b4_8, an_b4_8, _, _ = _pasn(abtt_mats["b4_r8"], stats_d["b4_r8"])

        mu_u, sd_u, t10_arr, t1_arr = stats_d["unit_raw"]
        t10_min, t10_diff = np.minimum(t10_arr[loc1], t10_arr[loc2]), np.abs(t10_arr[loc1] - t10_arr[loc2])
        t1_m, mu_m, sd_m = 0.5 * (t1_arr[loc1] + t1_arr[loc2]), 0.5 * (mu_u[loc1] + mu_u[loc2]), 0.5 * (sd_u[loc1] + sd_u[loc2])
        cos_infl = rc_u / (np.maximum(t10_u, 0.0) + 0.05)

        gap_b1_b4, gap_b2_b4, gap_ua_u = an_b1 - an_b4, an_b2 - an_b4, rc_ua - rc_u
        gap_r8_r3_u, gap_r8_r3_b2 = an_u8 - an_ua, an_b2_8 - an_b2
        style_cons = (an_b1 + an_b2 + an_s1 + an_s2 + an_s3) / 5.0
        style_cons_r8 = (an_b1_8 + an_b2_8 + an_u8) / 3.0

        n_m, n_diff = 0.5 * (u_norm[loc1] + u_norm[loc2]), np.abs(u_norm[loc1] - u_norm[loc2])
        l2_u = np.linalg.norm(unit_np[loc1] - unit_np[loc2], axis=1).astype(np.float32)
        l1_u = np.sum(np.abs(unit_np[loc1] - unit_np[loc2]), axis=1).astype(np.float32)
        l2_raw = np.linalg.norm(u_np[loc1] - u_np[loc2], axis=1).astype(np.float32)
        l1t, l2t = doc_lens[loc1], doc_lens[loc2]
        len_min, len_sum = np.log1p(np.minimum(l1t, l2t)), np.log1p(l1t + l2t)
        len_rat = np.minimum(l1t, l2t) / np.maximum(np.maximum(l1t, l2t), 1.0)

        sc_m = np.column_stack([
            raw_lg, rc_u, an_u, mar_u,
            rc_ua, an_ua, mar_ua,
            rc_b1, an_b1, mar_b1, rc_b2, an_b2, mar_b2, rc_b3, an_b3, mar_b3, rc_b4, an_b4, mar_b4,
            rc_s1, an_s1, rc_s2, an_s2, rc_s3, an_s3,
            rc_u8, an_u8, mar_u8, rc_b1_8, an_b1_8, rc_b2_8, an_b2_8, rc_b4_8, an_b4_8,
            gap_b1_b4, gap_b2_b4, gap_ua_u, gap_r8_r3_u, gap_r8_r3_b2, style_cons, style_cons_r8,
            t10_u, t10_min, t10_diff, t1_m, mu_m, sd_m, cos_infl,
            t10_ua, t10_u8, t10_b1, t10_b2, t10_b4,
            n_m, n_diff, l2_u, l1_u, l2_raw, len_min, len_sum, len_rat,
            raw_lg * t10_u, raw_lg * t10_ua, raw_lg * t10_u8, raw_lg * mu_m, raw_lg * cos_infl,
            raw_lg * n_m, raw_lg * gap_r8_r3_u, an_ua * t10_ua, an_b2 * t10_b2, style_cons * t10_b1,
        ]).astype(np.float32)
        c_dom = np.hstack([
            0.5 * (abtt_mats["unit"][loc1] + abtt_mats["unit"][loc2]),
            0.5 * (abtt_mats["b1"][loc1] + abtt_mats["b1"][loc2]),
            0.5 * (abtt_mats["b2"][loc1] + abtt_mats["b2"][loc2]),
        ]).astype(np.float32)
        c_agr = np.hstack([
            abtt_mats["unit_r8"][loc1] * abtt_mats["unit_r8"][loc2],
            abtt_mats["b2_r8"][loc1] * abtt_mats["b2_r8"][loc2],
        ]).astype(np.float32)
        sc_l.append(sc_m); cdom_l.append(c_dom); cagr_l.append(c_agr); rl_l.append(raw_lg)
    return sc_l, cdom_l, cagr_l, rl_l, abtt_bundle


def compute_anchor_hub_priors(anchor_pairs_df: pd.DataFrame, anchor_logits: np.ndarray) -> dict[tuple[str, str], float]:
    ap = anchor_pairs_df[["cohort_domain", "cand_a", "cand_b"]].copy()
    ap["logit"] = anchor_logits
    both = pd.concat([ap, ap.rename(columns={"cand_a": "cand_b", "cand_b": "cand_a"})], ignore_index=True)
    p_means = both.groupby(["cohort_domain", "cand_a", "cand_b"], as_index=False)["logit"].mean()
    return {
        (dom, cid): float(np.mean(np.sort(grp["logit"].to_numpy())[-5:]))
        for (dom, cid), grp in p_means.groupby(["cohort_domain", "cand_a"], sort=False)
    }


def evaluate_closed_set_1to30(pairs_df: pd.DataFrame, pair_logits: np.ndarray, hub_priors=None, use_csls: bool = True, hub_weight: float = 0.85):
    work = pairs_df[["query_id", "cohort_domain", "true_author_id", "candidate_author_id"]].copy()
    work["logit"] = pair_logits
    cand_df = work.groupby(["query_id", "cohort_domain", "true_author_id", "candidate_author_id"], as_index=False)["logit"].mean()
    cand_df.rename(columns={"logit": "cand_score"}, inplace=True)
    if use_csls and hub_priors is not None and hub_weight > 0.0:
        p_vals = np.fromiter((hub_priors.get((d, c), 0.0) for d, c in zip(cand_df["cohort_domain"], cand_df["candidate_author_id"])), dtype=np.float64, count=len(cand_df))
        cand_df["cand_score"] = cand_df["cand_score"] - hub_weight * p_vals

    cand_df = cand_df.sort_values(["query_id", "cand_score"], ascending=[True, False])
    top1 = cand_df.groupby("query_id", as_index=False).first().rename(columns={"candidate_author_id": "pred_author_id", "cand_score": "top1_score"})
    top3_sets = cand_df.groupby("query_id")["candidate_author_id"].apply(lambda s: set(s.iloc[:3])).to_dict()
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
    n_u, ans = int(np.sum(is_u)), ~is_u
    pred = (y_prob >= tau).astype(np.int32)
    n_c = int(np.sum((pred == y_true) & ans))
    c_at_1 = (n_c + n_u * (n_c / n)) / n
    tp, fp = int(np.sum((pred == 1) & (y_true == 1) & ans)), int(np.sum((pred == 1) & (y_true == 0) & ans))
    fn_u = int(np.sum((y_true == 1) & ((pred == 0) | is_u)))
    d05 = 1.25 * tp + 0.25 * fn_u + fp
    f_05u = (1.25 * tp / d05) if d05 > 0 else 0.0
    pan_brier = 1.0 - float(brier_score_loss(y_true, np.where(is_u, 0.5, y_prob)))
    f1_bin = float(f1_score(y_true, pred, zero_division=0))
    f1_mac = float(f1_score(y_true, pred, average="macro", zero_division=0))
    return {
        "tau": round(float(tau), 4), "delta": round(float(delta), 4), "abstain_pct": round(float(n_u / n * 100.0), 2),
        "roc_auc": round(roc_auc, 5), "pan_c_at_1": round(float(c_at_1), 5), "pan_f_05u": round(float(f_05u), 5),
        "pan_brier": round(float(pan_brier), 5), "pan_overall": round(float(np.mean([roc_auc, c_at_1, f1_bin, f_05u, pan_brier])), 5),
        "forced_macro_f1": round(f1_mac, 5), "answered_acc": round(float(n_c / max(n - n_u, 1)), 5),
    }


def exact_mcnemar_and_bootstrap(name_a: str, name_b: str, y_true: np.ndarray, prob_a, pred_a: np.ndarray, prob_b, pred_b: np.ndarray, task_label: str, n_boot: int = 1000) -> dict:
    ca, cb = (pred_a == y_true).astype(np.int32), (pred_b == y_true).astype(np.int32)
    n10, n01 = int(np.sum((ca == 1) & (cb == 0))), int(np.sum((ca == 0) & (cb == 1)))
    p_val = float(binomtest(n10, n10 + n01, 0.5, alternative="two-sided").pvalue) if (n10 + n01) > 0 else 1.0
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    d_acc, d_auc = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        d_acc.append(float(np.mean(ca[idx] - cb[idx])))
        if prob_a is not None and prob_b is not None and len(np.unique(y_true[idx])) == 2:
            d_auc.append(float(roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx])))
    return {
        "Task": task_label, "Comparison": f"{name_a} vs {name_b}",
        "A_wins_n10": n10, "B_wins_n01": n01, "McNemar_p": round(p_val, 6),
        "Significant_0.05": bool(p_val < 0.05),
        "Delta_Acc_CI95": f"[{np.percentile(d_acc, 2.5)*100:+.2f}%, {np.percentile(d_acc, 97.5)*100:+.2f}%]",
        "Delta_AUC_CI95": f"[{np.percentile(d_auc, 2.5)*100:+.2f}%, {np.percentile(d_auc, 97.5)*100:+.2f}%]" if d_auc else "N/A",
    }
'''),
    md_cell(r'''## 5. Optional 4-Way Micro Hyperparameter Probe

기본값(`CONFIG["run_hp_probe"] = False`)은 사전 검증된 최적 설정(`backbone_lr=1.6e-5, head_lr=1.0e-4, LLRD gamma=0.94868, Prior-WD=0.01, Pure Label-Smoothed BCE`)을 바로 사용합니다.
4개 후보를 10% 층화 서브셋에서 재검증하려면 `CONFIG["run_hp_probe"] = True`로 변경하세요.
'''),
    code_cell(r'''best_hp = {
    "name": "Probe_B_LLRD_HigherLR_PureBCE",
    "backbone_lr": CONFIG["backbone_lr"],
    "head_lr": CONFIG["head_lr"],
    "use_llrd": CONFIG["use_llrd"],
    "use_prior_wd": CONFIG["use_prior_wd"],
    "cosent_weight": CONFIG["cosent_weight"],
}

if CONFIG["run_hp_probe"]:
    strat_key = train_df["domain"].astype(str) + "_" + train_df["label"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.10, random_state=SEED)
    p_idx, _ = next(sss.split(train_df, strat_key))
    probe_df = train_df.iloc[p_idx].reset_index(drop=True)

    candidates = [
        {"name": "Probe_A_LLRD_PriorWD_PureBCE", "backbone_lr": 1.2e-5, "head_lr": 8.0e-5, "use_llrd": True, "use_prior_wd": True, "cosent_weight": 0.0},
        {"name": "Probe_B_LLRD_HigherLR_PureBCE", "backbone_lr": 1.6e-5, "head_lr": 1.0e-4, "use_llrd": True, "use_prior_wd": True, "cosent_weight": 0.0},
        {"name": "Probe_C_LLRD_PriorWD_MildCoSENT", "backbone_lr": 1.2e-5, "head_lr": 8.0e-5, "use_llrd": True, "use_prior_wd": True, "cosent_weight": 0.08},
        {"name": "Probe_D_UniformLR_PureBCE", "backbone_lr": 1.2e-5, "head_lr": 8.0e-5, "use_llrd": False, "use_prior_wd": False, "cosent_weight": 0.0},
    ]
    probe_rows = []
    best_score = -1.0
    for cand in candidates:
        seed_everything(SEED)
        pm = SymmetricDebertaVerifier(CONFIG["model_name"], grad_ckpt=USE_GRAD_CKPT).to(DEVICE)
        opt, p_anc = build_llrd_priorwd_optimizer(pm, cand["backbone_lr"], cand["head_lr"], cand["use_llrd"], CONFIG["llrd_gamma"], cand["use_prior_wd"])
        ds = PairDataset(probe_df["d1"].to_numpy(), probe_df["d2"].to_numpy(), probe_df["label"].to_numpy(), doc_store, is_train=True, entity_mask_prob=CONFIG["entity_mask_prob"], seed=SEED)
        ld = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=dynamic_pair_collate, drop_last=True)
        tot_s = max(len(ld) // GRAD_ACCUM, 10)
        sch = get_cosine_schedule_with_warmup(opt, max(int(tot_s * 0.10), 2), tot_s)
        sc = torch.amp.GradScaler("cuda", init_scale=2048.0, enabled=(not USE_BF16))

        pm.train()
        opt.zero_grad(set_to_none=True)
        for si, batch in enumerate(ld, start=1):
            i1, m1, i2, m2, ys = [t.to(DEVICE, non_blocking=True) for t in batch]
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                lg, cs = pm(i1, m1, i2, m2)
                tgt = ys.float() * (1.0 - CONFIG["label_smoothing"]) + 0.5 * CONFIG["label_smoothing"]
                loss = F.binary_cross_entropy_with_logits(lg.float(), tgt)
                if cand["cosent_weight"] > 0.0:
                    loss = loss + cand["cosent_weight"] * cosent_loss(cs, ys)
                loss = loss / GRAD_ACCUM
            sc.scale(loss).backward()
            if si % GRAD_ACCUM == 0:
                sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(pm.parameters(), 1.0)
                sc.step(opt); sc.update()
                if p_anc:
                    apply_prior_wd_step(opt, p_anc, CONFIG["prior_wd_alpha"])
                sch.step(); opt.zero_grad(set_to_none=True)

        del opt, p_anc, sc, sch
        gc.collect(); torch.cuda.empty_cache()

        vo_lg, vc_lg = fast_infer_logits(pm, [val_open_df, val_closed_pairs_df], doc_store, EVAL_BATCH_SIZE)
        auc = float(roc_auc_score(y_vo, expit(vo_lg)))
        cl_s, _, _ = evaluate_closed_set_1to30(val_closed_pairs_df, vc_lg, use_csls=False)
        comp = 0.35 * cl_s["overall_top1_acc"] + 0.65 * auc
        probe_rows.append({"candidate": cand["name"], "closed_top1": cl_s["overall_top1_acc"], "open_auc": round(auc, 5), "composite": round(comp, 5)})
        print(f"[probe] {cand['name']} closed_top1={cl_s['overall_top1_acc']:.4f} open_auc={auc:.4f} comp={comp:.5f}")
        if comp > best_score:
            best_score, best_hp = comp, dict(cand)
        del pm
        gc.collect(); torch.cuda.empty_cache()

    pd.DataFrame(probe_rows).to_csv(OUTPUT_DIR / "large_micro_hp_probe_results.csv", index=False)

print(f"[hp] selected={best_hp['name']} backbone_lr={best_hp['backbone_lr']} head_lr={best_hp['head_lr']} llrd={best_hp['use_llrd']} prior_wd={best_hp['use_prior_wd']}")
'''),
    md_cell(r'''## 6. Backbone Training & Interpolated Checkpoint Soup

전체 84,860쌍을 2.5 epoch 학습하며 후반 체크포인트(`50%, 62.5%, 75%, 87.5%, 100%`)를 디스크에 저장합니다.
학습 종료 후 옵티마이저를 해제해 VRAM을 확보한 뒤 모든 스냅샷과 EMA를 평가하고, 상위 2개 체크포인트의 보간 수프(`0.65 * best + 0.35 * second`)를 최종 백본 가중치로 확정합니다.
'''),
    code_cell(r'''LARGE_CKPT = CKPT_DIR / "deberta_v3_large_soup.pt"
seed_everything(SEED)
large_model = SymmetricDebertaVerifier(CONFIG["model_name"], grad_ckpt=USE_GRAD_CKPT).to(DEVICE)

if LARGE_CKPT.exists() and (not CONFIG["force_retrain"]):
    large_model.load_state_dict(torch.load(LARGE_CKPT, map_location=DEVICE, weights_only=True))
    print(f"[ckpt] loaded {LARGE_CKPT}")
    soup_info = {"loaded_from_disk": True}
else:
    t0 = time.perf_counter()
    optimizer, prior_anchors = build_llrd_priorwd_optimizer(
        large_model,
        backbone_lr=best_hp["backbone_lr"],
        head_lr=best_hp["head_lr"],
        use_llrd=best_hp["use_llrd"],
        llrd_gamma=CONFIG["llrd_gamma"],
        use_prior_wd=best_hp["use_prior_wd"],
    )
    ema = ModelEMA(large_model, decay=CONFIG["ema_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=2048.0, enabled=(not USE_BF16))

    train_ds = PairDataset(
        train_df["d1"].to_numpy(), train_df["d2"].to_numpy(), train_df["label"].to_numpy(),
        doc_store, is_train=True, entity_mask_prob=CONFIG["entity_mask_prob"], seed=SEED,
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=dynamic_pair_collate, drop_last=True)

    steps_per_ep = max(len(loader) // GRAD_ACCUM, 1)
    total_steps = max(int(steps_per_ep * CONFIG["epochs"]), 20)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)
    snap_steps = sorted({max(int(total_steps * r), 1) for r in (0.50, 0.625, 0.75, 0.875, 1.00)})

    snap_paths: dict[str, Path] = {}
    opt_step, run_loss = 0, 0.0
    optimizer.zero_grad(set_to_none=True)

    for ep in range(int(math.ceil(CONFIG["epochs"]))):
        train_ds.set_epoch_seed(SEED + ep * 97)
        large_model.train()
        pbar = tqdm(loader, desc=f"ep {ep+1}", leave=False)
        for step_i, batch in enumerate(pbar, start=1):
            ids1, m1, ids2, m2, ys = [t.to(DEVICE, non_blocking=True) for t in batch]
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                logits, cos_sim = large_model(ids1, m1, ids2, m2)
                targets = ys.float() * (1.0 - CONFIG["label_smoothing"]) + 0.5 * CONFIG["label_smoothing"]
                loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
                if best_hp["cosent_weight"] > 0.0:
                    loss = loss + best_hp["cosent_weight"] * cosent_loss(cos_sim, ys)
                loss = loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            run_loss += loss.item() * GRAD_ACCUM

            if step_i % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(large_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if prior_anchors:
                    apply_prior_wd_step(optimizer, prior_anchors, CONFIG["prior_wd_alpha"])
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(large_model)
                opt_step += 1

                if opt_step % 100 == 0:
                    pbar.set_postfix(loss=f"{run_loss / (100.0 * GRAD_ACCUM):.4f}", step=f"{opt_step}/{total_steps}")
                    run_loss = 0.0
                if opt_step in snap_steps:
                    tag = f"step_{opt_step}"
                    sp = CKPT_DIR / f"snap_{tag}.pt"
                    torch.save({k: v.detach().cpu() for k, v in large_model.state_dict().items()}, sp)
                    snap_paths[tag] = sp
                if opt_step >= total_steps:
                    break
        if opt_step >= total_steps:
            break

    ema_path = CKPT_DIR / "snap_ema.pt"
    torch.save(ema.state_dict_copy(), ema_path)
    snap_paths["ema"] = ema_path

    # Free optimizer states and Prior-WD anchors before evaluating snapshots
    del optimizer, prior_anchors, scaler, scheduler, ema
    gc.collect()
    torch.cuda.empty_cache()

    snap_scores = []
    for tag, sp in snap_paths.items():
        sd = torch.load(sp, map_location="cpu", weights_only=True)
        large_model.load_state_dict(sd)
        vo_lg, vc_lg = fast_infer_logits(large_model, [val_open_df, val_closed_pairs_df], doc_store, EVAL_BATCH_SIZE)
        auc = float(roc_auc_score(y_vo, expit(vo_lg)))
        cl_s, _, _ = evaluate_closed_set_1to30(val_closed_pairs_df, vc_lg, use_csls=False)
        comp = 0.35 * cl_s["overall_top1_acc"] + 0.65 * auc
        snap_scores.append((comp, auc, cl_s["overall_top1_acc"], tag, sp))
        print(f"[snap] {tag:10s} closed_top1={cl_s['overall_top1_acc']:.4f} open_auc={auc:.5f} comp={comp:.5f}")
        del sd

    snap_scores.sort(key=lambda x: x[0], reverse=True)
    step_only = [s for s in snap_scores if s[3].startswith("step_")]
    alpha = CONFIG["soup_alpha"]

    if len(step_only) >= 2:
        t1, p1 = step_only[0][3], step_only[0][4]
        t2, p2 = step_only[1][3], step_only[1][4]
        sd1 = torch.load(p1, map_location="cpu", weights_only=True)
        sd2 = torch.load(p2, map_location="cpu", weights_only=True)
        soup_sd = {k: (alpha * sd1[k].float() + (1.0 - alpha) * sd2[k].float() if sd1[k].dtype.is_floating_point else sd1[k]) for k in sd1}
        large_model.load_state_dict(soup_sd)
        soup_info = {t1: alpha, t2: round(1.0 - alpha, 4)}
        del sd1, sd2, soup_sd
    else:
        t1, p1 = snap_scores[0][3], snap_scores[0][4]
        large_model.load_state_dict(torch.load(p1, map_location="cpu", weights_only=True))
        soup_info = {t1: 1.0}

    torch.save(large_model.state_dict(), LARGE_CKPT)
    print(f"[train] done in {(time.perf_counter()-t0)/60:.1f}m | soup={soup_info}")
    gc.collect()
    torch.cuda.empty_cache()
'''),
    md_cell(r'''## 7. Dual-Rank ABTT + Multi-Std + Symmetric Agreement PCA Calibration

24개 층에서 4개 깊이(`L2-6`, `L8-12`, `L14-18`의 평균·표준편차 및 `L22-24` 평균)를 추출하고, 256개 학습 사칭자 기준 이중 랭크 All-but-the-Top(`r=3`, `r=8`)과 대칭 합의축 PCA(`e1 * e2`, 16d)를 구성합니다.
5-Bucket 양방향 저자 배제 OOF로 `LogisticRegression(C=0.06) + 5-Seed ResidualOODGate`를 학습한 뒤 `sklearn` 의존성 없는 순수 텐서 게이트로 변환합니다.
'''),
    code_cell(r'''(S_vo, S_vc, S_ah), (Cd_vo, Cd_vc, Cd_ah), (Ca_vo, Ca_vc, Ca_ah), (RL_vo, RL_vc, RL_ah), abtt_bundle = extract_dual_abtt_features(
    large_model, [val_open_df, val_closed_pairs_df, anchor_hub_pairs_df], doc_store, IMPOSTOR_DOC_IDS, EVAL_BATCH_SIZE
)

val_auth_dom = {}
for a1, a2, dm in zip(val_open_df["author_1"].astype(str), val_open_df["author_2"].astype(str), val_open_df["domain"].astype(str)):
    val_auth_dom.setdefault(a1, dm)
    val_auth_dom.setdefault(a2, dm)

rng_b = np.random.default_rng(SEED)
val_auth_bkt = {}
for dm in sorted(set(val_auth_dom.values())):
    al = sorted([a for a, d in val_auth_dom.items() if d == dm])
    rng_b.shuffle(al)
    for i, a in enumerate(al):
        val_auth_bkt[a] = i % 5

vb1 = np.fromiter((val_auth_bkt[a] for a in val_open_df["author_1"].astype(str)), dtype=np.int32, count=len(val_open_df))
vb2 = np.fromiter((val_auth_bkt[a] for a in val_open_df["author_2"].astype(str)), dtype=np.int32, count=len(val_open_df))

oof_acc = np.zeros(len(val_open_df), dtype=np.float64)
oof_cnt = np.zeros(len(val_open_df), dtype=np.float64)
gate_folds_pure = []
ik = CONFIG["pca_interact_k"]
lw, mw = CONFIG["gate_lr_w"], 1.0 - CONFIG["gate_lr_w"]

with threadpool_limits(limits=1):
    for k in range(5):
        tr_m, va_m = (vb1 != k) & (vb2 != k), (vb1 == k) | (vb2 == k)
        p_dom = PCA(n_components=CONFIG["pca_dom_k"], random_state=SEED + 40 + k).fit(Cd_vo[tr_m])
        p_agr = PCA(n_components=CONFIG["pca_agree_k"], random_state=SEED + 80 + k).fit(Ca_vo[tr_m])

        zd_tr, zd_va = p_dom.transform(Cd_vo[tr_m]).astype(np.float32), p_dom.transform(Cd_vo[va_m]).astype(np.float32)
        za_tr, za_va = p_agr.transform(Ca_vo[tr_m]).astype(np.float32), p_agr.transform(Ca_vo[va_m]).astype(np.float32)
        F_tr = np.hstack([S_vo[tr_m], zd_tr, S_vo[tr_m, 0:1] * zd_tr[:, :ik], za_tr]).astype(np.float32)
        F_va = np.hstack([S_vo[va_m], zd_va, S_vo[va_m, 0:1] * zd_va[:, :ik], za_va]).astype(np.float32)

        sc_k = StandardScaler()
        X_tr = np.clip(sc_k.fit_transform(np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
        X_va = np.clip(sc_k.transform(np.nan_to_num(F_va, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)

        lr_k = LogisticRegression(C=CONFIG["gate_lr_C"], max_iter=400, solver="lbfgs").fit(X_tr, y_vo[tr_m])
        p_lr_va = lr_k.predict_proba(X_va)[:, 1]

        xt_t = torch.from_numpy(X_tr.astype(np.float32)).to(DEVICE)
        rlt_t = torch.from_numpy(RL_vo[tr_m]).to(DEVICE)
        yt_t = torch.from_numpy(y_vo[tr_m].astype(np.float32)).to(DEVICE)
        xv_t = torch.from_numpy(X_va.astype(np.float32)).to(DEVICE)
        rlv_t = torch.from_numpy(RL_vo[va_m]).to(DEVICE)

        sd_l, pm_l = [], []
        for s_off in CONFIG["gate_mlp_seeds"]:
            seed_everything(SEED + s_off + k)
            m = ResidualOODGate(X_tr.shape[1]).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=CONFIG["gate_mlp_lr"], weight_decay=CONFIG["gate_mlp_wd"])
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["gate_mlp_epochs"])
            m.train()
            for _ in range(CONFIG["gate_mlp_epochs"]):
                opt.zero_grad(set_to_none=True)
                F.binary_cross_entropy_with_logits(m(xt_t, rlt_t), yt_t).backward()
                opt.step(); sch.step()
            m.eval()
            with torch.no_grad():
                pm_l.append(torch.sigmoid(m(xv_t, rlv_t)).cpu().numpy())
            sd_l.append({kk: vv.detach().float().cpu().clone() for kk, vv in m.state_dict().items()})

        oof_acc[va_m] += lw * p_lr_va + mw * np.mean(pm_l, axis=0)
        oof_cnt[va_m] += 1.0
        gate_folds_pure.append({
            "pca_dom_mean": torch.from_numpy(p_dom.mean_.astype(np.float32)),
            "pca_dom_components": torch.from_numpy(p_dom.components_.astype(np.float32)),
            "pca_agr_mean": torch.from_numpy(p_agr.mean_.astype(np.float32)),
            "pca_agr_components": torch.from_numpy(p_agr.components_.astype(np.float32)),
            "scaler_mean": torch.from_numpy(sc_k.mean_.astype(np.float32)),
            "scaler_scale": torch.from_numpy(sc_k.scale_.astype(np.float32)),
            "lr_coef": torch.from_numpy(lr_k.coef_.ravel().astype(np.float32)),
            "lr_intercept": float(lr_k.intercept_.ravel()[0]),
            "lr_weight": float(lw),
            "mlp_weight": float(mw),
            "mlp_in_dim": int(sc_k.mean_.shape[0]),
            "mlp_state_dicts": sd_l,
        })

val_oof_p = oof_acc / oof_cnt


def predict_pure_gate(S_i: np.ndarray, Cd_i: np.ndarray, Ca_i: np.ndarray, RL_i: np.ndarray, folds: list[dict] = gate_folds_pure, dev: torch.device = DEVICE) -> np.ndarray:
    preds = []
    rl_t = torch.from_numpy(RL_i.astype(np.float32)).to(dev)
    rl_col = S_i[:, 0:1].astype(np.float32)
    for fd in folds:
        zd = (Cd_i.astype(np.float32) - fd["pca_dom_mean"].numpy()) @ fd["pca_dom_components"].numpy().T
        za = (Ca_i.astype(np.float32) - fd["pca_agr_mean"].numpy()) @ fd["pca_agr_components"].numpy().T
        Fm = np.nan_to_num(np.hstack([S_i.astype(np.float32), zd, rl_col * zd[:, :ik], za]), nan=0.0, posinf=0.0, neginf=0.0)
        sc_s = fd["scaler_scale"].numpy()
        Xm = np.clip((Fm - fd["scaler_mean"].numpy()) / np.where(sc_s == 0.0, 1.0, sc_s), -8.0, 8.0)
        p_lr = expit(Xm @ fd["lr_coef"].numpy() + float(fd["lr_intercept"]))
        xt = torch.from_numpy(Xm.astype(np.float32)).to(dev)
        mp = []
        for sd_m in fd["mlp_state_dicts"]:
            m = ResidualOODGate(int(fd["mlp_in_dim"])).to(dev)
            m.load_state_dict(sd_m); m.eval()
            with torch.no_grad():
                mp.append(torch.sigmoid(m(xt, rl_t)).cpu().numpy())
        preds.append(float(fd["lr_weight"]) * p_lr + float(fd["mlp_weight"]) * np.mean(mp, axis=0))
    return np.mean(preds, axis=0)


gkf_t = GroupKFold(n_splits=2)
t_grps = val_open_df["author_1"].astype(str).to_numpy()
tau_grid = np.linspace(0.30, 0.70, 161)
f_taus, f_dels = [], []
for c_tr, _ in gkf_t.split(val_open_df, y_vo, groups=t_grps):
    tf = float(tau_grid[int(np.argmax([f1_score(y_vo[c_tr], (val_oof_p[c_tr] >= t).astype(int), average="macro") for t in tau_grid]))])
    db, c1b = 0.0, -1.0
    for dv in np.linspace(0.0, 0.16, 81):
        mv = compute_open_pan_metrics(y_vo[c_tr], val_oof_p[c_tr], tau=tf, delta=float(dv))
        if mv["abstain_pct"] <= CONFIG["max_abstention_pct"] and mv["pan_c_at_1"] > c1b:
            c1b, db = mv["pan_c_at_1"], float(dv)
    f_taus.append(tf); f_dels.append(db)

best_tau, best_delta = float(np.mean(f_taus)), float(np.mean(f_dels))
val_op_f = compute_open_pan_metrics(y_vo, val_oof_p, tau=best_tau, delta=0.0)
val_op_a = compute_open_pan_metrics(y_vo, val_oof_p, tau=best_tau, delta=best_delta)

vc_p = np.clip(predict_pure_gate(S_vc, Cd_vc, Ca_vc, RL_vc), 1e-6, 1.0 - 1e-6)
ah_p = np.clip(predict_pure_gate(S_ah, Cd_ah, Ca_ah, RL_ah), 1e-6, 1.0 - 1e-6)
vc_g, ah_g = np.log(vc_p / (1.0 - vc_p)), np.log(ah_p / (1.0 - ah_p))

best_s1 = {"key": (-1.0, -1.0, -1.0)}
for m_nm, vsc, asc in [
    ("raw_logit", RL_vc, RL_ah),
    ("r4_gate_logit", vc_g, ah_g),
    ("blend_80gate_20raw", 0.80 * vc_g + 0.20 * RL_vc, 0.80 * ah_g + 0.20 * RL_ah),
    ("blend_65gate_35raw", 0.65 * vc_g + 0.35 * RL_vc, 0.65 * ah_g + 0.35 * RL_ah),
    ("blend_50gate_50raw", 0.50 * vc_g + 0.50 * RL_vc, 0.50 * ah_g + 0.50 * RL_ah),
]:
    hp = compute_anchor_hub_priors(anchor_hub_pairs_df, asc)
    for hw in CONFIG["csls_grid"]:
        cs, cd, cp = evaluate_closed_set_1to30(val_closed_pairs_df, vsc, hp, use_csls=(hw > 0), hub_weight=float(hw))
        km = (cs["overall_top1_acc"], cs["overall_macro_f1"], cs["overall_top3_acc"])
        if km > best_s1["key"]:
            best_s1 = {"key": km, "mode": m_nm, "hw": float(hw), "hp": hp, "sum": cs, "dom": cd, "preds": cp}

best_s1["preds"].to_csv(OUTPUT_DIR / "val_closed_predictions_deberta_large.csv", index=False)
vo_exp = val_open_df[["pair_id", "eval_stratum", "domain", "is_cross_topic", "is_cross_genre", "label"]].copy()
vo_exp["deberta_large_prob"] = np.round(val_oof_p, 6)
vo_exp["deberta_large_pred"] = (val_oof_p >= best_tau).astype(np.int32)
vo_exp.to_csv(OUTPUT_DIR / "val_open_oof_predictions_deberta_large.csv", index=False)

(OUTPUT_DIR / "val_summary_deberta_large.json").write_text(json.dumps({
    "track": "03_deberta_v3_large",
    "selected_hp": best_hp,
    "soup_weights": soup_info,
    "val_composite": round(0.35 * best_s1["sum"]["overall_top1_acc"] + 0.65 * val_op_a["pan_overall"], 5),
    "val_closed_set": best_s1["sum"],
    "val_open_forced": val_op_f,
    "val_open_abstention": val_op_a,
    "closed_mode": best_s1["mode"],
    "best_hub_weight": best_s1["hw"],
    "tau": round(best_tau, 4),
    "delta": round(best_delta, 4),
}, indent=2, ensure_ascii=False), encoding="utf-8")

print(
    f"[val] closed_top1={best_s1['sum']['overall_top1_acc']:.4f} ({best_s1['mode']}, hw={best_s1['hw']}) | "
    f"open_auc={val_op_a['roc_auc']:.5f} pan_overall={val_op_a['pan_overall']:.5f} "
    f"forced_f1={val_op_f['forced_macro_f1']:.5f} tau={best_tau:.4f} delta={best_delta:.4f}"
)
display(best_s1["dom"])
'''),
    md_cell("## 8. Locked Test Evaluation & 3-Way Head-to-Head Comparison"),
    code_cell(r'''if not CONFIG["run_final_test"]:
    print("[test] skipped (set CONFIG['run_final_test'] = True to evaluate locked test split)")
    S_to, Cd_to, Ca_to, to_lg, to_p, sym_err = None, None, None, None, None, 0.0
else:
    (S_to, S_tc), (Cd_to, Cd_tc), (Ca_to, Ca_tc), (to_lg, tc_lg), _ = extract_dual_abtt_features(
        large_model, [test_open_df, test_closed_pairs_df], doc_store, IMPOSTOR_DOC_IDS, EVAL_BATCH_SIZE, abtt_bundle=abtt_bundle
    )
    to_p = np.clip(predict_pure_gate(S_to, Cd_to, Ca_to, to_lg), 1e-6, 1.0 - 1e-6)
    tc_p = np.clip(predict_pure_gate(S_tc, Cd_tc, Ca_tc, tc_lg), 1e-6, 1.0 - 1e-6)
    tc_g = np.log(tc_p / (1.0 - tc_p))

    s1_mode, hw = best_s1["mode"], best_s1["hw"]
    if s1_mode == "raw_logit":
        tc_score = tc_lg
    elif s1_mode == "r4_gate_logit":
        tc_score = tc_g
    elif s1_mode == "blend_80gate_20raw":
        tc_score = 0.80 * tc_g + 0.20 * tc_lg
    elif s1_mode == "blend_65gate_35raw":
        tc_score = 0.65 * tc_g + 0.35 * tc_lg
    else:
        tc_score = 0.50 * tc_g + 0.50 * tc_lg

    test_cl_sum, test_cl_dom, test_cl_preds = evaluate_closed_set_1to30(test_closed_pairs_df, tc_score, best_s1["hp"], use_csls=(hw > 0), hub_weight=hw)
    test_op_f = compute_open_pan_metrics(y_to, to_p, tau=best_tau, delta=0.0)
    test_op_a = compute_open_pan_metrics(y_to, to_p, tau=best_tau, delta=best_delta)

    # Symmetry & latency check
    sub_ab = test_open_df.iloc[:32].copy().reset_index(drop=True)
    sub_ba = sub_ab.rename(columns={"d1": "d2", "d2": "d1", "text_1": "text_2", "text_2": "text_1"})
    (sab, sba), (cdab, cdba), (caab, caba), (lab, lba), _ = extract_dual_abtt_features(large_model, [sub_ab, sub_ba], doc_store, IMPOSTOR_DOC_IDS, EVAL_BATCH_SIZE, abtt_bundle=abtt_bundle)
    sym_err = float(np.max(np.abs(predict_pure_gate(sab, cdab, caab, lab) - predict_pure_gate(sba, cdba, caba, lba))))

    torch.cuda.synchronize()
    t_lat = time.perf_counter()
    sl, cdl, cal, ll, _ = extract_dual_abtt_features(large_model, [test_open_df.iloc[:64].copy().reset_index(drop=True)], doc_store, IMPOSTOR_DOC_IDS, EVAL_BATCH_SIZE, abtt_bundle=abtt_bundle)
    _ = predict_pure_gate(sl[0], cdl[0], cal[0], ll[0])
    torch.cuda.synchronize()
    ms_per_pair = (time.perf_counter() - t_lat) * 1000.0 / 64.0

    st_test_df = pd.DataFrame([
        {"Eval_Stratum": st_nm, "Pairs": len(grp), **compute_open_pan_metrics(y_to[grp.index.to_numpy()], to_p[grp.index.to_numpy()], best_tau, best_delta)}
        for st_nm, grp in test_open_df.groupby("eval_stratum", sort=False)
    ]).sort_values("Pairs", ascending=False).reset_index(drop=True)

    dom_rows = []
    for dm_nm, grp in test_open_df.groupby("domain", sort=False):
        idx_d = grp.index.to_numpy()
        if len(idx_d) < 20:
            continue
        yd, pf = y_to[idx_d], to_p[idx_d]
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
        {"Doc_Length_Bucket": str(wb), "Pairs": len(grp), **compute_open_pan_metrics(y_to[grp.index.to_numpy()], to_p[grp.index.to_numpy()], best_tau, best_delta)}
        for wb, grp in test_open_df.groupby(w_bins, observed=False) if len(grp) > 10
    ])

    test_cl_dom.to_csv(OUTPUT_DIR / "final_test_closed_by_domain_large.csv", index=False)
    test_cl_preds.to_csv(OUTPUT_DIR / "test_closed_predictions_deberta_large.csv", index=False)
    st_test_df.to_csv(OUTPUT_DIR / "final_test_open_by_stratum_large.csv", index=False)
    dom_test_df.to_csv(OUTPUT_DIR / "final_test_open_by_domain_large.csv", index=False)
    len_test_df.to_csv(OUTPUT_DIR / "final_test_open_by_length_large.csv", index=False)

    to_pred_bin = (to_p >= best_tau).astype(np.int32)
    to_exp = test_open_df[["pair_id", "eval_stratum", "domain", "topic_1", "topic_2", "is_cross_topic", "is_cross_genre", "label"]].copy()
    to_exp["deberta_large_prob"] = np.round(to_p, 6)
    to_exp["deberta_large_pred"] = to_pred_bin
    to_exp["is_abstained"] = ((to_p > best_tau - best_delta) & (to_p < best_tau + best_delta)).astype(np.int32)
    to_exp["is_correct"] = (to_pred_bin == y_to).astype(np.int32)
    to_exp.to_csv(OUTPUT_DIR / "test_open_predictions_deberta_large.csv", index=False)

    (OUTPUT_DIR / "final_locked_test_large_summary.json").write_text(json.dumps({
        "track": "03_deberta_v3_large",
        "soup_weights": soup_info,
        "closed_set_matching": {"mode": s1_mode, "csls_hub_weight": hw},
        "open_thresholds": {"tau": round(best_tau, 4), "delta": round(best_delta, 4)},
        "system_metrics": {
            "gpu_device": GPU_NAME, "gpu_latency_ms_per_pair": round(ms_per_pair, 2),
            "symmetry_max_error": float(f"{sym_err:.2e}"),
            "num_parameters_M": round(sum(p.numel() for p in large_model.parameters()) / 1e6, 2),
        },
        "scenario_1_closed_set_test": test_cl_sum,
        "scenario_2_open_cross_test_forced": test_op_f,
        "scenario_2_open_cross_test_abstention": test_op_a,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[test] closed_top1={test_cl_sum['overall_top1_acc']:.4f} closed_top3={test_cl_sum['overall_top3_acc']:.4f} | "
        f"open_auc={test_op_a['roc_auc']:.5f} pan_overall={test_op_a['pan_overall']:.5f} "
        f"sym_err={sym_err:.2e} latency={ms_per_pair:.1f}ms/pair"
    )
    display(test_cl_dom)
    display(dom_test_df)

    # 4-panel test summary chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    x_d = np.arange(len(test_cl_dom))
    axes[0, 0].bar(x_d - 0.18, test_cl_dom["Top1_Accuracy"], width=0.35, color="#1f77b4", label="Top-1")
    axes[0, 0].bar(x_d + 0.18, test_cl_dom["Top3_Accuracy"], width=0.35, color="#6baed6", label="Top-3")
    axes[0, 0].axhline(test_cl_sum["overall_top1_acc"], ls="--", color="#d62728", label=f"Overall Top-1 ({test_cl_sum['overall_top1_acc']*100:.1f}%)")
    axes[0, 0].set_xticks(x_d); axes[0, 0].set_xticklabels(test_cl_dom["Cohort_Domain"]); axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Closed-Set (1:30) by Domain"); axes[0, 0].legend(loc="lower right")

    fpr, tpr, _ = roc_curve(y_to, to_p)
    pt_c, pp_c = calibration_curve(y_to, to_p, n_bins=10)
    axes[0, 1].plot(fpr, tpr, lw=2.2, color="#2ca02c", label=f"ROC (AUC={test_op_a['roc_auc']:.4f})")
    axes[0, 1].plot(pp_c, pt_c, "s-", lw=1.6, color="#ff7f0e", label=f"Calibration (Brier={test_op_a['pan_brier']:.4f})")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 1].set_title("Open-Set ROC & Reliability"); axes[0, 1].legend(loc="lower right")

    ys = np.arange(len(st_test_df))
    axes[1, 0].barh(ys - 0.18, st_test_df["roc_auc"], height=0.35, color="#2ca02c", label="ROC-AUC")
    axes[1, 0].barh(ys + 0.18, st_test_df["pan_overall"], height=0.35, color="#9467bd", label="PAN Overall")
    axes[1, 0].set_yticks(ys); axes[1, 0].set_yticklabels(st_test_df["Eval_Stratum"], fontsize=8.5); axes[1, 0].set_xlim(0.45, 1.03)
    axes[1, 0].set_title("Performance by Stratum"); axes[1, 0].legend(loc="lower right")

    yd = np.arange(len(dom_test_df))
    axes[1, 1].barh(yd - 0.18, dom_test_df["Forced_MacroF1"], height=0.35, color="#1f77b4", label="Macro-F1")
    axes[1, 1].barh(yd + 0.18, dom_test_df["Test_FPR_%"] / 100.0, height=0.35, color="#d62728", label="FPR")
    axes[1, 1].set_yticks(yd); axes[1, 1].set_yticklabels(dom_test_df["Domain"], fontsize=8.5); axes[1, 1].set_xlim(0, 1.03)
    axes[1, 1].set_title("Domain Macro-F1 vs FPR"); axes[1, 1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_test_03_deberta_large_4panel.png", bbox_inches="tight")
    plt.show()

    # Optional 3-way statistical comparison when Track 01 and Track 02 result archives are present
    peer_specs = [
        ("01. Stylometric ML", ROOT_OUT / "01_stylometric_ml", "final_locked_test_summary.json", "test_closed_predictions_ml.csv", "test_open_predictions_ml.csv", "ml_sota_prob", "ml_sota_pred"),
        ("02. DeBERTa-v3-base", ROOT_OUT / "02_deberta_v3_base", "final_locked_test_base_summary.json", "test_closed_predictions_deberta_base.csv", "test_open_predictions_deberta_base.csv", "deberta_base_prob", "deberta_base_pred"),
        ("03. DeBERTa-v3-large", OUTPUT_DIR, "final_locked_test_large_summary.json", "test_closed_predictions_deberta_large.csv", "test_open_predictions_deberta_large.csv", "deberta_large_prob", "deberta_large_pred"),
    ]
    t_sums, t_cl, t_op = [], {}, {}
    for lbl, dp, sfn, cfn, ofn, pcol, dcol in peer_specs:
        if (dp / sfn).exists():
            js = json.loads((dp / sfn).read_text(encoding="utf-8"))
            t_sums.append({
                "Track": lbl,
                "Test_Closed_Top1": js["scenario_1_closed_set_test"]["overall_top1_acc"],
                "Test_Closed_Top3": js["scenario_1_closed_set_test"]["overall_top3_acc"],
                "Test_Open_ROC_AUC": js["scenario_2_open_cross_test_abstention"]["roc_auc"],
                "Test_Open_PAN_c@1": js["scenario_2_open_cross_test_abstention"]["pan_c_at_1"],
                "Test_Open_PAN_Overall": js["scenario_2_open_cross_test_abstention"]["pan_overall"],
                "Test_Open_Forced_F1": js["scenario_2_open_cross_test_forced"]["forced_macro_f1"],
            })
            if (dp / cfn).exists():
                t_cl[lbl] = pd.read_csv(dp / cfn).sort_values("query_id").reset_index(drop=True)
            if (dp / ofn).exists():
                t_op[lbl] = (pd.read_csv(dp / ofn).sort_values("pair_id").reset_index(drop=True), pcol, dcol)

    if len(t_sums) >= 2:
        comp_df = pd.DataFrame(t_sums)
        comp_df.to_csv(OUTPUT_DIR / "comparison_3way_final_test.csv", index=False)
        display(comp_df)

        stat_rows = []
        names = list(t_op.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                na, nb = names[j], names[i]
                if na in t_cl and nb in t_cl:
                    ca, cb = t_cl[na], t_cl[nb]
                    stat_rows.append(exact_mcnemar_and_bootstrap(na, nb, np.ones(len(ca), dtype=np.int32), None, ca["is_top1_correct"].to_numpy(), None, cb["is_top1_correct"].to_numpy(), "Scenario 1: Closed 1:30 Top-1", n_boot=2000))
                dfa, pca, pda = t_op[na]
                dfb, pcb, pdb = t_op[nb]
                stat_rows.append(exact_mcnemar_and_bootstrap(na, nb, dfa["label"].to_numpy(dtype=np.int32), dfa[pca].to_numpy(), dfa[pda].to_numpy(), dfb[pcb].to_numpy(), dfb[pdb].to_numpy(), "Scenario 2: Open & Cross-Domain", n_boot=1000))
        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(OUTPUT_DIR / "comparison_3way_statistical_tests.csv", index=False)
        display(stat_df)
'''),
    md_cell("## 9. Pure-Tensor Deployment Export & Results Archive"),
    code_cell(r'''anc_dids = doc_store.add_and_map(anchors_df["text"])
anc_enc = encode_docs_rich(large_model, anc_dids, doc_store, batch_size=EVAL_BATCH_SIZE)

abtt_pure = {"imp_raw_unit": torch.from_numpy(abtt_bundle["imp_raw_unit"].astype(np.float32)), "bands": {}}
anc_abtt = {}
for bk, bp in abtt_bundle["bands"].items():
    abtt_pure["bands"][bk] = {
        "src": str(bp["src"]),
        "mu": torch.from_numpy(bp["mu"].astype(np.float32)),
        "top_v": torch.from_numpy(bp["top_v"].astype(np.float32)),
        "imp_abtt": torch.from_numpy(bp["imp_abtt"].astype(np.float32)),
    }
    anc_abtt[bk] = torch.from_numpy(apply_abtt(anc_enc[bp["src"]], bp["mu"], bp["top_v"]).astype(np.float32))

model_out = Path.cwd() / "model_03_deberta_v3_large.pt"
torch.save({
    "artifact_version": "2025_dl_large_dual_abtt_multistd_agree_v3",
    "model_name": CONFIG["model_name"],
    "max_doc_tokens": CONFIG["max_doc_tokens"],
    "model_init_kwargs": {
        "use_weighted_layers": bool(large_model.use_weighted_layers),
        "use_asp_pooling": bool(large_model.use_asp_pooling),
        "use_msd_head": bool(large_model.use_msd_head),
        "emb_dim": int(large_model.classifier.in_features),
    },
    "band_slices": {k: list(v) for k, v in CONFIG["band_slices"].items()},
    "soup_weights": soup_info,
    "model_state_dict_fp16": {k: (v.detach().cpu().half() if v.dtype.is_floating_point else v.detach().cpu()) for k, v in large_model.state_dict().items()},
    "abtt_bundle": abtt_pure,
    "gate_folds": gate_folds_pure,
    "tau": float(best_tau),
    "delta": float(best_delta),
    "closed_mode": best_s1["mode"],
    "best_hub_weight": float(best_s1["hw"]),
    "hub_priors": {f"{d}|||{c}": float(v) for (d, c), v in best_s1["hp"].items()},
    "closed_anchors_meta": anchors_df[["cohort_domain", "candidate_author_id", "anchor_idx"]].to_dict(orient="records"),
    "closed_anchors_u": torch.from_numpy(anc_enc["u"]).half(),
    "closed_anchors_unit": torch.from_numpy(anc_enc["unit"]),
    "closed_anchors_tok_len": torch.from_numpy(anc_enc["tok_len"]),
    "closed_anchors_abtt": anc_abtt,
}, model_out)

chk = torch.load(model_out, map_location="cpu", weights_only=False)
if to_p is not None:
    reload_diff = float(np.max(np.abs(to_p[:16] - np.clip(predict_pure_gate(S_to[:16], Cd_to[:16], Ca_to[:16], to_lg[:16], chk["gate_folds"], DEVICE), 1e-6, 1.0 - 1e-6))))
    assert reload_diff < 1e-5, f"Reload mismatch: {reload_diff}"
else:
    reload_diff = 0.0
del chk

bundle_out = Path.cwd() / "results_03_deberta_v3_large.zip"
with zipfile.ZipFile(bundle_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fp in sorted(OUTPUT_DIR.rglob("*")):
        if fp.is_file() and fp.suffix not in (".pt", ".npz") and "checkpoints" not in fp.parts:
            zf.write(fp, arcname=str(fp.relative_to(ROOT_OUT)))

print(f"[export] model={model_out.name} ({model_out.stat().st_size/(1024**2):.1f} MB) reload_diff={reload_diff:.2e}")
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
    build_notebook(Path(r"D:\Projects\NLU-Authorship-Verification\notebooks\colab\train_03_deberta_v3_large.ipynb"))
