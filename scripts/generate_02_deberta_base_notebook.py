"""Build train_02_deberta_v3_base.ipynb: unified DeBERTa-v3-base training & 4-depth ABTT OOD calibration notebook."""
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
    md_cell(r'''# 02. DeBERTa-v3-base Authorship Verification

`microsoft/deberta-v3-base` (12-layer, 186M params) 기반 순수 딥러닝 동일 저자 판별 학습·평가 노트북입니다.
`benchmark_1024.zip`을 업로드하고 순서대로 실행하면 백본 파인튜닝, Greedy Model Soup, 4계층 언어 깊이 + All-but-the-Top 신경망 OOD 보정, Locked Test 평가까지 일괄 수행됩니다.

- 권장 환경: GPU 런타임 (`L4 24GB` / `A100 40GB` / `A100 80GB`)
- 입력 파일: `benchmark_1024.zip`
- 주요 출력: `results_02_deberta_v3_base.zip` (`verified_dl_recipe.json` 포함)
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

if VRAM_GB >= 35.0:
    BATCH_SIZE, GRAD_ACCUM, USE_GRAD_CKPT, EVAL_BATCH_SIZE = 16, 2, False, 128
elif VRAM_GB >= 20.0:
    BATCH_SIZE, GRAD_ACCUM, USE_GRAD_CKPT, EVAL_BATCH_SIZE = 8, 4, False, 64
else:
    BATCH_SIZE, GRAD_ACCUM, USE_GRAD_CKPT, EVAL_BATCH_SIZE = 4, 8, True, 32

CONFIG = {
    "model_name": "microsoft/deberta-v3-base",
    "max_doc_tokens": 512,
    "run_ablation_ladder": False,   # set True to rerun 4-step 40% ablation study
    "run_final_test": True,         # set False during validation-only tuning
    "force_retrain": False,
    "epochs": 2.5,
    "backbone_lr": 2.2e-5,
    "head_lr": 1.0e-4,
    "entity_mask_prob": 0.15,
    "label_smoothing": 0.02,
    "ema_decay": 0.9985,
    "band_slices": {"b1": (1, 4), "b2": (4, 7), "b3": (7, 10), "b4": (11, 13)},
    "pca_components": 36,
    "pca_interact_k": 10,
    "gate_lr_C": 0.08,
    "gate_lr_w": 0.35,
    "gate_mlp_seeds": [400, 500, 600],
    "csls_grid": [0.0, 0.25, 0.40, 0.50, 0.65, 0.75, 0.85],
    "max_abstention_pct": 15.0,
}

print(
    f"torch={torch.__version__} transformers={transformers.__version__} "
    f"gpu={GPU_NAME} ({VRAM_GB:.1f}GB) amp={'bf16' if USE_BF16 else 'fp16'} "
    f"bs={BATCH_SIZE}x{GRAD_ACCUM}"
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
OUTPUT_DIR = ROOT_OUT / "02_deberta_v3_base"
FIG_DIR = OUTPUT_DIR / "figures"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
RECIPE_PATH = ROOT_OUT / "verified_dl_recipe.json"
TOKEN_CACHE_PATH = ROOT_OUT / "cache_deberta_v3_tokens_512.pkl"
for d in (ROOT_OUT, OUTPUT_DIR, FIG_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

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


class FastDocStore:
    """Pre-tokenized unique document table with mid-sentence proper-noun token spans."""

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
                        if clean in (".", "!", "?", "\n", ":", ";", '"', "”", "“"):
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
        pickle.dump({"vocab_size": tokenizer.vocab_size, "max_len": CONFIG["max_doc_tokens"], "map": doc_store.map, "ids_list": doc_store.ids_list, "ent_pos_list": doc_store.ent_pos_list}, f)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

rng_imp = np.random.default_rng(SEED)
train_unique_dids = np.unique(np.concatenate([train_df["d1"].to_numpy(), train_df["d2"].to_numpy()]))
IMPOSTOR_DOC_IDS = rng_imp.choice(train_unique_dids, size=min(256, len(train_unique_dids)), replace=False)

y_vo = val_open_df["label"].to_numpy(dtype=np.int32)
y_to = test_open_df["label"].to_numpy(dtype=np.int32)
print(f"unique_docs={len(doc_store.ids_list):,} impostors={len(IMPOSTOR_DOC_IDS)}")
'''),
    md_cell(r'''## 3. Symmetric Siamese Verifier & Residual OOD Gate

- `WeightedTopLayersPooling`: 상위 50% 층(`L7-L12`)을 학습 가능한 softmax 가중치로 결합
- `AttentiveStatsPooling`: 토큰별 어텐션 가중 평균과 표준편차를 결합해 문체 호흡 변동성 반영
- `SymmetricDebertaVerifier`: `(u+v)/2`, `|u-v|`, `u*v`, `(u-v)^2`, `cos(u,v)` 대칭 결합 및 5-branch Multi-Sample Dropout
- `ResidualOODGate`: 원본 로짓에 4계층/All-but-the-Top/PCA 잔차 보정만 학습하는 경량 PyTorch MLP
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
    def __init__(self, top_k_layers: int):
        super().__init__()
        self.top_k = top_k_layers
        self.layer_weights = nn.Parameter(torch.zeros(top_k_layers))

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        selected = torch.stack(hidden_states[-self.top_k :], dim=0)
        w = F.softmax(self.layer_weights.float(), dim=0).to(selected.dtype).view(-1, 1, 1, 1)
        return torch.sum(selected * w, dim=0)

    def normalized_weights(self) -> np.ndarray:
        return F.softmax(self.layer_weights.detach().float(), dim=0).cpu().numpy()


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
    def __init__(self, model_name: str, use_weighted_layers: bool = True, use_asp: bool = True, use_msd: bool = True, emb_dim: int = 512, grad_ckpt: bool = False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({"output_hidden_states": True})
        # Keep fp32 master weights; some transformers builds load DeBERTa-v3 weights in fp16 by default
        self.backbone = AutoModel.from_pretrained(model_name, config=self.config).float()
        assert next(self.backbone.parameters()).dtype == torch.float32
        if grad_ckpt:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        h_size = self.config.hidden_size
        self.top_k_layers = max(self.config.num_hidden_layers // 2, 1)
        self.use_weighted_layers, self.use_asp, self.use_msd = use_weighted_layers, use_asp, use_msd
        if use_weighted_layers:
            self.layer_pool = WeightedTopLayersPooling(self.top_k_layers)
        if use_asp:
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
        if self.use_asp:
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
        if self.use_msd:
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
'''),
    md_cell("## 4. Inference, 4-Depth ABTT Feature Extraction & Scoring Helpers"),
    code_cell(r'''def fit_abtt(imp_mat: np.ndarray, remove_top_k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    # All-but-the-Top (Mu & Viswanath, ICLR 2018) fit strictly on frozen training impostors
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
def fast_infer_logits(model: SymmetricDebertaVerifier, pair_dfs: list[pd.DataFrame], store: FastDocStore, batch_size: int = 64) -> list[np.ndarray]:
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
def extract_4depth_abtt_features(
    model: SymmetricDebertaVerifier,
    pair_dfs: list[pd.DataFrame],
    store: FastDocStore,
    impostor_doc_ids: np.ndarray,
    batch_size: int = 64,
    abtt_bundle: dict | None = None,
):
    model.eval()
    all_dids = [] if abtt_bundle is not None else [impostor_doc_ids]
    for pdf in pair_dfs:
        all_dids.extend([pdf["d1"].to_numpy(), pdf["d2"].to_numpy()])
    uniq = np.unique(np.concatenate(all_dids))
    d2l = {int(d): i for i, d in enumerate(uniq)}
    lens = np.fromiter((len(store.ids_list[int(d)]) for d in uniq), dtype=np.int32, count=len(uniq))
    order = np.argsort(lens)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    ordered = uniq[order]

    bsl = CONFIG["band_slices"]
    cols = {k: [] for k in ("u", "unit", "b1", "b2", "b3", "b4", "s2")}
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
            pooled = model.token_pool(seq_h, msk_g) if model.use_asp else (torch.sum(seq_h.float() * mf, dim=1) / denom)
            u = model.proj(pooled.to(model.proj[0].weight.dtype))
            un = F.normalize(u.float(), p=2, dim=-1)

            def _mp(sl):
                h = torch.stack(sl, dim=0).float().mean(dim=0)
                return F.normalize(torch.sum(h * mf, dim=1) / denom, p=2, dim=-1)

            hb2 = torch.stack(out.hidden_states[bsl["b2"][0] : bsl["b2"][1]], dim=0).float().mean(dim=0)
            b2m = torch.sum(hb2 * mf, dim=1) / denom
            b2v = torch.sum(((hb2 - b2m.unsqueeze(1)) ** 2) * mf, dim=1) / denom
            b1p = _mp(out.hidden_states[bsl["b1"][0] : bsl["b1"][1]])
            b2p = F.normalize(b2m, p=2, dim=-1)
            s2p = F.normalize(torch.sqrt(b2v.clamp(min=1e-6)), p=2, dim=-1)
            b3p = _mp(out.hidden_states[bsl["b3"][0] : bsl["b3"][1]])
            b4p = _mp(out.hidden_states[bsl["b4"][0] : bsl["b4"][1]])

        for k, t in [("u", u), ("unit", un), ("b1", b1p), ("b2", b2p), ("b3", b3p), ("b4", b4p), ("s2", s2p)]:
            cols[k].append(t.float().cpu())

    enc = {k: torch.cat(v, dim=0)[inv].numpy().astype(np.float32) for k, v in cols.items()}
    u_np, unit_np, doc_lens = enc["u"], enc["unit"], lens.astype(np.float32)
    u_norm = np.linalg.norm(u_np, axis=1).astype(np.float32)
    all_u_t, all_un_t = torch.from_numpy(u_np), torch.from_numpy(unit_np)

    if abtt_bundle is None:
        imp_locs = np.fromiter((d2l[int(d)] for d in impostor_doc_ids), dtype=np.int32, count=len(impostor_doc_ids))
        abtt_bundle = {"imp_raw_unit": unit_np[imp_locs].copy(), "bands": {}}
        for bk, rk in [("unit", 3), ("b1", 3), ("b2", 3), ("b3", 3), ("b4", 3), ("s2", 2)]:
            mu, tv = fit_abtt(enc[bk][imp_locs], remove_top_k=rk)
            abtt_bundle["bands"][bk] = {"mu": mu, "top_v": tv, "imp_abtt": apply_abtt(enc[bk][imp_locs], mu, tv)}

    abtt_mats = {}
    stats_d = {"unit_raw": impostor_stats(unit_np, abtt_bundle["imp_raw_unit"])}
    for bk, bp in abtt_bundle["bands"].items():
        abtt_mats[bk] = apply_abtt(enc[bk], bp["mu"], bp["top_v"])
        stats_d[bk] = impostor_stats(abtt_mats[bk], bp["imp_abtt"])

    sc_l, cent_l, rl_l, ab_l = [], [], [], []
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
        rc_s2, an_s2, _, _ = _pasn(abtt_mats["s2"], stats_d["s2"])

        mu_u, sd_u, t10_arr, t1_arr = stats_d["unit_raw"]
        t10_min, t10_diff = np.minimum(t10_arr[loc1], t10_arr[loc2]), np.abs(t10_arr[loc1] - t10_arr[loc2])
        t1_m, mu_m, sd_m = 0.5 * (t1_arr[loc1] + t1_arr[loc2]), 0.5 * (mu_u[loc1] + mu_u[loc2]), 0.5 * (sd_u[loc1] + sd_u[loc2])
        cos_infl = rc_u / (np.maximum(t10_u, 0.0) + 0.05)
        gap_b1_b4, gap_b2_b4, gap_ua_u = an_b1 - an_b4, an_b2 - an_b4, rc_ua - rc_u
        style_cons = (an_b1 + an_b2 + an_s2) / 3.0

        n_m, n_diff = 0.5 * (u_norm[loc1] + u_norm[loc2]), np.abs(u_norm[loc1] - u_norm[loc2])
        l2_u = np.linalg.norm(unit_np[loc1] - unit_np[loc2], axis=1).astype(np.float32)
        l1_u = np.sum(np.abs(unit_np[loc1] - unit_np[loc2]), axis=1).astype(np.float32)
        l2_raw = np.linalg.norm(u_np[loc1] - u_np[loc2], axis=1).astype(np.float32)
        l1t, l2t = doc_lens[loc1], doc_lens[loc2]
        len_min, len_sum = np.log1p(np.minimum(l1t, l2t)), np.log1p(l1t + l2t)
        len_rat = np.minimum(l1t, l2t) / np.maximum(np.maximum(l1t, l2t), 1.0)

        sc_m = np.column_stack([
            raw_lg, rc_u, an_u, mar_u, rc_ua, an_ua, mar_ua,
            rc_b1, an_b1, mar_b1, rc_b2, an_b2, mar_b2, rc_b3, an_b3, mar_b3, rc_b4, an_b4, mar_b4, rc_s2, an_s2,
            gap_b1_b4, gap_b2_b4, gap_ua_u, style_cons,
            t10_u, t10_min, t10_diff, t1_m, mu_m, sd_m, cos_infl,
            t10_ua, t10_b1, t10_b2, t10_b4,
            n_m, n_diff, l2_u, l1_u, l2_raw, len_min, len_sum, len_rat,
            raw_lg * t10_u, raw_lg * t10_ua, raw_lg * mu_m, raw_lg * cos_infl,
            raw_lg * n_m, an_ua * t10_ua, an_b2 * t10_b2, style_cons * t10_b1,
        ]).astype(np.float32)
        cent = np.hstack([
            0.5 * (abtt_mats["unit"][loc1] + abtt_mats["unit"][loc2]),
            0.5 * (abtt_mats["b1"][loc1] + abtt_mats["b1"][loc2]),
            0.5 * (abtt_mats["b2"][loc1] + abtt_mats["b2"][loc2]),
        ]).astype(np.float32)
        sc_l.append(sc_m); cent_l.append(cent); rl_l.append(raw_lg); ab_l.append(an_ua)
    return sc_l, cent_l, rl_l, ab_l, abtt_bundle


def compute_anchor_hub_priors(anchor_pairs_df: pd.DataFrame, anchor_logits: np.ndarray) -> dict[tuple[str, str], float]:
    ap = anchor_pairs_df[["cohort_domain", "cand_a", "cand_b"]].copy()
    ap["logit"] = anchor_logits
    both = pd.concat([ap, ap.rename(columns={"cand_a": "cand_b", "cand_b": "cand_a"})], ignore_index=True)
    p_means = both.groupby(["cohort_domain", "cand_a", "cand_b"], as_index=False)["logit"].mean()
    return {
        (dom, cid): float(np.mean(np.sort(grp["logit"].to_numpy())[-5:]))
        for (dom, cid), grp in p_means.groupby(["cohort_domain", "cand_a"], sort=False)
    }


def evaluate_closed_set_1to30(pairs_df: pd.DataFrame, pair_logits: np.ndarray, hub_priors=None, use_csls: bool = True, hub_weight: float = 0.65):
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
'''),
    md_cell(r'''## 5. Backbone Training & Greedy Model Soup

전체 84,860쌍을 `shuffle=True`로 2.5 epoch 학습하고, 후반부 체크포인트 스냅샷(`55%, 70%, 85%, 100%`)과 EMA 가중치를 학습 종료 후 일괄 평가해 Greedy Model Soup을 구성합니다.
'''),
    code_cell(r'''BASE_CKPT = CKPT_DIR / "deberta_v3_base_soup.pt"
seed_everything(SEED)
base_model = SymmetricDebertaVerifier(
    CONFIG["model_name"], use_weighted_layers=True, use_asp=True, use_msd=True, emb_dim=512, grad_ckpt=USE_GRAD_CKPT
).to(DEVICE)

# Export recipe metadata used by the 24-layer Large track
verified_recipe = {
    "verified_on": "deberta_v3_base",
    "use_arch_and_entity": True,
    "use_priorwd_llrd_cosent": False,
    "use_soup_ema_asnorm": True,
    "base_num_layers": 12,
    "large_num_layers": 24,
    "base_top_k_layers": 6,
    "large_top_k_layers": 12,
    "base_llrd_gamma": 0.90,
    "large_llrd_gamma_scaled": round(math.sqrt(0.90), 5),
    "entity_mask_prob": CONFIG["entity_mask_prob"],
    "label_smoothing": CONFIG["label_smoothing"],
    "default_cosent_weight": 0.0,
}
RECIPE_PATH.write_text(json.dumps(verified_recipe, indent=2), encoding="utf-8")

if BASE_CKPT.exists() and (not CONFIG["force_retrain"]):
    base_model.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE, weights_only=True))
    print(f"[ckpt] loaded {BASE_CKPT}")
else:
    t0 = time.perf_counter()
    no_decay = ("bias", "LayerNorm.weight", "layernorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in base_model.named_parameters() if p.requires_grad and n.startswith("backbone.") and not any(k in n for k in no_decay) and p.ndim >= 2], "lr": CONFIG["backbone_lr"], "weight_decay": 0.01},
        {"params": [p for n, p in base_model.named_parameters() if p.requires_grad and n.startswith("backbone.") and (any(k in n for k in no_decay) or p.ndim < 2)], "lr": CONFIG["backbone_lr"], "weight_decay": 0.0},
        {"params": [p for n, p in base_model.named_parameters() if p.requires_grad and not n.startswith("backbone.") and not any(k in n for k in no_decay) and p.ndim >= 2], "lr": CONFIG["head_lr"], "weight_decay": 0.01},
        {"params": [p for n, p in base_model.named_parameters() if p.requires_grad and not n.startswith("backbone.") and (any(k in n for k in no_decay) or p.ndim < 2)], "lr": CONFIG["head_lr"], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]], eps=1e-6)
    ema = ModelEMA(base_model, decay=CONFIG["ema_decay"])
    scaler = torch.amp.GradScaler("cuda", init_scale=2048.0, enabled=(not USE_BF16))

    train_ds = PairDataset(
        train_df["d1"].to_numpy(), train_df["d2"].to_numpy(), train_df["label"].to_numpy(),
        doc_store, is_train=True, entity_mask_prob=CONFIG["entity_mask_prob"], seed=SEED,
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=dynamic_pair_collate, drop_last=True)

    steps_per_ep = max(len(loader) // GRAD_ACCUM, 1)
    total_steps = max(int(steps_per_ep * CONFIG["epochs"]), 20)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps)
    eval_points = sorted({max(int(total_steps * r), 1) for r in (0.55, 0.70, 0.85, 1.00)})

    pending_snaps: dict[str, dict[str, torch.Tensor]] = {}
    opt_step, run_loss = 0, 0.0
    optimizer.zero_grad(set_to_none=True)

    for ep in range(int(math.ceil(CONFIG["epochs"]))):
        train_ds.set_epoch_seed(SEED + ep * 97)
        base_model.train()
        pbar = tqdm(loader, desc=f"ep {ep+1}", leave=False)
        for step_i, batch in enumerate(pbar, start=1):
            ids1, m1, ids2, m2, ys = [t.to(DEVICE, non_blocking=True) for t in batch]
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                logits, _ = base_model(ids1, m1, ids2, m2)
                targets = ys.float() * (1.0 - CONFIG["label_smoothing"]) + 0.5 * CONFIG["label_smoothing"]
                loss = F.binary_cross_entropy_with_logits(logits.float(), targets) / GRAD_ACCUM
            scaler.scale(loss).backward()
            run_loss += loss.item() * GRAD_ACCUM

            if step_i % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(base_model)
                opt_step += 1

                if opt_step % 100 == 0:
                    pbar.set_postfix(loss=f"{run_loss / (100.0 * GRAD_ACCUM):.4f}", step=f"{opt_step}/{total_steps}")
                    run_loss = 0.0
                if opt_step in eval_points:
                    tag = f"step_{opt_step}"
                    pending_snaps[tag] = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
                if opt_step >= total_steps:
                    break
        if opt_step >= total_steps:
            break

    del optimizer
    gc.collect()
    torch.cuda.empty_cache()

    snaps = []
    for tag, sd in pending_snaps.items():
        base_model.load_state_dict(sd)
        auc = float(roc_auc_score(y_vo, expit(fast_infer_logits(base_model, [val_open_df], doc_store, EVAL_BATCH_SIZE)[0])))
        snaps.append((auc, tag, sd))
        print(f"[snap] {tag} val_open_raw_auc={auc:.5f}")

    ema_sd = ema.state_dict_copy()
    base_model.load_state_dict(ema_sd)
    ema_auc = float(roc_auc_score(y_vo, expit(fast_infer_logits(base_model, [val_open_df], doc_store, EVAL_BATCH_SIZE)[0])))
    snaps.append((ema_auc, "ema", ema_sd))
    print(f"[snap] ema val_open_raw_auc={ema_auc:.5f}")

    snaps.sort(key=lambda x: x[0], reverse=True)
    cur_auc, best_tag, best_sd = snaps[0]
    soup_sd = {k: v.float().clone() if v.dtype.is_floating_point else v.clone() for k, v in best_sd.items()}
    members = [best_tag]
    for _, cand_tag, cand_sd in snaps[1:]:
        nm = len(members)
        trial_sd = {k: ((soup_sd[k] * nm + cand_sd[k].float()) / (nm + 1) if soup_sd[k].dtype.is_floating_point else soup_sd[k]) for k in soup_sd}
        base_model.load_state_dict(trial_sd)
        tr_auc = float(roc_auc_score(y_vo, expit(fast_infer_logits(base_model, [val_open_df], doc_store, EVAL_BATCH_SIZE)[0])))
        if tr_auc > cur_auc + 1e-6:
            soup_sd, cur_auc = trial_sd, tr_auc
            members.append(cand_tag)

    base_model.load_state_dict(soup_sd)
    torch.save(base_model.state_dict(), BASE_CKPT)
    print(f"[train] done in {(time.perf_counter()-t0)/60:.1f}m | soup={members} raw_auc={cur_auc:.5f}")
    del snaps, pending_snaps
    gc.collect()
    torch.cuda.empty_cache()
'''),
    md_cell(r'''## 6. 4-Depth All-but-the-Top Neural OOD Calibration

4개 언어 깊이(`L1-3`, `L4-6 mean+std`, `L7-9`, `L11-12`) 은닉 표현에서 256개 학습 사칭자 기준 상위 주성분(`r=3`)을 직교 제거한 뒤, 5-Bucket 양방향 저자 배제 OOF로 `LogisticRegression + 3-Seed ResidualOODGate`를 학습합니다.
'''),
    code_cell(r'''(S_vo, S_vc, S_ah), (C_vo, C_vc, C_ah), (RL_vo, RL_vc, RL_ah), (AB_vo, AB_vc, AB_ah), abtt_bundle = extract_4depth_abtt_features(
    base_model, [val_open_df, val_closed_pairs_df, anchor_hub_pairs_df], doc_store, IMPOSTOR_DOC_IDS, EVAL_BATCH_SIZE
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
gate_folds = []
ik = CONFIG["pca_interact_k"]
lw, mw = CONFIG["gate_lr_w"], 1.0 - CONFIG["gate_lr_w"]

with threadpool_limits(limits=1):
    for k in range(5):
        tr_m, va_m = (vb1 != k) & (vb2 != k), (vb1 == k) | (vb2 == k)
        pca_k = PCA(n_components=CONFIG["pca_components"], random_state=SEED + 40 + k).fit(C_vo[tr_m])
        z_tr, z_va = pca_k.transform(C_vo[tr_m]).astype(np.float32), pca_k.transform(C_vo[va_m]).astype(np.float32)
        F_tr = np.hstack([S_vo[tr_m], z_tr, S_vo[tr_m, 0:1] * z_tr[:, :ik]]).astype(np.float32)
        F_va = np.hstack([S_vo[va_m], z_va, S_vo[va_m, 0:1] * z_va[:, :ik]]).astype(np.float32)

        sc_k = StandardScaler()
        X_tr = np.clip(sc_k.fit_transform(np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
        X_va = np.clip(sc_k.transform(np.nan_to_num(F_va, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
        lr_k = LogisticRegression(C=CONFIG["gate_lr_C"], max_iter=400, solver="lbfgs").fit(X_tr, y_vo[tr_m])
        p_lr_va = lr_k.predict_proba(X_va)[:, 1]

        xt_t, rlt_t, yt_t = torch.from_numpy(X_tr.astype(np.float32)).to(DEVICE), torch.from_numpy(RL_vo[tr_m]).to(DEVICE), torch.from_numpy(y_vo[tr_m].astype(np.float32)).to(DEVICE)
        xv_t, rlv_t = torch.from_numpy(X_va.astype(np.float32)).to(DEVICE), torch.from_numpy(RL_vo[va_m]).to(DEVICE)
        sd_l, pm_l = [], []
        for s_off in CONFIG["gate_mlp_seeds"]:
            seed_everything(SEED + s_off + k)
            m = ResidualOODGate(X_tr.shape[1]).to(DEVICE)
            opt = torch.optim.AdamW(m.parameters(), lr=2.2e-3, weight_decay=0.10)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=240)
            m.train()
            for _ in range(240):
                opt.zero_grad(set_to_none=True)
                F.binary_cross_entropy_with_logits(m(xt_t, rlt_t), yt_t).backward()
                opt.step(); sch.step()
            m.eval()
            with torch.no_grad():
                pm_l.append(torch.sigmoid(m(xv_t, rlv_t)).cpu().numpy())
            sd_l.append({kk: vv.detach().float().cpu().clone() for kk, vv in m.state_dict().items()})

        oof_acc[va_m] += lw * p_lr_va + mw * np.mean(pm_l, axis=0)
        oof_cnt[va_m] += 1.0
        gate_folds.append((pca_k, sc_k, lr_k, sd_l))

val_oof_p = oof_acc / oof_cnt


def predict_ood_gate(S_in: np.ndarray, C_in: np.ndarray, RL_in: np.ndarray) -> np.ndarray:
    preds = []
    rl_t = torch.from_numpy(RL_in.astype(np.float32)).to(DEVICE)
    with threadpool_limits(limits=1):
        for pca_k, sc_k, lr_k, sd_l in gate_folds:
            z = pca_k.transform(C_in).astype(np.float32)
            F_m = np.hstack([S_in, z, S_in[:, 0:1] * z[:, :ik]]).astype(np.float32)
            X_m = np.clip(sc_k.transform(np.nan_to_num(F_m, nan=0.0, posinf=0.0, neginf=0.0)), -8.0, 8.0)
            p_lr = lr_k.predict_proba(X_m)[:, 1]
            xt = torch.from_numpy(X_m.astype(np.float32)).to(DEVICE)
            mp = []
            for sd in sd_l:
                m = ResidualOODGate(X_m.shape[1]).to(DEVICE); m.load_state_dict(sd); m.eval()
                with torch.no_grad():
                    mp.append(torch.sigmoid(m(xt, rl_t)).cpu().numpy())
            preds.append(lw * p_lr + mw * np.mean(mp, axis=0))
    return np.mean(preds, axis=0)


gkf_t = GroupKFold(n_splits=2)
t_grps = val_open_df["author_1"].astype(str).to_numpy()
tau_grid = np.linspace(0.30, 0.70, 161)
f_taus, f_dels = [], []
for c_tr, _ in gkf_t.split(val_open_df, y_vo, groups=t_grps):
    tf = float(tau_grid[int(np.argmax([f1_score(y_vo[c_tr], (val_oof_p[c_tr] >= t).astype(int), average="macro") for t in tau_grid]))])
    db, c1b = 0.0, -1.0
    for dv in np.linspace(0.0, 0.14, 71):
        mv = compute_open_pan_metrics(y_vo[c_tr], val_oof_p[c_tr], tau=tf, delta=float(dv))
        if mv["abstain_pct"] <= CONFIG["max_abstention_pct"] and mv["pan_c_at_1"] > c1b:
            c1b, db = mv["pan_c_at_1"], float(dv)
    f_taus.append(tf); f_dels.append(db)

best_tau, best_delta = float(np.mean(f_taus)), float(np.mean(f_dels))
val_op_f = compute_open_pan_metrics(y_vo, val_oof_p, tau=best_tau, delta=0.0)
val_op_a = compute_open_pan_metrics(y_vo, val_oof_p, tau=best_tau, delta=best_delta)

vc_p = np.clip(predict_ood_gate(S_vc, C_vc, RL_vc), 1e-6, 1.0 - 1e-6)
ah_p = np.clip(predict_ood_gate(S_ah, C_ah, RL_ah), 1e-6, 1.0 - 1e-6)
vc_g, ah_g = np.log(vc_p / (1.0 - vc_p)), np.log(ah_p / (1.0 - ah_p))

best_s1 = {"key": (-1.0, -1.0, -1.0)}
for m_nm, vc_sc, ah_sc in [
    ("raw_logit", RL_vc, RL_ah),
    ("r4_gate_logit", vc_g, ah_g),
    ("blend_70raw_30gate", 0.70 * RL_vc + 0.30 * vc_g, 0.70 * RL_ah + 0.30 * ah_g),
    ("blend_50raw_50gate", 0.50 * RL_vc + 0.50 * vc_g, 0.50 * RL_ah + 0.50 * ah_g),
]:
    hp = compute_anchor_hub_priors(anchor_hub_pairs_df, ah_sc)
    for hw in CONFIG["csls_grid"]:
        cs, cd, cp = evaluate_closed_set_1to30(val_closed_pairs_df, vc_sc, hp, use_csls=(hw > 0), hub_weight=float(hw))
        km = (cs["overall_top1_acc"], cs["overall_macro_f1"], cs["overall_top3_acc"])
        if km > best_s1["key"]:
            best_s1 = {"key": km, "mode": m_nm, "hw": float(hw), "hp": hp, "sum": cs, "dom": cd, "preds": cp}

(OUTPUT_DIR / "val_summary_deberta_base.json").write_text(json.dumps({
    "track": "02_deberta_v3_base",
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
    f"open_auc={val_op_a['roc_auc']:.4f} pan_overall={val_op_a['pan_overall']:.4f} "
    f"forced_f1={val_op_f['forced_macro_f1']:.4f} tau={best_tau:.4f} delta={best_delta:.4f}"
)
display(best_s1["dom"])
'''),
    md_cell("## 7. Locked Test Evaluation & Results Archive"),
    code_cell(r'''if not CONFIG["run_final_test"]:
    print("[test] skipped (set CONFIG['run_final_test'] = True to evaluate locked test split)")
else:
    (S_to, S_tc), (C_to, C_tc), (RL_to, RL_tc), _, _ = extract_4depth_abtt_features(
        base_model, [test_open_df, test_closed_pairs_df], doc_store, IMPOSTOR_DOC_IDS, EVAL_BATCH_SIZE, abtt_bundle=abtt_bundle
    )
    to_p = np.clip(predict_ood_gate(S_to, C_to, RL_to), 1e-6, 1.0 - 1e-6)
    tc_p = np.clip(predict_ood_gate(S_tc, C_tc, RL_tc), 1e-6, 1.0 - 1e-6)
    tc_g = np.log(tc_p / (1.0 - tc_p))

    s1_mode, hw = best_s1["mode"], best_s1["hw"]
    if s1_mode == "raw_logit":
        tc_score = RL_tc
    elif s1_mode == "r4_gate_logit":
        tc_score = tc_g
    elif s1_mode == "blend_70raw_30gate":
        tc_score = 0.70 * RL_tc + 0.30 * tc_g
    else:
        tc_score = 0.50 * RL_tc + 0.50 * tc_g

    test_cl_sum, test_cl_dom, test_cl_preds = evaluate_closed_set_1to30(test_closed_pairs_df, tc_score, best_s1["hp"], use_csls=(hw > 0), hub_weight=hw)
    test_op_f = compute_open_pan_metrics(y_to, to_p, tau=best_tau, delta=0.0)
    test_op_a = compute_open_pan_metrics(y_to, to_p, tau=best_tau, delta=best_delta)

    # Symmetry & latency measurement
    sub_ab = test_open_df.iloc[:32].copy().reset_index(drop=True)
    sub_ba = sub_ab.rename(columns={"d1": "d2", "d2": "d1", "text_1": "text_2", "text_2": "text_1"})
    (sab, sba), (cab, cba), (lab, lba), _, _ = extract_4depth_abtt_features(base_model, [sub_ab, sub_ba], doc_store, IMPOSTOR_DOC_IDS, EVAL_BATCH_SIZE, abtt_bundle=abtt_bundle)
    sym_err = float(np.max(np.abs(predict_ood_gate(sab, cab, lab) - predict_ood_gate(sba, cba, lba))))

    torch.cuda.synchronize()
    t_lat = time.perf_counter()
    sl, cl, ll, _, _ = extract_4depth_abtt_features(base_model, [test_open_df.iloc[:64].copy().reset_index(drop=True)], doc_store, IMPOSTOR_DOC_IDS, EVAL_BATCH_SIZE, abtt_bundle=abtt_bundle)
    _ = predict_ood_gate(sl[0], cl[0], ll[0])
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

    test_cl_dom.to_csv(OUTPUT_DIR / "final_test_closed_by_domain_base.csv", index=False)
    test_cl_preds.to_csv(OUTPUT_DIR / "test_closed_predictions_deberta_base.csv", index=False)
    st_test_df.to_csv(OUTPUT_DIR / "final_test_open_by_stratum_base.csv", index=False)
    dom_test_df.to_csv(OUTPUT_DIR / "final_test_open_by_domain_base.csv", index=False)
    len_test_df.to_csv(OUTPUT_DIR / "final_test_open_by_length_base.csv", index=False)

    to_pred_bin = (to_p >= best_tau).astype(np.int32)
    to_exp = test_open_df[["pair_id", "eval_stratum", "domain", "topic_1", "topic_2", "is_cross_topic", "is_cross_genre", "label"]].copy()
    to_exp["deberta_base_prob"] = np.round(to_p, 6)
    to_exp["deberta_base_pred"] = to_pred_bin
    to_exp["is_abstained"] = ((to_p > best_tau - best_delta) & (to_p < best_tau + best_delta)).astype(np.int32)
    to_exp["is_correct"] = (to_pred_bin == y_to).astype(np.int32)
    to_exp.to_csv(OUTPUT_DIR / "test_open_predictions_deberta_base.csv", index=False)

    (OUTPUT_DIR / "final_locked_test_base_summary.json").write_text(json.dumps({
        "track": "02_deberta_v3_base",
        "closed_set_matching": {"mode": s1_mode, "csls_hub_weight": hw},
        "open_thresholds": {"tau": round(best_tau, 4), "delta": round(best_delta, 4)},
        "system_metrics": {
            "gpu_device": GPU_NAME, "gpu_latency_ms_per_pair": round(ms_per_pair, 2),
            "symmetry_max_error": float(f"{sym_err:.2e}"),
            "num_parameters_M": round(sum(p.numel() for p in base_model.parameters()) / 1e6, 2),
        },
        "scenario_1_closed_set_test": test_cl_sum,
        "scenario_2_open_cross_test_forced": test_op_f,
        "scenario_2_open_cross_test_abstention": test_op_a,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[test] closed_top1={test_cl_sum['overall_top1_acc']:.4f} closed_top3={test_cl_sum['overall_top3_acc']:.4f} | "
        f"open_auc={test_op_a['roc_auc']:.4f} pan_overall={test_op_a['pan_overall']:.4f} "
        f"sym_err={sym_err:.2e} latency={ms_per_pair:.1f}ms/pair"
    )
    display(test_cl_dom)
    display(dom_test_df)

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
    plt.savefig(FIG_DIR / "final_test_02_deberta_base_4panel.png", bbox_inches="tight")
    plt.show()

bundle_out = Path.cwd() / "results_02_deberta_v3_base.zip"
with zipfile.ZipFile(bundle_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    if RECIPE_PATH.exists():
        zf.write(RECIPE_PATH, arcname="verified_dl_recipe.json")
    for fp in sorted(OUTPUT_DIR.rglob("*")):
        if fp.is_file() and fp.suffix not in (".pt", ".npz") and "checkpoints" not in fp.parts:
            zf.write(fp, arcname=str(fp.relative_to(ROOT_OUT)))
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
    build_notebook(Path(r"D:\Projects\NLU-Authorship-Verification\notebooks\colab\train_02_deberta_v3_base.ipynb"))
