"""Standalone inference engine for Track 03: DeBERTa-v3-large + Dual-Rank ABTT + ResidualOODGate."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import expit
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nlu_authorship.artifacts import find_artifact, find_local_tokenizer


class WeightedTopLayersPooling(nn.Module):
    def __init__(self, num_total_layers: int, top_k_layers: int):
        super().__init__()
        self.num_total_layers = num_total_layers
        self.top_k = top_k_layers
        self.layer_weights = nn.Parameter(torch.zeros(top_k_layers))

    def forward(self, all_hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        selected = torch.stack(all_hidden_states[-self.top_k :], dim=0)
        w = F.softmax(self.layer_weights.float(), dim=0).to(selected.dtype).view(-1, 1, 1, 1)
        return torch.sum(selected * w, dim=0)


class AttentiveStatsPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = hidden.float()
        scores = self.attn(hidden).squeeze(-1).float().masked_fill(~mask.bool(), -1e9)
        alpha = F.softmax(scores, dim=1).unsqueeze(-1)
        mean = torch.sum(alpha * h, dim=1)
        var = torch.sum(alpha * ((h - mean.unsqueeze(1)) ** 2), dim=1).clamp(min=1e-6)
        return torch.cat([mean, torch.sqrt(var)], dim=-1)


class SymmetricDebertaVerifier(nn.Module):
    def __init__(self, config, use_weighted_layers: bool = True, use_asp_pooling: bool = True, use_msd_head: bool = True, emb_dim: int = 512):
        super().__init__()
        self.config = config
        self.config.output_hidden_states = True
        self.backbone = AutoModel.from_config(self.config).float()
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


def _apply_abtt(arr: np.ndarray, mu: np.ndarray, top_v: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32) - mu
    x = x - (x @ top_v.T) @ top_v
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def _impostor_stats(mat: np.ndarray, imp_mat: np.ndarray):
    sim = mat @ imp_mat.T
    ms = np.where(sim > 0.9995, np.nan, sim)
    mu = np.nanmean(ms, axis=1).astype(np.float32)
    sd = np.maximum(np.nanstd(ms, axis=1), 1e-4).astype(np.float32)
    sr = np.sort(np.where(np.isnan(ms), -1.0, ms), axis=1)
    return mu, sd, sr[:, -10:].mean(axis=1).astype(np.float32), sr[:, -1].astype(np.float32)


class DebertaLargeEngine:
    """Loads pure-tensor model_03_deberta_v3_large.pt and runs symmetric pair verification."""

    def __init__(self, checkpoint_path: Path | None = None, tokenizer_dir: Path | None = None, device: str | None = None):
        ckpt = checkpoint_path or find_artifact("dl")
        if ckpt is None or not ckpt.is_file():
            raise FileNotFoundError("model_03_deberta_v3_large.pt not found. Run `python verify_author.py --download dl` first.")
        self.checkpoint_path = ckpt
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.device_label = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"

        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.model_name = str(payload["model_name"])
        self.max_doc_tokens = int(payload["max_doc_tokens"])
        self.tau = float(payload["tau"])
        self.delta = float(payload["delta"])
        self.band_slices = {k: tuple(v) for k, v in payload["band_slices"].items()}

        tok_src = tokenizer_dir or find_local_tokenizer() or self.model_name
        cfg = AutoConfig.from_pretrained(tok_src)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)

        self.model = SymmetricDebertaVerifier(cfg, **payload["model_init_kwargs"]).to(self.device)
        sd_fp32 = {k: (v.float() if v.dtype.is_floating_point else v) for k, v in payload["model_state_dict_fp16"].items()}
        self.model.load_state_dict(sd_fp32, strict=True)
        self.model.eval()

        ab = payload["abtt_bundle"]
        self.imp_raw_unit = ab["imp_raw_unit"].numpy().astype(np.float32)
        self.abtt_bands = {
            bk: {
                "src": str(bp["src"]),
                "mu": bp["mu"].numpy().astype(np.float32),
                "top_v": bp["top_v"].numpy().astype(np.float32),
                "imp_abtt": bp["imp_abtt"].numpy().astype(np.float32),
            }
            for bk, bp in ab["bands"].items()
        }

        # Pre-instantiate 5x5 residual gate MLPs on target device for low-latency serving
        self.gate_folds = []
        for fd in payload["gate_folds"]:
            mlps = []
            for sd_m in fd["mlp_state_dicts"]:
                m = ResidualOODGate(int(fd["mlp_in_dim"])).to(self.device)
                m.load_state_dict(sd_m)
                m.eval()
                mlps.append(m)
            sc_s = fd["scaler_scale"].numpy().astype(np.float32)
            self.gate_folds.append({
                "pca_dom_mean": fd["pca_dom_mean"].numpy().astype(np.float32),
                "pca_dom_comp_T": fd["pca_dom_components"].numpy().astype(np.float32).T,
                "pca_agr_mean": fd["pca_agr_mean"].numpy().astype(np.float32),
                "pca_agr_comp_T": fd["pca_agr_components"].numpy().astype(np.float32).T,
                "scaler_mean": fd["scaler_mean"].numpy().astype(np.float32),
                "scaler_scale": np.where(sc_s == 0.0, 1.0, sc_s).astype(np.float32),
                "lr_coef": fd["lr_coef"].numpy().astype(np.float32),
                "lr_intercept": float(fd["lr_intercept"]),
                "lr_weight": float(fd["lr_weight"]),
                "mlp_weight": float(fd["mlp_weight"]),
                "mlps": mlps,
            })

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> dict[str, np.ndarray]:
        enc_tok = self.tokenizer(texts, max_length=self.max_doc_tokens, truncation=True, padding=True, return_tensors="pt")
        ids_g = enc_tok["input_ids"].to(self.device)
        msk_g = enc_tok["attention_mask"].to(self.device)
        tok_lens = msk_g.sum(dim=1).float().cpu().numpy().astype(np.float32)

        use_amp = self.device.type == "cuda"
        amp_dt = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.amp.autocast("cuda", dtype=amp_dt, enabled=use_amp):
            out = self.model.backbone(input_ids=ids_g, attention_mask=msk_g, output_hidden_states=True)
            seq_h = self.model.layer_pool(out.hidden_states) if self.model.use_weighted_layers else out.last_hidden_state
            mf = msk_g.unsqueeze(-1).float()
            denom = torch.clamp(mf.sum(dim=1), min=1.0)
            pooled = self.model.token_pool(seq_h, msk_g) if self.model.use_asp_pooling else (torch.sum(seq_h.float() * mf, dim=1) / denom)
            u = self.model.proj(pooled.to(self.model.proj[0].weight.dtype))
            un = F.normalize(u.float(), p=2, dim=-1)

            def _ms(hs_slice):
                h = torch.stack(hs_slice, dim=0).float().mean(dim=0)
                mu = torch.sum(h * mf, dim=1) / denom
                vr = torch.sum(((h - mu.unsqueeze(1)) ** 2) * mf, dim=1) / denom
                return F.normalize(mu, p=2, dim=-1), F.normalize(torch.sqrt(vr.clamp(min=1e-6)), p=2, dim=-1)

            bsl = self.band_slices
            b1, s1 = _ms(out.hidden_states[bsl["b1"][0] : bsl["b1"][1]])
            b2, s2 = _ms(out.hidden_states[bsl["b2"][0] : bsl["b2"][1]])
            b3, s3 = _ms(out.hidden_states[bsl["b3"][0] : bsl["b3"][1]])
            h4 = torch.stack(out.hidden_states[bsl["b4"][0] : bsl["b4"][1]], dim=0).float().mean(dim=0)
            b4 = F.normalize(torch.sum(h4 * mf, dim=1) / denom, p=2, dim=-1)

        return {
            "u": u.float().cpu().numpy().astype(np.float32),
            "unit": un.float().cpu().numpy().astype(np.float32),
            "b1": b1.float().cpu().numpy().astype(np.float32),
            "s1": s1.float().cpu().numpy().astype(np.float32),
            "b2": b2.float().cpu().numpy().astype(np.float32),
            "s2": s2.float().cpu().numpy().astype(np.float32),
            "b3": b3.float().cpu().numpy().astype(np.float32),
            "s3": s3.float().cpu().numpy().astype(np.float32),
            "b4": b4.float().cpu().numpy().astype(np.float32),
            "tok_len": tok_lens,
        }

    @torch.no_grad()
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
            "model_id": "dl",
            "model_name": "03. DeBERTa-v3-large (Dual ABTT + Soup)",
            "probability": round(float(prob), 6),
            "verdict": verdict,
            "forced_verdict": "same_author" if prob >= self.tau else "different_author",
            "tau": round(self.tau, 4),
            "delta": round(self.delta, 4),
            "latency_ms": round(latency_ms, 2),
            "symmetry_error": float(f"{sym_err:.2e}"),
            "device": self.device_label,
            "signals": signals,
        }

    @torch.no_grad()
    def _score_single(self, text_1: str, text_2: str) -> tuple[float, dict[str, Any]]:
        enc = self._encode_texts([text_1, text_2])
        u_np, unit_np, doc_lens = enc["u"], enc["unit"], enc["tok_len"]
        u_norm = np.linalg.norm(u_np, axis=1).astype(np.float32)

        abtt_mats = {}
        stats_d = {"unit_raw": _impostor_stats(unit_np, self.imp_raw_unit)}
        for bk, bp in self.abtt_bands.items():
            abtt_mats[bk] = _apply_abtt(enc[bp["src"]], bp["mu"], bp["top_v"])
            stats_d[bk] = _impostor_stats(abtt_mats[bk], bp["imp_abtt"])

        u_t = torch.from_numpy(u_np).to(self.device)
        un_t = torch.from_numpy(unit_np).to(self.device)
        raw_lg_t, _ = self.model.classify_pair_embeddings(u_t[0:1], un_t[0:1], u_t[1:2], un_t[1:2])
        raw_lg = raw_lg_t.float().cpu().numpy().astype(np.float32)

        loc1, loc2 = np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)

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

        preds = []
        rl_t = torch.from_numpy(raw_lg).to(self.device)
        rl_col = sc_m[:, 0:1]
        for fd in self.gate_folds:
            zd = (c_dom - fd["pca_dom_mean"]) @ fd["pca_dom_comp_T"]
            za = (c_agr - fd["pca_agr_mean"]) @ fd["pca_agr_comp_T"]
            Fm = np.nan_to_num(np.hstack([sc_m, zd, rl_col * zd[:, :10], za]), nan=0.0, posinf=0.0, neginf=0.0)
            Xm = np.clip((Fm - fd["scaler_mean"]) / fd["scaler_scale"], -8.0, 8.0)
            p_lr = expit(Xm @ fd["lr_coef"] + fd["lr_intercept"])
            xt = torch.from_numpy(Xm.astype(np.float32)).to(self.device)
            mp = [torch.sigmoid(m(xt, rl_t)).cpu().numpy() for m in fd["mlps"]]
            preds.append(fd["lr_weight"] * p_lr + fd["mlp_weight"] * np.mean(mp, axis=0))

        prob = float(np.clip(np.mean(preds), 1e-6, 1.0 - 1e-6))
        signals = {
            "tokens_a": int(doc_lens[0]),
            "tokens_b": int(doc_lens[1]),
            "raw_backbone_prob": round(float(expit(raw_lg[0])), 4),
            "band1_lexical_sim": round(float(rc_b1[0]), 4),
            "band2_syntax_sim": round(float(rc_b2[0]), 4),
            "band3_discourse_sim": round(float(rc_b3[0]), 4),
            "band4_semantic_sim": round(float(rc_b4[0]), 4),
            "abtt_r8_sim": round(float(rc_u8[0]), 4),
            "impostor_top10_affinity": round(float(t10_u[0]), 4),
        }
        return prob, signals
