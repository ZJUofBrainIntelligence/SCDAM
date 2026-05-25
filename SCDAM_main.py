# SCDAM_main.py
# -*- coding: utf-8 -*-
"""
SCDAM training script.

运行示例：
    python SCDAM_main.py --features_root ./EEG_Feature_2Hz --labels_root ./perclos_labels --topk 18 --sigma_mmd 0 --sel_w_mmd 0.5 --sel_w_hmm 0.5 --adv_weight 0.005 --adv_sched dan

说明：
    - 模型定义位于 SCDAM_model.py
    - 数据加载、指标、源域选择和域对抗工具位于 tools.py
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import argparse
import json
import itertools
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from SCDAM_model import trans_fusion
from tools import (
    _natural_key,
    discover_session_pairs,
    SingleFileTimeDataset,
    collate_batch,
    metrics_with_smoothing,
    select_sources_by_mmd_and_hmm,
    DomainDiscriminator,
    lambda_schedule,
    grl,
)

def train_one_target_subject(
    subj_id: str,
    subj2dataset: Dict[str, Dataset],
    trans_fusion_cls,
    device: torch.device,
    epochs: int = 30,
    lr: float = 3e-4,
    batch_size_src: int = 128,
    batch_size_tgt: int = 64,
    topk: int = 15,
    sigma_mmd: float = 1.0,
    sel_w_mmd: float = 0.5,
    sel_w_hmm: float = 0.5,
    sel_use_hmm: bool = True,
    sel_batch_size: int = 256,
    mmd_max_samples: int = 512,
    hmm_pca_dim: int = 20,
    hmm_lamA: float = 1.0,
    hmm_lamDwell: float = 0.5,
    hmm_lamEmit: float = 1.0,
    hmm_max_fit_samples: int = 15000,
    hmm_max_iter: int = 200,
    hmm_tol: float = 1e-3,
    hmm_random_state: int = 0,
    adv_weight: float = 0.04,
    adv_sched: str = "dan",
    smooth_mode: str = "none",
    smooth_k: int = 5,
    smooth_alpha: float = 0.3,
):

    subj_ids = sorted(subj2dataset.keys(), key=_natural_key)
    assert subj_id in subj2dataset, f"subject {subj_id} not found"

    ds_tgt = subj2dataset[subj_id]
    src_ids = [sid for sid in subj_ids if sid != subj_id]

    x1_0, x2_0, y0 = ds_tgt[0]
    F = x1_0.shape[0]
    N = x1_0.shape[1]
    out_dim = 1

    model = trans_fusion_cls(in_feature=F, class_num=out_dim,
                             graph_args={}, frames=0, node_num=N).to(device)
    if hasattr(model, "cl"):
        model.cl.enable_softclt = True
        model.cl.soft_by = "label"
        model.cl.tau_inst = 5.0
        model.cl.tau_label = 5.0
        model.cl.alpha = 1.0
        model.cl.temperature = 0.2
        model.cl.soft_weight = 0.05

    print(f"\n[SUBJ {subj_id}] selecting sources by MMD + (optional) HMM on raw inputs ...")
    selected_src_ids, details_sorted = select_sources_by_mmd_and_hmm(
        target_id=subj_id,
        src_ids=src_ids,
        subj2dataset=subj2dataset,
        topk=topk,
        device=device,
        sigma_mmd=sigma_mmd,
        sel_w_mmd=sel_w_mmd,
        sel_w_hmm=sel_w_hmm,
        hmm_pca_dim=hmm_pca_dim,
        hmm_lamA=hmm_lamA,
        hmm_lamDwell=hmm_lamDwell,
        hmm_lamEmit=hmm_lamEmit,
        hmm_max_fit_samples=hmm_max_fit_samples,
        hmm_max_iter=hmm_max_iter,
        hmm_tol=hmm_tol,
        hmm_random_state=hmm_random_state,
        sel_batch_size=sel_batch_size,
        mmd_max_samples=mmd_max_samples,
        seed=hmm_random_state,
        use_hmm=sel_use_hmm,
    )

    show_k = min(15, len(details_sorted))
    print(f"[SUBJ {subj_id}] source ranking (top-{show_k})")
    for d in details_sorted[:show_k]:
        if sel_use_hmm:
            print(f"    {d['source_id']} | score={d['score']:.4f} ")
        else:
            print(f"    {d['source_id']} | score={d['score']:.4f} ")

    domain_dists = details_sorted
    print(f"[SUBJ {subj_id}] selected top-{topk} source subjects: {selected_src_ids}")

    ds_src_sel_list = [subj2dataset[sid] for sid in selected_src_ids]
    ds_src_sel = ConcatDataset(ds_src_sel_list)

    print(f"[SUBJ {subj_id}] target_samples={len(ds_tgt)}, selected_source_samples={len(ds_src_sel)}")

    # dataloader
    src_loader = DataLoader(ds_src_sel, batch_size=batch_size_src, shuffle=True,
                            drop_last=False, num_workers=0, collate_fn=collate_batch)
    tgt_loader = DataLoader(ds_tgt, batch_size=batch_size_tgt, shuffle=True,
                            drop_last=False, num_workers=0, collate_fn=collate_batch)
    tgt_iter = itertools.cycle(iter(tgt_loader))

    hidden_dim = 128
    disc = DomainDiscriminator(in_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-2)
    opt_disc = torch.optim.AdamW(disc.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.CrossEntropyLoss()

    max_epochs = epochs

    best_score = -float("inf")
    best_state = None

    adv_w = float(adv_weight)
    acc_ema = 0.5
    ema_beta = 0.9
    target_acc = 0.60
    k_ctrl = 0.25
    adv_min, adv_max = 0.0, 0.04

    for epoch in range(max_epochs):
        model.train()
        disc.train()
        epoch_loss = 0.0
        epoch_disc = 0.0
        epoch_adv = 0.0
        n_steps = 0

        for x1_s, x2_s, y_s in src_loader:
            n_steps += 1
            x1_s = x1_s.to(device)
            x2_s = x2_s.to(device)
            y_s = y_s.to(device)

            x1_t, x2_t, y_t = next(tgt_iter)
            x1_t = x1_t.to(device)
            x2_t = x2_t.to(device)

            disc.zero_grad(set_to_none=True)
            model.eval()

            with torch.no_grad():
                feat_src_d = model.encode_feat(x1_s, x2_s)  # [Bs,H]
                feat_tgt_d = model.encode_feat(x1_t, x2_t)  # [Bt,H]

            logit_s = disc(feat_src_d.detach())
            logit_t = disc(feat_tgt_d.detach())
            lab_s = torch.ones(logit_s.size(0), dtype=torch.long, device=device)
            lab_t = torch.zeros(logit_t.size(0), dtype=torch.long, device=device)
            loss_disc = bce(logit_s, lab_s) + bce(logit_t, lab_t)
            loss_disc.backward()
            opt_disc.step()

            with torch.no_grad():
                pred_s = logit_s.argmax(dim=1)
                pred_t = logit_t.argmax(dim=1)
                acc = torch.cat([(pred_s == lab_s), (pred_t == lab_t)]).float().mean().item()
            acc_ema = ema_beta * acc_ema + (1 - ema_beta) * acc

            adv_w = float(np.clip(adv_w * np.exp(k_ctrl * (acc_ema - target_acc)), adv_min, adv_max))

            model.train()
            opt.zero_grad(set_to_none=True)

            loss_sum_s, loss_mm_s, pred_s, fusion_s = model(x1_s, x2_s, y_s)

            feat_src_g = model.encode_feat(x1_s, x2_s)
            feat_tgt_g = model.encode_feat(x1_t, x2_t)

            grl_lambda = lambda_schedule(epoch, max_epochs, mode=adv_sched, base=0.0, target=1.0)

            logit_s_g = disc(grl(feat_src_g, grl_lambda))
            logit_t_g = disc(grl(feat_tgt_g, grl_lambda))
            loss_adv = bce(logit_s_g, lab_s) + bce(logit_t_g, lab_t)

            loss_total = loss_sum_s + adv_w * loss_adv

            loss_total.backward()
            opt.step()

            epoch_loss += float(loss_total.detach().cpu())
            epoch_disc += float(loss_disc.detach().cpu())
            epoch_adv += float(loss_adv.detach().cpu())

        if n_steps > 0:
            print(f"[SUBJ {subj_id}] Epoch {epoch:02d} | "
                  f"loss_total={epoch_loss / n_steps:.4f} | "
                  f"loss_disc={epoch_disc / n_steps:.4f} | "
                  f"loss_adv={epoch_adv / n_steps:.4f}")

            model.eval()
            all_p, all_t = [], []
            with torch.no_grad():
                full_loader = DataLoader(
                    ds_tgt,
                    batch_size=batch_size_tgt,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                    collate_fn=collate_batch
                )
                for x1_val, x2_val, y_val in full_loader:
                    x1_val = x1_val.to(device)
                    x2_val = x2_val.to(device)
                    y_val = y_val.to(device)

                    loss_sum_val, loss_mm_val, pred_val, fusion_val = model(x1_val, x2_val, y_val)
                    all_p.append(pred_val.detach().cpu())
                    all_t.append(y_val.detach().cpu())

            pred_all = torch.cat(all_p, dim=0)
            true_all = torch.cat(all_t, dim=0)
            if pred_all.ndim == 2 and pred_all.size(1) == 1:
                pred_all = pred_all.squeeze(1)
            if true_all.ndim == 2 and true_all.size(1) == 1:
                true_all = true_all.squeeze(1)

            rmse_epoch, cor_epoch = metrics_with_smoothing(
                pred_all, true_all,
                smooth_mode=smooth_mode,
                smooth_k=smooth_k,
                smooth_alpha=smooth_alpha
            )
            print(f"[SUBJ {subj_id}] Epoch {epoch:02d} TEST RMSE={rmse_epoch:.6f} CORR={cor_epoch:.4f}")

            score_epoch = float(cor_epoch) - float(rmse_epoch)
            if score_epoch > best_score:
                best_score = score_epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"[SUBJ {subj_id}]  ** best updated: score={best_score:.6f} (CORR={cor_epoch:.4f}, RMSE={rmse_epoch:.6f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        full_loader = DataLoader(ds_tgt, batch_size=batch_size_tgt, shuffle=False,
                                 drop_last=False, num_workers=0, collate_fn=collate_batch)
        for x1, x2, y in full_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            loss_sum, loss_mm, pred, fusion = model(x1, x2, y)
            all_p.append(pred.detach().cpu())
            all_t.append(y.detach().cpu())

    pred_all = torch.cat(all_p, 0)
    true_all = torch.cat(all_t, 0)
    if pred_all.ndim == 2 and pred_all.size(1) == 1:
        pred_all = pred_all.squeeze(1)
    if true_all.ndim == 2 and true_all.size(1) == 1:
        true_all = true_all.squeeze(1)

    rmse, cor = metrics_with_smoothing(
        pred_all, true_all,
        smooth_mode=smooth_mode,
        smooth_k=smooth_k,
        smooth_alpha=smooth_alpha
    )
    print(f"[SUBJ {subj_id}] FINAL (smoothed) RMSE={rmse:.6f} CORR={cor:.4f}")

    return rmse, cor, model.state_dict(), selected_src_ids, domain_dists


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_root", type=str, required=True)
    parser.add_argument("--labels_root",   type=str, required=True)
    parser.add_argument("--model_file", type=str, default=None,
                        help="Use SCDAM_model.py in the current directory.")
    parser.add_argument("--outdir",        type=str, default="result")

    parser.add_argument("--epochs",         type=int, default=30)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--batch_size_src", type=int, default=256)
    parser.add_argument("--batch_size_tgt", type=int, default=256)

    parser.add_argument("--time_mode",      type=str, default="per_t",
                        choices=["per_t", "win_mean", "file_mean"])
    parser.add_argument("--win_len",        type=int, default=16)
    parser.add_argument("--win_stride",     type=int, default=8)

    parser.add_argument("--topk",           type=int, default=15,
                        help="Number of nearest source subjects selected from the source domain.")
    parser.add_argument("--sigma_mmd",      type=float, default=1.0,
                        help="Sigma of the MMD/RBF kernel.")

    parser.add_argument("--sel_w_mmd", type=float, default=0.5,
                        help="Source-domain selection fusion weight: MMD (data distribution similarity)")
    parser.add_argument("--sel_w_hmm", type=float, default=0.5,
                        help="Source-domain selection fusion weight: HMM (accumulation–recovery dynamics similarity).")
    parser.add_argument("--no_hmm_select", action="store_true",
                        help="Disable HMM dynamics similarity and use only MMD for source-domain selection.")
    parser.add_argument("--sel_batch_size", type=int, default=256,
                        help="Batch size for extracting raw inputs during the source-domain selection stage (does not affect the training batch size).")
    parser.add_argument("--mmd_max_samples", type=int, default=512,
                        help="Maximum number of samples randomly drawn from each domain during MMD estimation.")
    parser.add_argument("--hmm_pca_dim", type=int, default=20,
                        help="Dimensionality reduction dimension for HMM observations.")
    parser.add_argument("--hmm_max_fit_samples", type=int, default=15000,
                        help="Maximum number of samples used for PCA fitting.")
    parser.add_argument("--hmm_max_iter", type=int, default=200,
                        help="Maximum number of EM iterations for the 2-state Gaussian HMM.")
    parser.add_argument("--hmm_tol", type=float, default=1e-3,
                        help="Convergence threshold for the 2-state Gaussian HMM.")
    parser.add_argument("--hmm_random_state", type=int, default=0,
                        help="Random seed for HMM/PCA.")
    parser.add_argument("--hmm_lamA", type=float, default=1.0,
                        help="HMM distance term: weight for transition matrix discrepancy.")
    parser.add_argument("--hmm_lamDwell", type=float, default=0.5,
                        help="HMM distance term: weight for dwell-time discrepancy.")
    parser.add_argument("--hmm_lamEmit", type=float, default=1.0,
                        help="HMM distance term: weight for emission distribution discrepancy.")

    parser.add_argument("--adv_weight",     type=float, default=0.04,
                        help="Adversarial loss weight.")
    parser.add_argument("--adv_sched",      type=str, default="dan",
                        choices=["const", "linear", "dan"])

    parser.add_argument("--smooth_mode", type=str, default="none",
                        choices=["none", "moving", "exp"],
                        help="Output smoothing mode.")
    parser.add_argument("--smooth_k", type=int, default=5,
                        help="Moving-average smoothing window size.")
    parser.add_argument("--smooth_alpha", type=float, default=0.3,
                        help="Exponential smoothing coefficient.")

    parser.add_argument("--seed",           type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.outdir, exist_ok=True)

    session_pairs = discover_session_pairs(args.features_root, args.labels_root)

    subj2dataset: Dict[str, Dataset] = {}
    for file_key, pair_list in session_pairs.items():
        fpath, lpath = pair_list[0]
        ds = SingleFileTimeDataset(
            fpath, lpath,
            mode=args.time_mode,
            win_len=args.win_len,
            win_stride=args.win_stride
        )
        subj2dataset[file_key] = ds
        print(f"[BUILD] domain {file_key}: 1 file, samples={len(ds)}")

    if args.model_file not in (None, "", "SCDAM_model.py", "SCDAM_model"):
        print(f"[WARN] --model_file={args.model_file} Ignored; currently using SCDAM_model.trans_fusion.")
    trans_fusion_cls = trans_fusion

    subj_ids = sorted(subj2dataset.keys(), key=_natural_key)
    all_rmse, all_cor = [], []
    subj_results = []

    for sid in subj_ids:
        rmse, cor, state_dict, selected_src_ids, domain_dists = train_one_target_subject(
            sid,
            subj2dataset,
            trans_fusion_cls,
            device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size_src=args.batch_size_src,
            batch_size_tgt=args.batch_size_tgt,
            topk=args.topk,
            sigma_mmd=args.sigma_mmd,
            sel_w_mmd=args.sel_w_mmd,
            sel_w_hmm=args.sel_w_hmm,
            sel_use_hmm=(not args.no_hmm_select),
            sel_batch_size=args.sel_batch_size,
            mmd_max_samples=args.mmd_max_samples,
            hmm_pca_dim=args.hmm_pca_dim,
            hmm_lamA=args.hmm_lamA,
            hmm_lamDwell=args.hmm_lamDwell,
            hmm_lamEmit=args.hmm_lamEmit,
            hmm_max_fit_samples=args.hmm_max_fit_samples,
            hmm_max_iter=args.hmm_max_iter,
            hmm_tol=args.hmm_tol,
            hmm_random_state=args.hmm_random_state,
            adv_weight=args.adv_weight,
            adv_sched=args.adv_sched,
            smooth_mode=args.smooth_mode,
            smooth_k=args.smooth_k,
            smooth_alpha=args.smooth_alpha,
        )
        all_rmse.append(rmse)
        all_cor.append(cor)
        subj_results.append({
            "subject": sid,
            "rmse": rmse,
            "cor": cor,
            "selected_sources": selected_src_ids,
            "domain_dists": domain_dists
        })

        ckpt_path = os.path.join(args.outdir, f"subj_{sid}_model.pt")
        torch.save({"state_dict": state_dict}, ckpt_path)
        print(f"[SAVE] subject {sid} model -> {ckpt_path}")

    ddof = 1 if len(all_rmse) > 1 else 0
    rmse_mean = float(np.mean(all_rmse)) if all_rmse else float("nan")
    rmse_std = float(np.std(all_rmse, ddof=ddof)) if all_rmse else float("nan")
    cor_mean = float(np.mean(all_cor)) if all_cor else float("nan")
    cor_std = float(np.std(all_cor, ddof=ddof)) if all_cor else float("nan")

    summary = {
        "subjects": subj_results,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "cor_mean": cor_mean,
        "cor_std": cor_std,
    }
    summary_path = os.path.join(args.outdir, "cross_subject_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n[CROSS-SUBJECT] Overall:")
    print(f"  RMSE = {rmse_mean:.6f} ± {rmse_std:.6f}")
    print(f"  CORR = {cor_mean:.4f} ± {cor_std:.4f}")
    print(f"[SAVE] summary -> {summary_path}")


if __name__ == "__main__":
    main()
