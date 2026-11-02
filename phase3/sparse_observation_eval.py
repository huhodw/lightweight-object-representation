"""
sparse_observation_eval.py

Phase 3 の SAVi モデルで「画像を k フレームに 1 回しか見せない」推論を行い、
画像を見せないフレームで、スロットをどう運ぶかを 2 通り比べる。
再訓練はしない。保存済みチェックポイントを読み込んで推論するだけ。

  k : 何フレームに 1 回画像を見せるか。フレーム t は t % k == 0 のときだけ観測する。
      k=1 は毎フレーム観測で、元のモデルとまったく同じ処理になる。
  見せないフレームでの運び方:
      predictor : 前のスロットを予測器に通したものを、そのフレームのスロットにする
      copy      : 前のスロットをそのまま使う（何も予測しない基準）

画像を見せるフレームの処理は元のモデルと同じ（フレーム1以外は 予測器 → 補正）。
画像を見せないフレームでは、エンコーダも補正も呼ばない。

置き場所: savi_phase3 フォルダの直下（diagnose_predictor.py と同じ場所。そこから部品を読み込む）
実行例:
  python sparse_observation_eval.py --num-videos 3    # まず3本で動作確認
  python sparse_observation_eval.py                   # 検証データ全部（128本）
"""
import sys
import os
import argparse
import csv
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "savi-pytorch"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset

from model import SAViModel
from params import SAViParams
from movi_dataset import MoviDataset
from evaluate_movi import adjusted_rand_index
# 元のモデルと完全一致することを確認済みの部品を、そのまま使う
from diagnose_predictor import encode_frame, run_predictor, run_correction, decode_slots

EQUIV_TOL = 1e-4
MODES = ("predictor", "copy")


def forward_sparse(model, video, noise, k, mode):
    """k フレームに 1 回だけ画像を見せて推論する。

    video: [B, T, 3, H, W], noise: 初期スロット用の乱数 [B, K, D]
    戻り値: recon_combined [B, T, 3, H, W], masks [B, K, T, 1, H, W],
            observed（各フレームで画像を見たかどうかの True/False のリスト）
    """
    c = model.corrector
    B, T = video.shape[0], video.shape[1]
    K = model.num_slots

    slots = c.slots_mu + c.slots_log_sigma.exp() * noise   # model.py 238行と同じ初期化
    prev = slots
    outs = []
    observed = []

    for t in range(T):
        obs = (t % k == 0)
        if obs:
            # 画像を見るフレーム：元のモデルと同じ処理
            enc = encode_frame(model, video[:, t])
            s_in = c.norm_slots(prev)
            s_pred = run_predictor(c, s_in) if t != 1 else s_in   # 元コードの `if i != 1`
            s_out = run_correction(c, enc, s_pred)
        elif mode == "predictor":
            # 画像を見ないフレーム：予測器だけでスロットを1フレーム先へ運ぶ
            s_out = run_predictor(c, c.norm_slots(prev))
        elif mode == "copy":
            # 画像を見ないフレーム：前のスロットをそのまま使う
            s_out = prev
        else:
            raise ValueError(f"未知の mode: {mode}")

        prev = s_out
        outs.append(decode_slots(model, s_out))
        observed.append(obs)

    out = torch.stack(outs, dim=1)                         # [B*K, T, 4, H, W]
    H, W = out.shape[-2], out.shape[-1]
    C = model.in_channels
    out = out.view(B, K, T, C + 1, H, W)
    recons = out[:, :, :, :C, :, :]
    masks = F.softmax(out[:, :, :, C:, :, :], dim=1)
    recon_combined = torch.sum(recons * masks, dim=1)
    return recon_combined, masks, observed


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_phase3/savi_final.pth",
                    help="savi_phase3 フォルダからの相対パス")
    ap.add_argument("--num-videos", type=int, default=None,
                    help="検証データの先頭から何本使うか（省略すると全部）")
    ap.add_argument("--num-frames", type=int, default=6,
                    help="動画の先頭から何フレーム使うか（訓練は6。最大24）")
    ap.add_argument("--ks", default="1,2,3,6",
                    help="試す k をカンマ区切りで（1 は必ず含める。等価性チェックに使う）")
    ap.add_argument("--seed", type=int, default=0,
                    help="初期スロットの乱数の基準値。動画 n 本目には seed+n を使う")
    ap.add_argument("--out", default=None,
                    help="結果の CSV（省略時は diagnosis_results/sparse_observation_T<フレーム数>.csv）")
    args = ap.parse_args()

    ks = sorted({int(x) for x in args.ks.split(",")})
    if 1 not in ks or min(ks) < 1:
        raise ValueError("--ks には 1 を含め、すべて 1 以上にしてください")
    out_rel = args.out or f"diagnosis_results/sparse_observation_T{args.num_frames}.csv"

    params = SAViParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ===== 検証データ: train_savi.py / evaluate_movi.py と同じ分割 =====
    full_dataset = MoviDataset(os.path.join(_HERE, params.data_root))
    total = len(full_dataset)
    val_size = min(params.num_val_videos or 250, total // 10)
    train_size = total - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    if args.num_videos is not None:
        val_dataset = Subset(val_dataset, range(min(args.num_videos, len(val_dataset))))
    print(f"使う検証データ: {len(val_dataset)}本 / 使うフレーム数: {args.num_frames} / k: {ks}")
    if args.num_frames != params.seq_len:
        print(f"注意: 訓練は {params.seq_len} フレームで行っています。"
              f"{args.num_frames} フレームでの結果は、訓練より長い動画への一般化も含みます。")
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ===== モデル =====
    model = SAViModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
    ).to(device)
    ckpt_path = os.path.join(_HERE, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"チェックポイント: {ckpt_path} "
          f"(epoch {ckpt.get('epoch', -1) + 1}, step {ckpt.get('global_step', '?')})")

    T = args.num_frames
    rows = []
    max_equiv_diff = 0.0

    with torch.no_grad():
        for vid_idx, batch in enumerate(loader):
            video = batch["video"].to(device)[:, :T]
            seg = batch["segmentations"].to(device)[:, :T]
            if video.shape[1] != T:
                raise ValueError(f"動画のフレーム数が {video.shape[1]} しかありません（--num-frames {T}）")
            gt_masks = seg.squeeze(0).permute(0, 3, 1, 2)       # [T, 1, H, W]

            # 元の model.forward を乱数固定で実行（k=1 の等価性チェック用）
            s = args.seed + vid_idx
            torch.manual_seed(s)
            ref_recon, _, ref_masks, _ = model(video)
            torch.manual_seed(s)
            noise = torch.randn((video.shape[0], model.num_slots, model.slot_size)).to(device)

            for k in ks:
                for mode in MODES:
                    recon, masks, observed = forward_sparse(model, video, noise, k, mode)

                    if k == 1:
                        diff = max((recon - ref_recon).abs().max().item(),
                                   (masks - ref_masks).abs().max().item())
                        max_equiv_diff = max(max_equiv_diff, diff)
                        if diff > EQUIV_TOL:
                            raise RuntimeError(
                                f"動画{vid_idx}, k=1, {mode}: 元の model.forward と一致しません "
                                f"(最大差 {diff:.3e})"
                            )

                    pm = masks.squeeze(0).permute(1, 0, 2, 3, 4)  # [T, K, 1, H, W]
                    ari = adjusted_rand_index(gt_masks, pm).tolist()
                    mse = ((recon - video) ** 2).mean(dim=(0, 2, 3, 4)).tolist()
                    for t in range(T):
                        rows.append({
                            "video": vid_idx, "k": k, "mode": mode, "frame": t,
                            "observed": int(observed[t]),
                            "fg_ari": ari[t], "recon_mse": mse[t],
                        })

            if (vid_idx + 1) % 16 == 0 or vid_idx + 1 == len(val_dataset):
                print(f"{vid_idx + 1}/{len(val_dataset)} 本 完了")

    # ===== CSV に保存 =====
    out_path = os.path.join(_HERE, out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ===== まとめ =====
    n_videos = len(val_dataset)

    def sel(k, mode, observed=None):
        return [r for r in rows if r["k"] == k and r["mode"] == mode
                and (observed is None or r["observed"] == observed)]

    print("=" * 78)
    print(f"等価性チェック（k=1）: 元の model.forward との最大差 {max_equiv_diff:.2e}（許容 {EQUIV_TOL:.0e}）")
    print()
    print(" k | 運び方    | エンコーダ呼び出し | FG-ARI 全体 | 見たフレーム | 見ないフレーム | MSE 見ないフレーム")
    for k in ks:
        n_obs = sum(1 for t in range(T) if t % k == 0)
        for mode in MODES:
            a_all = mean([r["fg_ari"] for r in sel(k, mode)])
            a_obs = mean([r["fg_ari"] for r in sel(k, mode, 1)])
            a_un = mean([r["fg_ari"] for r in sel(k, mode, 0)])
            m_un = mean([r["recon_mse"] for r in sel(k, mode, 0)])
            print(f"{k:2d} | {mode:9s} | {n_obs:2d}/{T:2d} 回           |   {a_all:.4f}    |    {a_obs:.4f}    |"
                  f"     {a_un:.4f}     |   {m_un:.6f}")

    print()
    print("見ないフレームでの比較（動画ごとに 予測器 − コピー の FG-ARI 平均を計算）")
    for k in ks:
        if k == 1:
            continue
        diffs = []
        for v in range(n_videos):
            p = mean([r["fg_ari"] for r in rows if r["video"] == v and r["k"] == k
                      and r["mode"] == "predictor" and r["observed"] == 0])
            c = mean([r["fg_ari"] for r in rows if r["video"] == v and r["k"] == k
                      and r["mode"] == "copy" and r["observed"] == 0])
            diffs.append(p - c)
        m, sd = mean(diffs), std(diffs)
        half = 1.96 * sd / math.sqrt(n_videos) if n_videos > 1 else float("nan")
        n_pred_better = sum(1 for d in diffs if d > 0)
        print(f"  k={k}: 平均 {m:+.4f}（近似95%区間 {m - half:+.4f}〜{m + half:+.4f}）, "
              f"予測器の方が高かった動画 {n_pred_better}/{n_videos}本")

    print()
    print(f"CSV を保存しました: {out_path}")


if __name__ == "__main__":
    main()
