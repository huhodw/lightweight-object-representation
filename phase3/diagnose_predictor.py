"""
diagnose_predictor.py

Phase 3 の SAVi モデルで「予測器が実際に働いているか」を診断する。
再訓練はしない。保存済みチェックポイントを読み込んで推論するだけ。

測るもの
  (A) 予測器ありの推論（元のモデルと同じ）と、予測器なしの推論
      （フレーム2以降で予測器を飛ばす）を、同じ初期スロットで実行し、
      フレームごとの FG-ARI と再構成誤差(MSE)を比べる。
  (B) 予測器ありの推論の中で、予測器がスロットをどれだけ動かしているか
      （相対変化量・コサイン類似度）と、その後の補正がどれだけ動かしているか。

置き場所: savi_phase3 フォルダの直下（evaluate_movi.py と同じ場所）
実行例:
  python diagnose_predictor.py --num-videos 3     # まず3本で動作確認
  python diagnose_predictor.py                    # 検証データ全部（128本）
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
# 0.6046 を出したときと同じ FG-ARI 関数をそのまま使う
from evaluate_movi import adjusted_rand_index

# 自前の推論と元の model.forward の出力が、この値以上ずれたら止める
EQUIV_TOL = 1e-4


# ---------------------------------------------------------------------------
# 元の SAViModel.forward / Corrector.forward を部品に分けて書き直したもの。
# 計算の中身と順番は model.py と同じ。違いは「予測器を使うかどうか」を
# 外から指定できることと、途中のスロットを取り出せることだけ。
# ---------------------------------------------------------------------------

def encode_frame(model, x):
    """1フレームを CNN エンコーダに通す（model.py 244-250行と同じ）。
    x: [B, 3, H, W] -> 戻り値: [B, H*W, 64]"""
    h = model.encoder(x)
    h = model.encoder_pos_embedding(h)
    h = torch.flatten(h, start_dim=2, end_dim=3)
    h = h.permute(0, 2, 1)
    h = model.encoder_out_layer(h)
    return h


def run_predictor(c, s):
    """予測器（スロット同士の自己注意 + MLP）。model.py 86-91行と同じ。
    s: norm_slots を通した後のスロット [B, K, D]"""
    prev = s
    t = s.permute(1, 0, 2)                      # MultiheadAttention は [K, B, D] で受け取る
    t, _ = c.multihead_attn(t, t, t)
    t = t.permute(1, 0, 2)
    t = prev + t
    t = c.norm_slots(t)
    t = c.mlp_attn(t) + t
    return t


def run_correction(c, inputs, slots):
    """Slot Attention による補正。model.py 74-79行と 95-126行と同じ。
    inputs: エンコーダ出力 [B, N, 64], slots: 補正前のスロット [B, K, D]"""
    B = inputs.shape[0]
    K, D = c.num_slots, c.slot_size
    inputs = c.norm_inputs(inputs)
    k1 = c.project_k1(inputs)
    v1 = c.project_v1(inputs)
    for _ in range(c.num_iterations):
        slots_prev = slots
        slots = c.norm_slots(slots)
        q = c.project_q(slots)
        attn_logits = (c.slot_size ** -0.5) * torch.matmul(k1, q.transpose(2, 1))
        attn = F.softmax(attn_logits, dim=-1)            # スロット方向に正規化
        attn = attn + c.epsilon
        attn = attn / torch.sum(attn, dim=1, keepdim=True)  # 画素方向に正規化（重み付き平均）
        updates = torch.matmul(attn.transpose(1, 2), v1)
        slots = c.gru(updates.reshape(B * K, D), slots_prev.reshape(B * K, D))
        slots = slots.view(B, K, D)
        slots = slots + c.mlp(c.norm_mlp(slots))
    return slots


def decode_slots(model, slots):
    """スロットを Spatial Broadcast Decoder で画像に戻す（model.py 262-267行と同じ）。
    slots: [B, K, D] -> 戻り値: [B*K, 4, H, W]（RGB 3ch + マスク用 1ch）"""
    B, K, D = slots.shape
    z = slots.reshape(B * K, D, 1, 1)
    z = z.repeat(1, 1, model.decoder_resolution[0], model.decoder_resolution[1])
    out = model.decoder_pos_embedding(z)
    out = model.decoder(out)
    return out


def slot_change_stats(s_in, s_pred, s_out):
    """予測器と補正が、それぞれスロットをどれだけ動かしたかを測る。
    すべてスロットごとに計算してから、全スロットで平均する。
      pred_rel : |予測器の出力 - 予測器の入力| / |予測器の入力|
      pred_cos : 予測器の入力と出力のコサイン類似度（1 に近いほど向きが変わっていない）
      corr_rel : |補正後 - 補正前| / |補正前|（予測器の動きと比べるための参考値）"""
    eps = 1e-12
    pred_rel = ((s_pred - s_in).norm(dim=-1) / (s_in.norm(dim=-1) + eps)).mean().item()
    pred_cos = F.cosine_similarity(s_pred, s_in, dim=-1).mean().item()
    corr_rel = ((s_out - s_pred).norm(dim=-1) / (s_pred.norm(dim=-1) + eps)).mean().item()
    return pred_rel, pred_cos, corr_rel


def forward_custom(model, video, noise, use_predictor):
    """SAViModel.forward と同じ処理を、予測器の有無を選べる形で行う。

    予測器をかけるかどうか:
      フレーム0 : 元コードどおりかける（ランダムな初期スロットへの処理で、時間方向の予測ではない）
      フレーム1 : 元コードの `if i != 1` により、かけない
      フレーム2以降 : use_predictor=True ならかける（元のモデルと同じ）、False なら飛ばす

    video: [B, T, 3, H, W], noise: 初期スロット用の乱数 [B, K, D]
    戻り値: recon_combined [B, T, 3, H, W], masks [B, K, T, 1, H, W], フレームごとの統計
    """
    c = model.corrector
    B, T = video.shape[0], video.shape[1]
    K = model.num_slots

    slots = c.slots_mu + c.slots_log_sigma.exp() * noise   # model.py 238行と同じ初期化
    prev = slots
    outs = []
    stats = []

    for i in range(T):
        enc = encode_frame(model, video[:, i])
        s_in = c.norm_slots(prev)                          # model.py 81行

        if i == 0:
            apply_pred = True
        elif i == 1:
            apply_pred = False
        else:
            apply_pred = use_predictor

        s_pred = run_predictor(c, s_in) if apply_pred else s_in
        s_out = run_correction(c, enc, s_pred)

        # 予測器が「前の時刻から次の時刻へ運ぶ」役割で使われたフレームだけ統計を取る
        if i >= 2 and apply_pred:
            stats.append(slot_change_stats(s_in, s_pred, s_out))
        else:
            stats.append(None)

        prev = s_out
        outs.append(decode_slots(model, s_out))

    out = torch.stack(outs, dim=1)                         # [B*K, T, 4, H, W]
    H, W = out.shape[-2], out.shape[-1]
    C = model.in_channels
    out = out.view(B, K, T, C + 1, H, W)
    recons = out[:, :, :, :C, :, :]
    masks = F.softmax(out[:, :, :, C:, :, :], dim=1)       # スロット方向の softmax
    recon_combined = torch.sum(recons * masks, dim=1)
    return recon_combined, masks, stats


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
    ap.add_argument("--seed", type=int, default=0,
                    help="初期スロットの乱数の基準値。動画 n 本目には seed+n を使う")
    ap.add_argument("--out", default="diagnosis_results/predictor_ablation.csv",
                    help="結果の CSV（savi_phase3 フォルダからの相対パス）")
    args = ap.parse_args()

    params = SAViParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    # GPU の畳み込みで、同じ入力から毎回まったく同じ値が出るようにする
    # （2条件の差が、計算の揺れではなくモデルの違いだけから来るようにするため）
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
    print(f"使う検証データ: {len(val_dataset)}本")
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

    T = params.seq_len
    conds = [("original", True), ("predictor_off", False)]
    rows = []
    max_equiv_diff = 0.0

    with torch.no_grad():
        for vid_idx, batch in enumerate(loader):
            video = batch["video"].to(device)[:, :T]            # [1, T, 3, H, W]
            seg = batch["segmentations"].to(device)[:, :T]      # [1, T, H, W, 1]
            gt_masks = seg.squeeze(0).permute(0, 3, 1, 2)       # [T, 1, H, W]（evaluate_movi.py と同じ）

            # 元の model.forward を、乱数を固定して実行（等価性チェック用）
            s = args.seed + vid_idx
            torch.manual_seed(s)
            ref_recon, _, ref_masks, _ = model(video)

            # 同じ乱数をもう一度作り、2つの条件で共通の初期スロットとして使う
            torch.manual_seed(s)
            noise = torch.randn((video.shape[0], model.num_slots, model.slot_size)).to(device)

            results = {}
            for name, use_pred in conds:
                results[name] = forward_custom(model, video, noise, use_pred)

            # 自前の推論（予測器あり）が元の model.forward と一致するか
            diff = max(
                (results["original"][0] - ref_recon).abs().max().item(),
                (results["original"][1] - ref_masks).abs().max().item(),
            )
            max_equiv_diff = max(max_equiv_diff, diff)
            if diff > EQUIV_TOL:
                raise RuntimeError(
                    f"動画{vid_idx}: 自前の推論と元の model.forward が一致しません "
                    f"(最大差 {diff:.3e})。model.py が想定と違う可能性があります。"
                )

            for name, _ in conds:
                recon, masks, stats = results[name]
                pm = masks.squeeze(0).permute(1, 0, 2, 3, 4)    # [T, K, 1, H, W]
                ari = adjusted_rand_index(gt_masks, pm).tolist()
                mse = ((recon - video) ** 2).mean(dim=(0, 2, 3, 4)).tolist()
                for t in range(T):
                    st = stats[t]
                    rows.append({
                        "video": vid_idx,
                        "frame": t,
                        "condition": name,
                        "fg_ari": ari[t],
                        "recon_mse": mse[t],
                        "pred_rel_change": st[0] if st else float("nan"),
                        "pred_cos": st[1] if st else float("nan"),
                        "corr_rel_change": st[2] if st else float("nan"),
                    })

            o = mean([r["fg_ari"] for r in rows if r["video"] == vid_idx and r["condition"] == "original"])
            p = mean([r["fg_ari"] for r in rows if r["video"] == vid_idx and r["condition"] == "predictor_off"])
            print(f"動画 {vid_idx+1:3d}: FG-ARI 予測器あり {o:.4f} / なし {p:.4f}")

    # ===== CSV に保存 =====
    out_path = os.path.join(_HERE, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ===== まとめ =====
    def pick(cond, frame, key):
        return [r[key] for r in rows if r["condition"] == cond and r["frame"] == frame]

    n_videos = len(val_dataset)
    print("=" * 72)
    print(f"等価性チェック: 元の model.forward との最大差 {max_equiv_diff:.2e}（許容 {EQUIV_TOL:.0e}）")
    print()
    print("フレーム | FG-ARI あり | FG-ARI なし |  差(あり-なし) |  MSE あり  |  MSE なし")
    for t in range(T):
        a_on, a_off = mean(pick("original", t, "fg_ari")), mean(pick("predictor_off", t, "fg_ari"))
        m_on, m_off = mean(pick("original", t, "recon_mse")), mean(pick("predictor_off", t, "recon_mse"))
        print(f"   {t}     |   {a_on:.4f}    |   {a_off:.4f}    |   {a_on - a_off:+.4f}     | "
              f"{m_on:.6f} | {m_off:.6f}")

    all_on = mean([r["fg_ari"] for r in rows if r["condition"] == "original"])
    print()
    print(f"予測器あり・全{T}フレーム平均の FG-ARI: {all_on:.4f}"
          f"（evaluate_movi.py の 0.6046 と近ければ、評価の手順がそろっている）")

    # フレーム0と1は2条件で処理が同じなので、値も同じになるはず
    same01 = all(
        abs(x - y) < 1e-6
        for t in (0, 1)
        for x, y in zip(pick("original", t, "fg_ari"), pick("predictor_off", t, "fg_ari"))
    )
    print(f"フレーム0・1が2条件で一致しているか: {'一致' if same01 else '不一致（要確認）'}")

    # 動画ごとに、フレーム2以降の FG-ARI 平均の差（あり - なし）
    per_video_diff = []
    for v in range(n_videos):
        on = mean([r["fg_ari"] for r in rows
                   if r["video"] == v and r["condition"] == "original" and r["frame"] >= 2])
        off = mean([r["fg_ari"] for r in rows
                    if r["video"] == v and r["condition"] == "predictor_off" and r["frame"] >= 2])
        per_video_diff.append(on - off)
    n_off_better = sum(1 for d in per_video_diff if d < 0)
    print()
    print(f"フレーム2〜{T-1}の FG-ARI 差（あり - なし）: 平均 {mean(per_video_diff):+.4f}, "
          f"標準偏差 {std(per_video_diff):.4f}, 予測器なしの方が高かった動画 {n_off_better}/{n_videos}本")

    st_rows = [r for r in rows if r["condition"] == "original" and r["frame"] >= 2]
    print(f"予測器の相対変化量 |出力-入力|/|入力|: {mean([r['pred_rel_change'] for r in st_rows]):.4f}")
    print(f"予測器の入力と出力のコサイン類似度   : {mean([r['pred_cos'] for r in st_rows]):.4f}")
    print(f"補正の相対変化量 |補正後-補正前|/|補正前|（参考）: "
          f"{mean([r['corr_rel_change'] for r in st_rows]):.4f}")
    print()
    print(f"CSV を保存しました: {out_path}")


if __name__ == "__main__":
    main()
