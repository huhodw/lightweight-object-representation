import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "savi-pytorch"))

import numpy as np
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from model import SAViModel
from params import SAViParams
from movi_dataset import MoviDataset


# ===== どの動画を見るか（評価時の「動画N」に対応。0始まりなので N-1）=====
VIDEO_INDEX = 6   # 評価時の「動画7」（0.99→0.27…と崩れた動画）


def main():
    params = SAViParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ===== 評価と全く同じ分割で検証データを作る（seed=42 を必ず一致させる）=====
    full_dataset = MoviDataset(params.data_root)
    total = len(full_dataset)
    val_size = min(params.num_val_videos or 250, total // 10)
    train_size = total - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"検証データ: {len(val_dataset)}本  /  見る動画: index {VIDEO_INDEX}")

    # ===== モデル読み込み（推論のみ）=====
    model = SAViModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
    ).to(device)
    checkpoint = torch.load("checkpoints_phase3/savi_final.pth",
                            map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("チェックポイント読み込み完了")

    # ===== 1本だけ取り出して、最初の seq_len コマを使う =====
    sample = val_dataset[VIDEO_INDEX]
    video = sample["video"].unsqueeze(0).to(device)     # [1, 24, 3, H, W]
    video = video[:, :params.seq_len, :, :, :]          # [1, seq_len, 3, H, W]

    with torch.no_grad():
        recon_combined, recons, masks, slots_all = model(video)
        # recon_combined: [1, seq_len, 3, H, W]
        # masks:          [1, num_slots, seq_len, 1, H, W]

    # ===== 表示用にCPU・numpyへ。画像は (H,W,3) の並びに戻す =====
    seq_len = params.seq_len
    num_slots = params.num_slots

    video_np = video[0].permute(0, 2, 3, 1).cpu().numpy()          # [seq_len, H, W, 3]
    recon_np = recon_combined[0].permute(0, 2, 3, 1).cpu().numpy() # [seq_len, H, W, 3]
    recon_np = np.clip(recon_np, 0.0, 1.0)                          # 表示のため0-1に収める

    # 各ピクセルを「一番担当しているスロット番号」に割り当てて色分け地図を作る
    masks_np = masks[0].squeeze(2).cpu().numpy()   # [num_slots, seq_len, H, W]
    seg_map = masks_np.argmax(axis=0)              # [seq_len, H, W]  値=スロット番号

    # ===== 図を描く：行=コマ、列=[元画像 / 復元 / スロット色分け] =====
    fig, axes = plt.subplots(seq_len, 3, figsize=(9, 3 * seq_len))
    col_titles = ["元画像", "復元", "スロット色分け"]
    for t in range(seq_len):
        axes[t, 0].imshow(video_np[t])
        axes[t, 1].imshow(recon_np[t])
        axes[t, 2].imshow(seg_map[t], cmap="tab20", vmin=0, vmax=num_slots - 1)
        axes[t, 0].set_ylabel(f"コマ {t+1}", fontsize=12)
        for c in range(3):
            axes[t, c].set_xticks([])
            axes[t, c].set_yticks([])
            if t == 0:
                axes[t, c].set_title(col_titles[c], fontsize=12)

    plt.tight_layout()
    out_path = f"vis_video_{VIDEO_INDEX}.png"
    plt.savefig(out_path, dpi=120)
    print(f"保存しました: {out_path}")


if __name__ == "__main__":
    main()