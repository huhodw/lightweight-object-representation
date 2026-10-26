import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "savi-pytorch"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from model import SAViModel
from params import SAViParams
from movi_dataset import MoviDataset


# ===== adjusted_rand_index: evaluation.py から関数の中身だけコピー =====
# (evaluation.py を直接importすると、その場でCATER専用データセットを
#  読みに行ってエラーになるため、関数だけをここに複製する)
def adjusted_rand_index(true_mask, pred_mask, exclude_background=True):
    """
    1枚(1フレーム)ごとのFG-ARIを計算する。
    true_mask: [N, 1, H, W]  正解の物体ID (背景=0)
    pred_mask: [N, K, 1, H, W]  各スロットの確率マスク
    戻り値: [N] 各フレームのARI
    """
    N, _, H, W = true_mask.shape

    true_group_ids = true_mask.view(N, H * W).long()
    true_mask_oh = torch.nn.functional.one_hot(true_group_ids).float()
    if exclude_background:
        true_mask_oh[..., 0] = 0

    pred_mask = pred_mask.squeeze(2)  # [N, K, H, W]
    pred_groups = pred_mask.shape[1]
    pred_mask = torch.argmax(pred_mask, dim=1)  # [N, H, W]
    pred_group_ids = pred_mask.view(N, H * W).long()
    pred_group_oh = torch.nn.functional.one_hot(pred_group_ids, pred_groups).float()

    n_points = torch.sum(true_mask_oh, dim=[1, 2])
    nij = torch.einsum('bji,bjk->bki', pred_group_oh, true_mask_oh)
    a = torch.sum(nij, 1)
    b = torch.sum(nij, 2)

    rindex = torch.sum(nij * (nij - 1), dim=[1, 2])
    aindex = torch.sum(a * (a - 1), 1)
    bindex = torch.sum(b * (b - 1), 1)
    expected_rindex = aindex * bindex / (n_points * (n_points - 1))
    max_rindex = (aindex + bindex) / 2
    ari = (rindex - expected_rindex) / (max_rindex - expected_rindex)

    check_single_cluster = torch.sum((nij > 0).int(), dim=[1, 2])
    check_single_cluster = (1 == check_single_cluster).int()
    ari[ari != ari] = 0  # NaNを0に置き換え
    ari = check_single_cluster * torch.ones_like(ari) + (1 - check_single_cluster) * ari

    return ari


def main():
    params = SAViParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ===== 訓練時と同じ分割で検証データを取り出す =====
    # (train_savi.pyと同じ seed=42 を使うことで、訓練で使わなかった
    #  「本当の検証データ」だけを評価に使う)
    full_dataset = MoviDataset(params.data_root)
    total = len(full_dataset)
    val_size = min(params.num_val_videos or 250, total // 10)
    train_size = total - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"検証データ: {len(val_dataset)}本")

    # ▼▼▼ 小規模テスト: まず3本だけに絞る ▼▼▼
    #NUM_TEST_VIDEOS = 3
    #from torch.utils.data import Subset
    #val_dataset = Subset(val_dataset, range(min(NUM_TEST_VIDEOS, len(val_dataset))))
    #print(f"[テストモード] 先頭 {len(val_dataset)}本だけで評価します")
    # ▲▲▲

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # ===== モデルを読み込む(訓練しない、推論のみ) =====
    model = SAViModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
    ).to(device)

    checkpoint_path = "checkpoints_phase3/savi_final.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()   # 評価モード(学習しない)
    print(f"チェックポイント読み込み完了: {checkpoint_path}")

    # ===== 評価ループ =====
    ari_scores = []
    with torch.no_grad():   # 勾配を計算しない(推論のみなので不要)
        for i, batch in enumerate(val_loader):
            video = batch["video"].to(device)          # [1, 24, 3, H, W]
            seg = batch["segmentations"].to(device)      # [1, 24, H, W, 1]

            # 訓練と条件を揃えるため、最初の seq_len フレームだけ使う
            video = video[:, :params.seq_len, :, :, :]
            seg = seg[:, :params.seq_len, :, :, :]

            # モデルにスロットマスクを出させる
            recon_combined, recons, pred_masks, slots_all = model(video)
            # pred_masks: [1, num_slots, seq_len, 1, H, W]
            pred_masks = pred_masks.squeeze(0)               # [num_slots, seq_len, 1, H, W]
            pred_masks = pred_masks.permute(1, 0, 2, 3, 4)    # [seq_len, num_slots, 1, H, W]

            # 正解マスクをフレームを"バッチ"とみなす形に整形
            gt_masks = seg.squeeze(0)          # [seq_len, H, W, 1]
            gt_masks = gt_masks.permute(0, 3, 1, 2)  # [seq_len, 1, H, W]

            ari = adjusted_rand_index(gt_masks, pred_masks)
            mean_ari = ari.mean().item()
            ari_scores.append(mean_ari)
            print(f"動画 {i+1}: FG-ARI = {mean_ari:.4f}  (フレームごと: {[round(a,3) for a in ari.tolist()]})")

    overall = sum(ari_scores) / len(ari_scores)
    print("=" * 50)
    print(f"平均 FG-ARI ({len(ari_scores)}本): {overall:.4f}")


if __name__ == "__main__":
    main()