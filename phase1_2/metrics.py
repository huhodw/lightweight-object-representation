# metrics.py
"""
物体分離評価のためのメトリクス。
Slot Attention の評価には Foreground ARI（FG-ARI）を使う。
"""

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score


def compute_fg_ari(pred_masks, gt_masks):
    """
    Foreground ARI（FG-ARI）を計算する。
    背景ピクセルを除外して、前景（物体）の分離精度を測る指標。
    
    Args:
        pred_masks: torch.Tensor (B, K, H, W) または (B, K, 1, H, W)
            モデルが出力したスロットマスク（softmax後の確率）
        
        gt_masks: torch.Tensor (B, N, H, W)
            正解マスク（multi_dspritesのマスク、値は0または1）
            慣習：チャンネル0は背景マスク
    
    Returns:
        float: バッチ平均のFG-ARI。範囲 [-1, 1]、1が最良。
    """
    # pred_masks の形を (B, K, H, W) に統一
    if pred_masks.dim() == 5:
        pred_masks = pred_masks.squeeze(2)
    
    B = pred_masks.shape[0]
    aris = []
    
    # 各ピクセル → 最も反応したスロットID
    pred_hard = pred_masks.argmax(dim=1)  # (B, H, W)
    
    for b in range(B):
        # 各ピクセル → 属する物体ID
        gt_hard = gt_masks[b].argmax(dim=0)  # (H, W)
        
        # 前景マスク（背景でないピクセル）
        bg_mask = gt_masks[b, 0]  # 背景マスク
        fg_pixels = (bg_mask < 0.5)
        
        # 前景がない画像はスキップ
        if fg_pixels.sum() == 0:
            continue
        
        # 前景ピクセルだけ抽出
        pred_fg = pred_hard[b][fg_pixels].cpu().numpy()
        gt_fg = gt_hard[fg_pixels].cpu().numpy()
        
        # sklearn でARI計算
        ari = adjusted_rand_score(gt_fg, pred_fg)
        aris.append(ari)
    
    return float(np.mean(aris)) if len(aris) > 0 else 0.0


def compute_ari(pred_masks, gt_masks):
    """通常のARI（背景含む全ピクセルで計算）。リファレンス用。"""
    if pred_masks.dim() == 5:
        pred_masks = pred_masks.squeeze(2)
    
    B = pred_masks.shape[0]
    aris = []
    
    pred_hard = pred_masks.argmax(dim=1)
    
    for b in range(B):
        gt_hard = gt_masks[b].argmax(dim=0)
        pred_flat = pred_hard[b].cpu().numpy().flatten()
        gt_flat = gt_hard.cpu().numpy().flatten()
        ari = adjusted_rand_score(gt_flat, pred_flat)
        aris.append(ari)
    
    return float(np.mean(aris)) if len(aris) > 0 else 0.0


# ============================================
# 動作テスト
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("FG-ARI 動作テスト")
    print("=" * 60)
    
    # ============================================
    # テスト1: 完全一致（FG-ARI = 1.0 を期待）
    # ============================================
    print("\n[テスト1] 予測 = GT の場合（理想的な一致）")
    
    # GTマスク：(B=1, N=6, H=8, W=8)
    gt = torch.zeros(1, 6, 8, 8)
    gt[0, 0, :, :] = 1.0  # 背景：全領域
    gt[0, 1, 1:3, 1:3] = 1.0; gt[0, 0, 1:3, 1:3] = 0.0  # 物体1
    gt[0, 2, 5:7, 5:7] = 1.0; gt[0, 0, 5:7, 5:7] = 0.0  # 物体2
    
    # 予測：スロット番号は別物だが、同じピクセルが同じグループ
    pred = torch.zeros(1, 8, 8, 8)
    pred[0, 7, :, :] = 1.0
    pred[0, 3, 1:3, 1:3] = 10.0; pred[0, 7, 1:3, 1:3] = 0.0
    pred[0, 5, 5:7, 5:7] = 10.0; pred[0, 7, 5:7, 5:7] = 0.0
    
    fg_ari = compute_fg_ari(pred, gt)
    full_ari = compute_ari(pred, gt)
    print(f"  FG-ARI:   {fg_ari:.4f} (期待値: 1.0)")
    print(f"  Full ARI: {full_ari:.4f}")
    
    # ============================================
    # テスト2: ランダム予測（FG-ARI ≈ 0 を期待）
    # ============================================
    print("\n[テスト2] ランダム予測の場合")
    torch.manual_seed(42)
    pred_random = torch.randn(1, 8, 8, 8)
    
    fg_ari = compute_fg_ari(pred_random, gt)
    print(f"  FG-ARI: {fg_ari:.4f} (期待値: 0付近)")
    
    # ============================================
    # テスト3: 実データで試す
    # ============================================
    print("\n[テスト3] 実データ（multi_dsprites）で試す")
    
    from multi_dsprites_dataset import MultiDSpritesDataset
    from torch.utils.data import DataLoader
    
    val_dataset = MultiDSpritesDataset(
        npz_path="data/multi_dsprites_70k.npz",
        split='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    batch = next(iter(val_loader))
    gt_masks = batch['mask']  # (8, 6, 64, 64)
    
    # GT を「完璧な予測」として使う
    pred_perfect = gt_masks.clone() * 10.0
    fg_ari = compute_fg_ari(pred_perfect, gt_masks)
    print(f"  完璧な予測の FG-ARI: {fg_ari:.4f} (期待値: 1.0)")
    
    # ランダム予測（K=7スロット、実モデルと同じ）
    torch.manual_seed(0)
    pred_random = torch.randn(8, 7, 64, 64)
    fg_ari_random = compute_fg_ari(pred_random, gt_masks)
    print(f"  ランダム予測の FG-ARI: {fg_ari_random:.4f} (期待値: 0付近)")
    
    print("\n" + "=" * 60)
    print("✅ 動作テスト完了")
    print("=" * 60)