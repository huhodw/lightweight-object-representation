# verify.py
"""
Phase 1 / Phase 2 の訓練結果を可視化する。

使い方:
  python verify.py --phase phase1   # Phase 1 のみ可視化
  python verify.py --phase phase2   # Phase 2 のみ可視化
  python verify.py --phase both     # 両方 + 比較

出力先: ./verify_output/
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse


# ============================================================
# 学習履歴の可視化
# ============================================================
def plot_history(history, title_prefix, save_path):
    """
    history.json から訓練曲線を描画。
    
    history の中身（タプルのリスト形式）:
      train_loss: [(step, loss), ...]
      val_loss:   [(step, loss), ...]
      val_fg_ari: [(step, ari), ...]
      lr:         [(step, lr), ...]
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # 1. 訓練 Loss
    ax = axes[0]
    if len(history['train_loss']) > 0:
        steps, losses = zip(*history['train_loss'])
        ax.plot(steps, losses, 'b-', label='train', alpha=0.7)
    if len(history['val_loss']) > 0:
        v_steps, v_losses = zip(*history['val_loss'])
        ax.plot(v_steps, v_losses, 'r-o', label='val', markersize=5)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'{title_prefix} Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. FG-ARI
    ax = axes[1]
    if len(history['val_fg_ari']) > 0:
        steps, aris = zip(*history['val_fg_ari'])
        ax.plot(steps, aris, 'g-o', markersize=5)
        # 標準偏差のシェード
        if len(history['val_fg_ari_std']) > 0:
            _, stds = zip(*history['val_fg_ari_std'])
            ax.fill_between(steps,
                            np.array(aris) - np.array(stds),
                            np.array(aris) + np.array(stds),
                            alpha=0.2, color='g')
    ax.set_xlabel('Step')
    ax.set_ylabel('FG-ARI')
    ax.set_title(f'{title_prefix} FG-ARI')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.0])
    
    # 3. Learning Rate
    ax = axes[2]
    if len(history['lr']) > 0:
        steps, lrs = zip(*history['lr'])
        ax.plot(steps, lrs, 'm-')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title(f'{title_prefix} Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: {save_path}")


# ============================================================
# Phase 1 モデルでの可視化
# ============================================================
def visualize_phase1(checkpoint_path, save_dir, num_samples=4, seed=42):
    """Phase 1 (CNN Encoder) のベストモデルで再構成画像を可視化"""
    from phase1_multi_dsprites import SlotAttentionAutoEncoder64
    from multi_dsprites_dataset import MultiDSpritesDataset
    
    print(f"\n[Phase 1] モデルロード中: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル構築・重みロード
    model = SlotAttentionAutoEncoder64(
        resolution=(64, 64), num_slots=7, num_iterations=3, hidden_dim=64
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  ステップ: {ckpt.get('step', '?')}")
    print(f"  ベスト FG-ARI: {ckpt.get('best_ari', '?')}")
    
    # 検証データから決定的に画像を選ぶ
    val_dataset = MultiDSpritesDataset(
        npz_path="data/multi_dsprites_70k.npz",
        split='val'
    )
    
    # seed を固定して同じ画像を選ぶ
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(val_dataset), num_samples, replace=False).tolist()
    print(f"  使用する検証画像インデックス: {indices}")
    
    # 画像を集める
    images = torch.stack([val_dataset[i]['image'] for i in indices]).to(device)
    
    # forward
    with torch.no_grad():
        recon, recons, masks, slots = model(images, return_slots=True)
    
    # CPU に移して numpy 化
    images_np = images.cpu().numpy()           # (N, 3, 64, 64), [0, 1]
    recon_np = recon.cpu().numpy()             # (N, 3, 64, 64), [可能性として 範囲外]
    recons_np = recons.cpu().numpy()           # (N, K, 3, 64, 64)
    masks_np = masks.cpu().numpy()             # (N, K, 1, 64, 64)
    
    # クリップ（[0, 1] に収める）
    recon_np = np.clip(recon_np, 0, 1)
    recons_np = np.clip(recons_np, 0, 1)
    
    # 可視化
    save_path = os.path.join(save_dir, 'phase1_reconstructions.png')
    _plot_reconstructions(images_np, recon_np, recons_np, masks_np,
                          title_prefix='Phase 1 (CNN Encoder)',
                          save_path=save_path,
                          best_ari=ckpt.get('best_ari', None))
    
    return indices, images_np, recon_np, recons_np, masks_np


# ============================================================
# Phase 2 モデルでの可視化
# ============================================================
def visualize_phase2(checkpoint_path, save_dir, num_samples=4, seed=42):
    """Phase 2 (DINOv2 ViT-B Encoder) のベストモデルで再構成画像を可視化"""
    from phase2_dinov2_slot import (
        SlotAttentionDINOv2, normalize_for_dinov2,
        IMAGENET_MEAN, IMAGENET_STD
    )
    from multi_dsprites_dataset import MultiDSpritesDataset
    
    print(f"\n[Phase 2] モデルロード中: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデル構築・重みロード
    print(f"  DINOv2 を読み込み中...")
    model = SlotAttentionDINOv2(
        num_slots=7, slot_dim=384, decoder_dim=64,
        num_iterations=3, freeze_vit=True
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  ステップ: {ckpt.get('step', '?')}")
    print(f"  ベスト FG-ARI: {ckpt.get('best_ari', '?')}")
    
    # 検証データから同じ画像を選ぶ（同じ seed なら Phase 1 と一致）
    val_dataset = MultiDSpritesDataset(
        npz_path="data/multi_dsprites_70k.npz",
        split='val'
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(val_dataset), num_samples, replace=False).tolist()
    print(f"  使用する検証画像インデックス: {indices}")
    
    images = torch.stack([val_dataset[i]['image'] for i in indices]).to(device)
    
    # 正規化（Phase 2 は ImageNet 正規化が必要）
    images_normalized = normalize_for_dinov2(images, device)
    
    # forward
    with torch.no_grad():
        recon_norm, recons_norm, masks, slots = model(images_normalized, return_slots=True)
    
    # 逆正規化：正規化された再構成画像 → 元のスケールへ
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    recon = recon_norm * std + mean
    recons = recons_norm * std + mean
    
    # CPU に移して numpy 化
    images_np = images.cpu().numpy()
    recon_np = recon.cpu().numpy()
    recons_np = recons.cpu().numpy()
    masks_np = masks.cpu().numpy()
    
    # クリップ
    recon_np = np.clip(recon_np, 0, 1)
    recons_np = np.clip(recons_np, 0, 1)
    
    # 可視化
    save_path = os.path.join(save_dir, 'phase2_reconstructions.png')
    _plot_reconstructions(images_np, recon_np, recons_np, masks_np,
                          title_prefix='Phase 2 (DINOv2 ViT-B Encoder)',
                          save_path=save_path,
                          best_ari=ckpt.get('best_ari', None))
    
    return indices, images_np, recon_np, recons_np, masks_np


# ============================================================
# 再構成画像のプロット（共通）
# ============================================================
def _plot_reconstructions(images_np, recon_np, recons_np, masks_np,
                          title_prefix, save_path, best_ari=None):
    """
    画像、合成再構成、各スロットの再構成、各スロットのマスクを並べる。
    
    images_np: (N, 3, 64, 64) [0, 1]
    recon_np:  (N, 3, 64, 64) [0, 1]
    recons_np: (N, K, 3, 64, 64) [0, 1]
    masks_np:  (N, K, 1, 64, 64) [0, 1]
    """
    N = images_np.shape[0]
    K = recons_np.shape[1]
    
    # 列数: 元 + 再構成 + K個のスロット再構成 + K個のスロットマスク
    n_cols = 2 + 2 * K
    
    fig, axes = plt.subplots(N, n_cols, figsize=(2 * n_cols, 2 * N))
    if N == 1:
        axes = axes[None, :]
    
    title = title_prefix
    if best_ari is not None:
        title += f' | FG-ARI = {best_ari:.4f}'
    fig.suptitle(title, fontsize=14, y=1.0)
    
    for n in range(N):
        # 元画像
        ax = axes[n, 0]
        ax.imshow(images_np[n].transpose(1, 2, 0))
        ax.set_title('Original' if n == 0 else '', fontsize=10)
        ax.axis('off')
        
        # 合成再構成
        ax = axes[n, 1]
        ax.imshow(recon_np[n].transpose(1, 2, 0))
        ax.set_title('Recon (sum)' if n == 0 else '', fontsize=10)
        ax.axis('off')
        
        # 各スロットの再構成
        for k in range(K):
            ax = axes[n, 2 + k]
            ax.imshow(recons_np[n, k].transpose(1, 2, 0))
            ax.set_title(f'Slot {k} recon' if n == 0 else '', fontsize=10)
            ax.axis('off')
        
        # 各スロットのマスク
        for k in range(K):
            ax = axes[n, 2 + K + k]
            ax.imshow(masks_np[n, k, 0], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Slot {k} mask' if n == 0 else '', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: {save_path}")


# ============================================================
# Phase 1 vs Phase 2 の並列比較
# ============================================================
def plot_comparison(images_np, p1_recon, p1_masks, p2_recon, p2_masks,
                    p1_ari, p2_ari, save_path):
    """
    同じ画像で Phase 1 と Phase 2 を並べる。
    
    各画像について：
    - 元画像
    - Phase 1 再構成 + マスク統合
    - Phase 2 再構成 + マスク統合
    """
    N = images_np.shape[0]
    
    fig, axes = plt.subplots(N, 5, figsize=(15, 3 * N))
    if N == 1:
        axes = axes[None, :]
    
    fig.suptitle(
        f'Phase 1 (FG-ARI = {p1_ari:.4f}) vs Phase 2 (FG-ARI = {p2_ari:.4f})',
        fontsize=14, y=1.0
    )
    
    for n in range(N):
        # 元画像
        ax = axes[n, 0]
        ax.imshow(images_np[n].transpose(1, 2, 0))
        ax.set_title('Original' if n == 0 else '', fontsize=11)
        ax.axis('off')
        
        # Phase 1 再構成
        ax = axes[n, 1]
        ax.imshow(p1_recon[n].transpose(1, 2, 0))
        ax.set_title('Phase 1 Recon' if n == 0 else '', fontsize=11)
        ax.axis('off')
        
        # Phase 1 マスク統合（各ピクセルがどのスロットに属するか色分け）
        ax = axes[n, 2]
        # masks: (K, 1, 64, 64) → argmax で各ピクセル→スロットID
        p1_seg = p1_masks[n, :, 0].argmax(axis=0)  # (64, 64)
        ax.imshow(p1_seg, cmap='tab10', vmin=0, vmax=9)
        ax.set_title('Phase 1 Segmentation' if n == 0 else '', fontsize=11)
        ax.axis('off')
        
        # Phase 2 再構成
        ax = axes[n, 3]
        ax.imshow(p2_recon[n].transpose(1, 2, 0))
        ax.set_title('Phase 2 Recon' if n == 0 else '', fontsize=11)
        ax.axis('off')
        
        # Phase 2 マスク統合
        ax = axes[n, 4]
        p2_seg = p2_masks[n, :, 0].argmax(axis=0)
        ax.imshow(p2_seg, cmap='tab10', vmin=0, vmax=9)
        ax.set_title('Phase 2 Segmentation' if n == 0 else '', fontsize=11)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: {save_path}")


# ============================================================
# 学習履歴の並列比較
# ============================================================
def plot_history_comparison(p1_history, p2_history, save_path):
    """Phase 1 と Phase 2 の学習曲線を重ねて表示"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 比較
    ax = axes[0]
    if len(p1_history['train_loss']) > 0:
        steps, losses = zip(*p1_history['train_loss'])
        ax.plot(steps, losses, 'b-', label='Phase 1 (CNN) train', alpha=0.5)
    if len(p1_history['val_loss']) > 0:
        steps, losses = zip(*p1_history['val_loss'])
        ax.plot(steps, losses, 'b-o', label='Phase 1 (CNN) val', markersize=5)
    if len(p2_history['train_loss']) > 0:
        steps, losses = zip(*p2_history['train_loss'])
        ax.plot(steps, losses, 'r-', label='Phase 2 (DINOv2) train', alpha=0.5)
    if len(p2_history['val_loss']) > 0:
        steps, losses = zip(*p2_history['val_loss'])
        ax.plot(steps, losses, 'r-o', label='Phase 2 (DINOv2) val', markersize=5)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # FG-ARI 比較
    ax = axes[1]
    if len(p1_history['val_fg_ari']) > 0:
        steps, aris = zip(*p1_history['val_fg_ari'])
        ax.plot(steps, aris, 'b-o', label='Phase 1 (CNN)', markersize=6)
    if len(p2_history['val_fg_ari']) > 0:
        steps, aris = zip(*p2_history['val_fg_ari'])
        ax.plot(steps, aris, 'r-o', label='Phase 2 (DINOv2)', markersize=6)
    ax.set_xlabel('Step')
    ax.set_ylabel('FG-ARI')
    ax.set_title('FG-ARI Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: {save_path}")


# ============================================================
# メイン
# ============================================================
def main(phase, num_samples=4, seed=42):
    save_dir = './verify_output'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"=" * 60)
    print(f"可視化開始")
    print(f"  対象: {phase}")
    print(f"  サンプル数: {num_samples}")
    print(f"  乱数シード: {seed}（同じシードで Phase 1, 2 同じ画像）")
    print(f"  出力先: {save_dir}")
    print(f"=" * 60)
    
    # ========================
    # 学習曲線の可視化
    # ========================
    if phase in ['phase1', 'both']:
        print(f"\n[1] Phase 1 学習曲線")
        history_path = './checkpoints_phase1/history.json'
        with open(history_path, 'r') as f:
            p1_history = json.load(f)
        plot_history(p1_history, 'Phase 1 (CNN Encoder)',
                     os.path.join(save_dir, 'phase1_history.png'))
    
    if phase in ['phase2', 'both']:
        print(f"\n[2] Phase 2 学習曲線")
        history_path = './checkpoints_phase2/history.json'
        with open(history_path, 'r') as f:
            p2_history = json.load(f)
        plot_history(p2_history, 'Phase 2 (DINOv2 ViT-B Encoder)',
                     os.path.join(save_dir, 'phase2_history.png'))
    
    # ========================
    # ベストモデルでの再構成
    # ========================
    p1_result = None
    p2_result = None
    
    if phase in ['phase1', 'both']:
        print(f"\n[3] Phase 1 再構成画像")
        p1_result = visualize_phase1(
            './checkpoints_phase1/best.pth',
            save_dir, num_samples=num_samples, seed=seed
        )
    
    if phase in ['phase2', 'both']:
        print(f"\n[4] Phase 2 再構成画像")
        p2_result = visualize_phase2(
            './checkpoints_phase2/best.pth',
            save_dir, num_samples=num_samples, seed=seed
        )
    
    # ========================
    # 並列比較（both のときだけ）
    # ========================
    if phase == 'both' and p1_result is not None and p2_result is not None:
        print(f"\n[5] Phase 1 vs Phase 2 比較")
        
        # 同じ画像インデックスを使っているか確認
        p1_indices = p1_result[0]
        p2_indices = p2_result[0]
        assert p1_indices == p2_indices, "両 Phase で同じインデックスを使うべき"
        
        images_np = p1_result[1]    # 元画像（Phase 1 のものを使う、Phase 2 でも同じ）
        p1_recon = p1_result[2]
        p1_masks = p1_result[4]
        p2_recon = p2_result[2]
        p2_masks = p2_result[4]
        
        p1_ari = json.load(open('./checkpoints_phase1/history.json'))
        p2_ari = json.load(open('./checkpoints_phase2/history.json'))
        # best_ari を best.pth から取り出す
        p1_best = torch.load('./checkpoints_phase1/best.pth', map_location='cpu').get('best_ari', 0)
        p2_best = torch.load('./checkpoints_phase2/best.pth', map_location='cpu').get('best_ari', 0)
        
        plot_comparison(
            images_np, p1_recon, p1_masks, p2_recon, p2_masks,
            p1_best, p2_best,
            os.path.join(save_dir, 'comparison.png')
        )
        
        # 学習曲線の並列比較
        print(f"\n[6] 学習曲線の並列比較")
        plot_history_comparison(
            p1_history, p2_history,
            os.path.join(save_dir, 'history_comparison.png')
        )
    
    print(f"\n" + "=" * 60)
    print(f"✅ 可視化完了")
    print(f"   出力先: {save_dir}")
    print(f"=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='both',
                       choices=['phase1', 'phase2', 'both'])
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    main(args.phase, num_samples=args.num_samples, seed=args.seed)