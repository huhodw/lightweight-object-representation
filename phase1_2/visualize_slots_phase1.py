"""
Phase 1 (CNN + Slot Attention) の best.pth をロードして、
MultiDSprites のサンプルに対してスロットマスクを可視化する。

使い方:
    python visualize_slots_phase1.py
    python visualize_slots_phase1.py --num_samples 4 --output plots/slot_viz_phase1.png
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from phase1_multi_dsprites import SlotAttentionAutoEncoder64
from multi_dsprites_dataset import MultiDSpritesDataset


def load_model(checkpoint_path, device):
    """Phase 1 のベストモデルをロード"""
    model = SlotAttentionAutoEncoder64(
        resolution=(64, 64),
        num_slots=7,
        num_iterations=3,
        hidden_dim=64,
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    step = ckpt.get('step', '?')
    fg_ari = ckpt.get('best_ari', ckpt.get('metrics', {}).get('val_fg_ari', '?'))
    print(f"[OK] Loaded Phase 1 checkpoint from step {step}, FG-ARI = {fg_ari}")
    
    return model


@torch.no_grad()
def visualize_sample(model, image, gt_mask, device):
    """1サンプルに対して推論し、可視化用のテンソルを返す
    Phase 1: 入力正規化なし、生の [0,1] をそのまま入力
    """
    img_batch = image.unsqueeze(0).to(device)  # (1, 3, 64, 64)
    
    # Phase 1 は正規化なしで、入力は [0, 1] のまま
    recon, recons, masks, slots = model(img_batch, return_slots=True)
    
    # CPU に戻して numpy 化
    img_np = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    recon_np = recon[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    recons_np = recons[0].cpu().numpy()  # (K, 3, H, W)
    masks_np = masks[0].cpu().numpy()  # (K, 1, H, W)
    gt_np = gt_mask.cpu().numpy()  # (K, H, W) one-hot
    
    # Phase 1 は正規化してないので、re-normalization なし
    recon_np_display = np.clip(recon_np, 0, 1)
    
    return {
        'image': img_np,
        'recon': recon_np_display,
        'recons': recons_np,
        'masks': masks_np,
        'gt': gt_np,
    }


def plot_results(results, num_slots, output_path, title_suffix=''):
    """複数サンプルの結果を1枚の Figure に並べる"""
    num_samples = len(results)
    cols = max(3, num_slots)
    rows = num_samples * 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    for i, res in enumerate(results):
        # 上段: 入力、再構築、GT マスク
        ax = axes[i*2, 0]
        ax.imshow(np.clip(res['image'], 0, 1))
        ax.set_title(f'Sample {i+1}: Input')
        ax.axis('off')
        
        ax = axes[i*2, 1]
        ax.imshow(res['recon'])
        ax.set_title('Reconstruction')
        ax.axis('off')
        
        ax = axes[i*2, 2]
        # 正解マスクは (K, H, W) のワンホット形式
        gt_label = res['gt'].argmax(axis=0)
        ax.imshow(gt_label, cmap='tab10')
        ax.set_title('Ground Truth Mask')
        ax.axis('off')
        
        for c in range(3, cols):
            axes[i*2, c].axis('off')
        
        # 下段: 各スロットのマスク
        for k in range(num_slots):
            ax = axes[i*2+1, k]
            mask = res['masks'][k, 0]
            ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Slot {k}')
            ax.axis('off')
        
        for c in range(num_slots, cols):
            axes[i*2+1, c].axis('off')
    
    fig.suptitle(f'Phase 1 (CNN + Slot Attention) Visualization{title_suffix}', fontsize=14)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_phase1_500k/best.pth')
    parser.add_argument('--npz_path', type=str, default='data/multi_dsprites_70k.npz')
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--output', type=str, default='plots/slot_viz_phase1.png')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    print(f"\nLoading dataset: {args.npz_path}")
    val_dataset = MultiDSpritesDataset(npz_path=args.npz_path, split='val')
    
    np.random.seed(args.seed)
    indices = np.random.choice(len(val_dataset), args.num_samples, replace=False)
    print(f"Sample indices: {indices.tolist()}")
    
    results = []
    for idx in indices:
        sample = val_dataset[idx]
        image = sample['image']
        gt_mask = sample['mask']
        
        res = visualize_sample(model, image, gt_mask, device)
        results.append(res)
    
    plot_results(results, num_slots=7, output_path=args.output)


if __name__ == '__main__':
    main()