"""
phase2_dinov2_slot.py の history.json を読んで、訓練の進捗をグラフ化する。

使い方:
    python plot_history.py
    python plot_history.py --history checkpoints_phase2_500k/history.json
    python plot_history.py --output plots/progress.png
"""

import json
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # GUI なしで使う（バックグラウンド実行に対応）
import matplotlib.pyplot as plt


def load_history(path):
    """history.json を読んで dict を返す"""
    with open(path, 'r') as f:
        return json.load(f)


def unpack(history_list):
    """history の (step, value) リストを 2 つのリストに分解"""
    if not history_list:
        return [], []
    steps, values = zip(*history_list)
    return list(steps), list(values)


def plot_history(history, output_path):
    """4つのサブプロットを描画"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Train Loss
    ax = axes[0, 0]
    steps, values = unpack(history.get('train_loss', []))
    if steps:
        ax.plot(steps, values, 'b-', linewidth=0.8, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Train Loss')
        ax.set_title(f'Train Loss (latest: {values[-1]:.4f} at step {steps[-1]})')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # loss は対数スケールで見やすい
    else:
        ax.set_title('Train Loss (no data)')

    # 2. Val Loss
    ax = axes[0, 1]
    steps, values = unpack(history.get('val_loss', []))
    if steps:
        ax.plot(steps, values, 'r-o', linewidth=1.5, markersize=4)
        ax.set_xlabel('Step')
        ax.set_ylabel('Val Loss')
        ax.set_title(f'Val Loss (latest: {values[-1]:.4f} at step {steps[-1]})')
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title('Val Loss (no data)')

    # 3. FG-ARI（メインの指標）
    ax = axes[1, 0]
    steps, values = unpack(history.get('val_fg_ari', []))
    if steps:
        ax.plot(steps, values, 'g-o', linewidth=2, markersize=5)
        ax.axhline(y=0.85, color='orange', linestyle='--', alpha=0.5, label='Locatello 2020 (0.85)')
        ax.axhline(y=0.94, color='red', linestyle='--', alpha=0.5, label='Phase 1 CNN (0.94)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Val FG-ARI')
        ax.set_title(f'Val FG-ARI (latest: {values[-1]:.4f} at step {steps[-1]})')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.0)
    else:
        ax.set_title('Val FG-ARI (no data)')

    # 4. Learning Rate
    ax = axes[1, 1]
    steps, values = unpack(history.get('lr', []))
    if steps:
        ax.plot(steps, values, 'm-', linewidth=1)
        ax.set_xlabel('Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'Learning Rate (latest: {values[-1]:.2e} at step {steps[-1]})')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    else:
        ax.set_title('Learning Rate (no data)')

    # 全体タイトル
    n_train = len(history.get('train_loss', []))
    n_val = len(history.get('val_fg_ari', []))
    fig.suptitle(
        f'Phase 2 (DINOv2 + Slot Attention) Training Progress  |  '
        f'train logs: {n_train}, val checkpoints: {n_val}',
        fontsize=14
    )

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f"[OK] Graph saved: {output_path}")

    # 最新の数値も表示
    print()
    print("=" * 60)
    print("Latest values")
    print("=" * 60)
    if history.get('val_fg_ari'):
        latest_step, latest_ari = history['val_fg_ari'][-1]
        print(f"  Val FG-ARI:  {latest_ari:.4f}  (step {latest_step})")
    if history.get('val_loss'):
        latest_step, latest_val = history['val_loss'][-1]
        print(f"  Val Loss:    {latest_val:.4f}  (step {latest_step})")
    if history.get('train_loss'):
        latest_step, latest_train = history['train_loss'][-1]
        print(f"  Train Loss:  {latest_train:.4f}  (step {latest_step})")
    if history.get('lr'):
        latest_step, latest_lr = history['lr'][-1]
        print(f"  LR:          {latest_lr:.2e}  (step {latest_step})")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--history',
        type=str,
        default='checkpoints_phase2_500k/history.json',
        help='Path to history.json'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='plots/progress.png',
        help='Output PNG path'
    )
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"[NG] history.json not found: {args.history}")
        return

    history = load_history(args.history)
    plot_history(history, args.output)


if __name__ == '__main__':
    main()