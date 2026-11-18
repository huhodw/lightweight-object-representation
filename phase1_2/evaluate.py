import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MultiDSprites
from model import SlotAttentionAutoEncoder


def compute_fg_ari(true_masks, pred_masks):
    """
    FG-ARI（Foreground Adjusted Rand Index）を計算する
    背景を除いた前景物体のみでARIを計算する

    true_masks: (B, num_objects+1, H, W) — 正解マスク（0番目が背景）
    pred_masks: (B, num_slots, 1, H, W) — 予測マスク
    """
    batch_size = true_masks.shape[0]
    ari_scores = []

    for b in range(batch_size):
        # 正解マスク：背景(index=0)を除いた前景のみ
        # (num_objects, H, W)
        true_fg = true_masks[b, 1:]  # 背景を除く

        # 前景ピクセルのマスク（どこかの物体に属するピクセル）
        fg_pixel_mask = true_fg.sum(dim=0) > 0  # (H, W)

        if fg_pixel_mask.sum() == 0:
            continue  # 前景ピクセルがない場合はスキップ

        # 正解ラベル：各ピクセルがどの物体に属するか
        # (num_objects, H, W) → 各ピクセルのクラスID
        true_label = true_fg.argmax(dim=0)  # (H, W)

        # 予測マスク：(num_slots, 1, H, W) → (num_slots, H, W)
        pred = pred_masks[b, :, 0, :]  # (num_slots, H, W)
        pred_label = pred.argmax(dim=0)  # (H, W) 最大スロットのID

        # 前景ピクセルのみ抽出
        true_flat = true_label[fg_pixel_mask].numpy()
        pred_flat = pred_label[fg_pixel_mask].numpy()

        # ARIを計算
        ari = adjusted_rand_index(true_flat, pred_flat)
        ari_scores.append(ari)

    return np.mean(ari_scores) if ari_scores else 0.0


def adjusted_rand_index(true_labels, pred_labels):
    """
    Adjusted Rand Index（ARI）をスクラッチで計算する
    2つのクラスタリング結果の一致度を測る指標（-1〜1、1が完全一致）
    """
    # 各ペアの組み合わせを数える
    n = len(true_labels)
    
    # 混同行列を作成
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)
    
    contingency = np.zeros((len(true_classes), len(pred_classes)), dtype=np.int64)
    
    true_to_idx = {c: i for i, c in enumerate(true_classes)}
    pred_to_idx = {c: i for i, c in enumerate(pred_classes)}
    
    for t, p in zip(true_labels, pred_labels):
        contingency[true_to_idx[t], pred_to_idx[p]] += 1
    
    # ARIの計算
    sum_comb_c = sum(comb2(n_c) for n_c in contingency.sum(axis=1))
    sum_comb_k = sum(comb2(n_k) for n_k in contingency.sum(axis=0))
    sum_comb = sum(comb2(n_ij) for n_ij in contingency.flatten())
    
    total_comb = comb2(n)
    
    expected_index = sum_comb_c * sum_comb_k / (total_comb + 1e-10)
    max_index = (sum_comb_c + sum_comb_k) / 2
    
    if max_index - expected_index == 0:
        return 1.0
    
    ari = (sum_comb - expected_index) / (max_index - expected_index + 1e-10)
    return float(ari)


def comb2(n):
    """C(n, 2) = n*(n-1)/2 を計算する"""
    return n * (n - 1) / 2


def evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # データセット
    val_dataset = MultiDSprites(
        num_samples=200,
        image_size=config['image_size'],
        num_objects=config['num_objects'],
        split='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # モデルをロード
    model = SlotAttentionAutoEncoder(
        num_slots=config['num_slots'],
        dim=config['dim'],
        iters=config['iters']
    ).to(device)

    model.load_state_dict(torch.load('slot_attention_model.pth', map_location=device))
    model.eval()
    print("モデルをロードしました")

    # FG-ARIを計算
    all_ari = []

    with torch.no_grad():
        for images, true_masks in val_loader:
            images = images.to(device)

            # 予測
            recon, recons, pred_masks, slots = model(images)

            # CPUに移してFG-ARIを計算
            pred_masks_cpu = pred_masks.cpu()
            true_masks_cpu = true_masks.cpu()

            ari = compute_fg_ari(true_masks_cpu, pred_masks_cpu)
            all_ari.append(ari)

    mean_ari = np.mean(all_ari)
    print(f"\n===== 評価結果 =====")
    print(f"FG-ARI: {mean_ari:.4f}")

    if mean_ari >= 0.85:
        print("✅ 論文値（0.85以上）達成！Phase 2に進めます！")
    else:
        print(f"❌ まだ不足（目標: 0.85、現在: {mean_ari:.4f}）")
        print("→ エポック数を増やすか、ハイパーパラメータを調整してください")

    return mean_ari


if __name__ == '__main__':
    config = {
        'image_size': 64,
        'num_objects': 3,
        'num_slots': 4,
        'dim': 64,
        'iters': 3,
    }
    evaluate(config)