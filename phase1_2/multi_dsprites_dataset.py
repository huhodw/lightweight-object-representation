# multi_dsprites_dataset.py
"""
multi_dsprites のPyTorch Datasetクラス。
事前変換済みの npz ファイルから高速にデータを読み込む。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiDSpritesDataset(Dataset):
    """
    multi_dsprites/colored_on_grayscale データセット。
    
    Args:
        npz_path: convert_to_npz.py で作成した .npz ファイルのパス
        split: 'train' / 'val' / 'test' のいずれか
        train_size: 訓練サンプル数（デフォルト 60000）
        val_size: 検証サンプル数（デフォルト 5000）
        test_size: テストサンプル数（デフォルト 5000）
                  合計 70000 になるよう設計
    
    Returns (各サンプル):
        'image': torch.Tensor (3, 64, 64), float32, 値域 [0, 1]
        'mask':  torch.Tensor (6, 64, 64), float32, 値域 [0, 1]
        'visibility': torch.Tensor (6,), float32, 0 or 1
    """
    
    def __init__(self, npz_path, split='train', 
                 train_size=60000, val_size=5000, test_size=5000):
        # データ読み込み（メモリマップで効率的に）
        # mmap_mode='r' でファイルをメモリにすべて展開せず、必要な部分だけ読む
        data = np.load(npz_path, mmap_mode='r')
        self.images = data['images']            # (70000, 64, 64, 3) uint8
        self.masks = data['masks']              # (70000, 6, 64, 64, 1) uint8
        self.visibilities = data['visibilities'] # (70000, 6) float32
        
        # スプリットに応じてインデックス範囲を決める
        total = len(self.images)
        assert train_size + val_size + test_size <= total, \
            f"split合計 {train_size + val_size + test_size} > 全サンプル数 {total}"
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = train_size
        elif split == 'val':
            self.start_idx = train_size
            self.end_idx = train_size + val_size
        elif split == 'test':
            self.start_idx = train_size + val_size
            self.end_idx = train_size + val_size + test_size
        else:
            raise ValueError(f"split must be 'train' / 'val' / 'test', got '{split}'")
        
        self.split = split
        print(f"[MultiDSpritesDataset:{split}] {self.end_idx - self.start_idx} samples")
    
    def __len__(self):
        """データセットのサンプル数を返す"""
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        """idx番目のサンプルを返す"""
        # 実際のインデックスを計算
        real_idx = self.start_idx + idx
        
        # 画像: (64, 64, 3) uint8 → (3, 64, 64) float32 [0,1]
        # PyTorchの畳み込み層は (C, H, W) 形式を期待するので、軸を入れ替える
        image_np = self.images[real_idx]                          # (64, 64, 3) uint8
        image = torch.from_numpy(image_np.copy()).float() / 255.0 # (64, 64, 3) float32 [0,1]
        image = image.permute(2, 0, 1)                            # (3, 64, 64)
        
        # マスク: (6, 64, 64, 1) uint8 → (6, 64, 64) float32 [0,1]
        # 末尾のチャンネル次元を削除し、値域を [0, 255] → [0, 1] に正規化
        mask_np = self.masks[real_idx]                            # (6, 64, 64, 1) uint8
        mask = torch.from_numpy(mask_np.copy()).float() / 255.0   # (6, 64, 64, 1) float32 [0,1]
        mask = mask.squeeze(-1)                                   # (6, 64, 64)
        
        # 可視性: (6,) float32
        visibility = torch.from_numpy(self.visibilities[real_idx].copy())
        
        return {
            'image': image,
            'mask': mask,
            'visibility': visibility,
        }


# ============================================
# 動作テスト
# ============================================
if __name__ == '__main__':
    """このファイルを直接実行した時のテストコード"""
    from torch.utils.data import DataLoader
    
    print("=" * 60)
    print("MultiDSpritesDataset 動作テスト")
    print("=" * 60)
    
    # Datasetインスタンスを作成
    train_dataset = MultiDSpritesDataset(
        npz_path="data/multi_dsprites_70k.npz",
        split='train'
    )
    val_dataset = MultiDSpritesDataset(
        npz_path="data/multi_dsprites_70k.npz",
        split='val'
    )
    
    print(f"\nDataset sizes:")
    print(f"  train: {len(train_dataset)}")
    print(f"  val:   {len(val_dataset)}")
    
    # 1サンプル取り出して構造確認
    print(f"\n--- 1サンプルの構造 ---")
    sample = train_dataset[0]
    print(f"  image:      shape={sample['image'].shape}, dtype={sample['image'].dtype}, range=[{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  mask:       shape={sample['mask'].shape}, dtype={sample['mask'].dtype}, range=[{sample['mask'].min():.3f}, {sample['mask'].max():.3f}]")
    print(f"  visibility: shape={sample['visibility'].shape}, value={sample['visibility'].tolist()}")
    
    # DataLoaderで1バッチ取り出してみる
    print(f"\n--- DataLoaderで1バッチ取得 ---")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    batch = next(iter(train_loader))
    print(f"  images.shape:       {batch['image'].shape}")
    print(f"  masks.shape:        {batch['mask'].shape}")
    print(f"  visibilities.shape: {batch['visibility'].shape}")
    
    print(f"\n✅ 動作テスト完了")