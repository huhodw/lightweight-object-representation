import torch
import numpy as np
from torch.utils.data import Dataset

class MultiDSprites(Dataset):
    """
    シンプルなMultiDSpritesライクなデータセット
    ランダムな色付き図形（円・四角）を背景に配置した画像を生成する
    """
    def __init__(self, num_samples=1000, image_size=64, num_objects=3, split='train'):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_objects = num_objects
        self.split = split
        
        # 再現性のためシードを固定
        seed = 0 if split == 'train' else 1
        self.rng = np.random.RandomState(seed)
        
        # データを事前生成
        self.images, self.masks = self._generate_data()
    
    def _generate_data(self):
        images = []
        masks = []
        
        for _ in range(self.num_samples):
            image, mask = self._generate_scene()
            images.append(image)
            masks.append(mask)
        
        images = np.stack(images)  # (N, 3, H, W)
        masks = np.stack(masks)    # (N, num_objects+1, H, W)
        return images, masks
    
    def _generate_scene(self):
        H, W = self.image_size, self.image_size
        
        # 黒背景
        image = np.zeros((3, H, W), dtype=np.float32)
        # マスク：0番目が背景、1以降が各物体
        mask = np.zeros((self.num_objects + 1, H, W), dtype=np.float32)
        mask[0] = 1.0  # 最初は全部背景
        
        for obj_idx in range(self.num_objects):
            # ランダムな色
            color = self.rng.uniform(0.3, 1.0, size=3)
            
            # ランダムな位置とサイズ
            size = self.rng.randint(8, 16)
            cx = self.rng.randint(size, W - size)
            cy = self.rng.randint(size, H - size)
            
            # 円か四角をランダムに選択
            shape = self.rng.randint(0, 2)
            
            obj_mask = np.zeros((H, W), dtype=bool)
            
            if shape == 0:  # 円
                for y in range(H):
                    for x in range(W):
                        if (x - cx)**2 + (y - cy)**2 <= size**2:
                            obj_mask[y, x] = True
            else:  # 四角
                y1, y2 = max(0, cy-size), min(H, cy+size)
                x1, x2 = max(0, cx-size), min(W, cx+size)
                obj_mask[y1:y2, x1:x2] = True
            
            # 画像に色を塗る
            for c in range(3):
                image[c][obj_mask] = color[c]
            
            # マスクを更新
            mask[0][obj_mask] = 0.0        # 背景から除外
            mask[obj_idx + 1][obj_mask] = 1.0  # 物体マスクに追加
        
        return image, mask
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = torch.FloatTensor(self.images[idx])
        mask = torch.FloatTensor(self.masks[idx])
        return image, mask


if __name__ == '__main__':
    # 動作確認
    dataset = MultiDSprites(num_samples=100, image_size=64, num_objects=3)
    image, mask = dataset[0]
    print(f"画像サイズ: {image.shape}")   # (3, 64, 64)
    print(f"マスクサイズ: {mask.shape}")  # (4, 64, 64)
    print("データセット生成OK!")