# test_dinov2.py
"""
DINOv2 ViT-B の動作確認。
- モデルが読み込めるか
- 出力形状を確認
- multi_dsprites の画像で実際に特徴抽出できるか
- 24GB VRAM で動くか
"""

import torch
import torch.nn.functional as F
import numpy as np

print("=" * 60)
print("DINOv2 ViT-B 動作確認")
print("=" * 60)

# ============================================
# Step 1: ライブラリ確認
# ============================================
print("\n[Step 1] ライブラリ確認")
try:
    from transformers import AutoImageProcessor, AutoModel
    print("  ✅ transformers インポート成功")
except ImportError as e:
    print(f"  ❌ transformers がない: {e}")
    print("  → pip install transformers を実行してください")
    exit(1)

# ============================================
# Step 2: DINOv2モデル読み込み
# ============================================
print("\n[Step 2] DINOv2 ViT-B を読み込み")
print("  初回はダウンロード（約350MB）が走ります...")

model_name = 'facebook/dinov2-base'
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 凍結
for param in model.parameters():
    param.requires_grad = False
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"  ✅ モデル読み込み成功")
print(f"  デバイス: {device}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  全パラメータ: {total_params:,} ({total_params/1e6:.1f} M)")
print(f"  訓練可能パラメータ: {trainable_params:,} (凍結確認)")

# ============================================
# Step 3: 1サンプルで動作確認
# ============================================
print("\n[Step 3] ダミー画像で動作確認")

# multi_dspritesと同じ形のダミー画像（64×64）
dummy_image_64 = torch.randn(1, 3, 64, 64)
print(f"  入力（リサイズ前）: {dummy_image_64.shape}")

# 224×224 にリサイズ
dummy_image_224 = F.interpolate(
    dummy_image_64, size=(224, 224), mode='bilinear', align_corners=False
).to(device)
print(f"  入力（リサイズ後）: {dummy_image_224.shape}")

# DINOv2 で特徴抽出
with torch.no_grad():
    outputs = model(pixel_values=dummy_image_224)

# 出力構造を確認
print(f"\n  出力オブジェクトの型: {type(outputs)}")
print(f"  出力に含まれる属性:")
for key in outputs.keys():
    val = outputs[key]
    if isinstance(val, torch.Tensor):
        print(f"    - {key}: shape={val.shape}, dtype={val.dtype}")
    else:
        print(f"    - {key}: type={type(val)}")

# last_hidden_state を詳しく
last_hidden = outputs.last_hidden_state
print(f"\n  last_hidden_state:")
print(f"    shape: {last_hidden.shape}")
print(f"    → これは (バッチ, トークン数, 特徴次元) の形")
print(f"    バッチサイズ:  {last_hidden.shape[0]}")
print(f"    トークン数:    {last_hidden.shape[1]} (内訳: CLS×1 + パッチ×N)")
print(f"    特徴次元:      {last_hidden.shape[2]}")

# CLSトークン以外（パッチトークンのみ）を取り出す
patch_tokens = last_hidden[:, 1:, :]
num_patches = patch_tokens.shape[1]
patch_size = int(num_patches ** 0.5)
print(f"\n  パッチトークン:")
print(f"    数: {num_patches}")
print(f"    格子サイズ: {patch_size} × {patch_size}")

# ============================================
# Step 4: バッチで動作確認（VRAM使用量チェック）
# ============================================
print("\n[Step 4] バッチサイズ64でVRAM確認")

if device.type == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

dummy_batch = torch.randn(64, 3, 224, 224).to(device)
print(f"  入力: {dummy_batch.shape}")

with torch.no_grad():
    outputs = model(pixel_values=dummy_batch)

if device.type == 'cuda':
    vram_used = torch.cuda.max_memory_allocated() / 1e9
    print(f"  VRAM使用量: {vram_used:.2f} GB")
    
last_hidden = outputs.last_hidden_state
print(f"  出力: {last_hidden.shape}")

# ============================================
# Step 5: 実際の multi_dsprites データで試す
# ============================================
print("\n[Step 5] multi_dsprites の実データで動作確認")

from multi_dsprites_dataset import MultiDSpritesDataset
from torch.utils.data import DataLoader

val_dataset = MultiDSpritesDataset(
    npz_path="data/multi_dsprites_70k.npz",
    split='val'
)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
batch = next(iter(val_loader))

images_64 = batch['image']  # (8, 3, 64, 64)
print(f"  multi_dsprites バッチ: {images_64.shape}")

# 224×224 にリサイズ
images_224 = F.interpolate(images_64, size=(224, 224), mode='bilinear', align_corners=False)
print(f"  リサイズ後: {images_224.shape}")

# DINOv2 で特徴抽出
images_224 = images_224.to(device)
with torch.no_grad():
    outputs = model(pixel_values=images_224)

features = outputs.last_hidden_state[:, 1:, :]  # CLSを除く
print(f"  DINOv2出力（パッチトークンのみ）: {features.shape}")
print(f"  → これがSlot Attentionに入る特徴")

print("\n" + "=" * 60)
print("✅ DINOv2 動作確認完了")
print("=" * 60)