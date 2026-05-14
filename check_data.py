# check_data.py
"""
multi_dsprites のデータ確認（最終版）。
公式通りにデコードし、正しく可視化する。
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

TFRECORD_PATH = "data/multi_dsprites_colored_on_grayscale.tfrecords"
IMAGE_SIZE = [64, 64]
MAX_NUM_ENTITIES = 6
COMPRESSION_TYPE = 'GZIP'
BYTE_FEATURES = ['mask', 'image']

# 公式定義
features = {
    'image': tf.io.FixedLenFeature(IMAGE_SIZE + [3], tf.string),
    'mask':  tf.io.FixedLenFeature(IMAGE_SIZE + [MAX_NUM_ENTITIES, 1], tf.string),
    'x':           tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'y':           tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'shape':       tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'color':       tf.io.FixedLenFeature([MAX_NUM_ENTITIES, 3], tf.float32),
    'visibility':  tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'orientation': tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
    'scale':       tf.io.FixedLenFeature([MAX_NUM_ENTITIES], tf.float32),
}

# 公式の_decode関数
def _decode(example_proto):
    parsed = tf.io.parse_single_example(example_proto, features)
    for k in BYTE_FEATURES:
        parsed[k] = tf.squeeze(tf.io.decode_raw(parsed[k], tf.uint8), axis=-1)
    parsed['mask'] = tf.transpose(parsed['mask'], [2, 0, 1, 3])
    return parsed

print("=" * 60)
print("multi_dsprites データ確認")
print("=" * 60)

file_size = os.path.getsize(TFRECORD_PATH)
print(f"\n✅ ファイル: {TFRECORD_PATH}")
print(f"   サイズ: {file_size / 1e9:.2f} GB")

ds = tf.data.TFRecordDataset(TFRECORD_PATH, compression_type=COMPRESSION_TYPE)
ds = ds.map(_decode)

# 3サンプルの情報を表示
print("\n" + "=" * 60)
print("最初の3サンプル")
print("=" * 60)

for i, sample in enumerate(ds.take(3)):
    print(f"\n--- サンプル {i} ---")
    print(f"  image.shape: {sample['image'].shape}, dtype={sample['image'].dtype}")
    print(f"  mask.shape:  {sample['mask'].shape}, dtype={sample['mask'].dtype}")
    print(f"  visibility:  {sample['visibility'].numpy()}")
    
    mask_np = sample['mask'].numpy()
    print(f"  各エンティティのマスクピクセル数（255の数）:")
    for e in range(MAX_NUM_ENTITIES):
        pixel_count = (mask_np[e] == 255).sum()
        vis = sample['visibility'].numpy()[e]
        print(f"    エンティティ{e}: {pixel_count:5d} pixels (vis={vis:.0f})")

# 可視化（vmax=255 に修正！）
print("\n" + "=" * 60)
print("可視化")
print("=" * 60)

first_sample = next(iter(ds.take(1)))
image = first_sample['image'].numpy()
masks = first_sample['mask'].numpy()
visibility = first_sample['visibility'].numpy()

fig, axes = plt.subplots(1, 7, figsize=(20, 3))

axes[0].imshow(image)
axes[0].set_title('Image')
axes[0].axis('off')

for j in range(6):
    mask_2d = masks[j].squeeze()  # (64, 64)
    # ★ ここが修正点: vmax=1 → vmax=255 ★
    axes[j + 1].imshow(mask_2d, cmap='gray', vmin=0, vmax=255)
    axes[j + 1].set_title(f'Entity {j}\nvis={visibility[j]:.0f}')
    axes[j + 1].axis('off')

plt.tight_layout()
plt.savefig('data_preview.png', dpi=80)
print("✅ 画像保存: data_preview.png")

print("\n" + "=" * 60)
print("✅ データ確認完了")
print("=" * 60)