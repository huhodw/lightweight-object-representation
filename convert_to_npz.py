# convert_to_npz.py
"""
multi_dsprites のtfrecord（GZIP圧縮）を numpy の npz 形式に変換する。
DeepMind公式コードに完全準拠。
一度実行すれば、以降はTensorFlow不要でデータが読める。
"""

import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm

# ============================================
# 設定
# ============================================
TFRECORD_PATH = "data/multi_dsprites_colored_on_grayscale.tfrecords"
OUTPUT_PATH = "data/multi_dsprites_70k.npz"

# 取り出すサンプル数（Slot Attention論文の使用量）
NUM_SAMPLES = 70000

# 公式定数
IMAGE_SIZE = [64, 64]
MAX_NUM_ENTITIES = 6  # colored_on_grayscale
COMPRESSION_TYPE = 'GZIP'
BYTE_FEATURES = ['mask', 'image']

# ============================================
# 公式定義（DeepMind multi_dsprites.py より）
# ============================================
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

def _decode(example_proto):
    """公式準拠のデコード関数"""
    parsed = tf.io.parse_single_example(example_proto, features)
    for k in BYTE_FEATURES:
        parsed[k] = tf.squeeze(tf.io.decode_raw(parsed[k], tf.uint8), axis=-1)
    parsed['mask'] = tf.transpose(parsed['mask'], [2, 0, 1, 3])
    return parsed

# ============================================
# 変換実行
# ============================================
print(f"変換開始")
print(f"  入力: {TFRECORD_PATH}")
print(f"  出力: {OUTPUT_PATH}")
print(f"  サンプル数: {NUM_SAMPLES}")

ds = tf.data.TFRecordDataset(TFRECORD_PATH, compression_type=COMPRESSION_TYPE)
ds = ds.map(_decode)

# numpy配列を事前確保（メモリ効率のため）
# image: (N, 64, 64, 3), uint8
# mask:  (N, 6, 64, 64, 1), uint8
# visibility: (N, 6), float32
print("\nNumpy配列を確保...")
images = np.zeros((NUM_SAMPLES, 64, 64, 3), dtype=np.uint8)
masks = np.zeros((NUM_SAMPLES, MAX_NUM_ENTITIES, 64, 64, 1), dtype=np.uint8)
visibilities = np.zeros((NUM_SAMPLES, MAX_NUM_ENTITIES), dtype=np.float32)

print(f"  images:       {images.shape} ({images.nbytes / 1e6:.1f} MB)")
print(f"  masks:        {masks.shape} ({masks.nbytes / 1e6:.1f} MB)")
print(f"  visibilities: {visibilities.shape} ({visibilities.nbytes / 1e6:.1f} MB)")
total_memory = (images.nbytes + masks.nbytes + visibilities.nbytes) / 1e9
print(f"  合計: {total_memory:.2f} GB")

# 変換ループ
print("\n変換中...")
for i, sample in enumerate(tqdm(ds.take(NUM_SAMPLES), total=NUM_SAMPLES, desc='Converting')):
    images[i] = sample['image'].numpy()
    masks[i] = sample['mask'].numpy()
    visibilities[i] = sample['visibility'].numpy()

# 保存
print(f"\n保存中（時間がかかります）...")
np.savez(
    OUTPUT_PATH,
    images=images,
    masks=masks,
    visibilities=visibilities,
)

# 確認
file_size = os.path.getsize(OUTPUT_PATH)
print(f"\n✅ 変換完了")
print(f"  出力ファイル: {OUTPUT_PATH}")
print(f"  サイズ: {file_size / 1e9:.2f} GB")
print(f"  images.shape:       {images.shape} ({images.dtype})")
print(f"  masks.shape:        {masks.shape} ({masks.dtype})")
print(f"  visibilities.shape: {visibilities.shape} ({visibilities.dtype})")