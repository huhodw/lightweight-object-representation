# test_tfrecord.py
"""
tfrecord読み込みのprotobuf互換性テスト。
"""

import tensorflow as tf
import numpy as np

print("=" * 60)
print("Step 1: TensorFlow import OK")
print(f"  TF version: {tf.__version__}")
print("=" * 60)

print("\nStep 2: ダミーtfrecordを書き出し中...")

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

import os
os.makedirs('./data', exist_ok=True)
output_path = './data/test_dummy.tfrecord'

with tf.io.TFRecordWriter(output_path) as writer:
    for i in range(3):
        img = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
        img_bytes = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(img_bytes),
            'index': _int64_feature(i),
        }))
        writer.write(example.SerializeToString())

print(f"  書き出し完了: {output_path}")

print("\nStep 3: tfrecordを読み込み中...")

feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'index': tf.io.FixedLenFeature([], tf.int64),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

raw_dataset = tf.data.TFRecordDataset(output_path)
parsed_dataset = raw_dataset.map(_parse_function)

for i, parsed in enumerate(parsed_dataset):
    img_raw = parsed['image'].numpy()
    img = np.frombuffer(img_raw, dtype=np.uint8).reshape(64, 64, 3)
    idx = parsed['index'].numpy()
    print(f"  サンプル {i}: index={idx}, image shape={img.shape}, dtype={img.dtype}")

print("\n" + "=" * 60)
print("✅ tfrecord読み込み成功！protobuf互換性は問題なし")
print("=" * 60)