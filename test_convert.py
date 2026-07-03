import tensorflow_datasets as tfds
import torch

print("MOVi-A を1本取り出して、PyTorchテンソルに変換します...")

# 1本だけ取り出す（前回と同じ）
ds = tfds.load(
    "movi_a/128x128",
    data_dir="gs://kubric-public/tfds",
    split="train",
)
ds = ds.take(1)

for example in tfds.as_numpy(ds):
    video_np = example["video"]          # NumPy配列（TensorFlow流の並び）
    seg_np = example["segmentations"]

    print("=" * 50)
    print("【変換前】NumPy配列")
    print("  動画の形:", video_np.shape, "／ 型:", video_np.dtype)

    # NumPy配列 → PyTorchテンソルに変換
    video_t = torch.from_numpy(video_np)
    seg_t = torch.from_numpy(seg_np)

    print("【変換後】PyTorchテンソル（まだTensorFlow流の並び）")
    print("  動画の形:", tuple(video_t.shape), "／ 型:", video_t.dtype)

    # 色を最後から前に移す（TF流→PyTorch流の並べ替え）
    # (コマ,縦,横,色) → (コマ,色,縦,横)
    video_pt = video_t.permute(0, 3, 1, 2)

    print("【並べ替え後】PyTorch流の並び (コマ,色,縦,横)")
    print("  動画の形:", tuple(video_pt.shape))
    print("=" * 50)

print("変換おわり。")