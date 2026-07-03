import tensorflow_datasets as tfds
import torch
import numpy as np
import os

# 保存先のフォルダを作る（なければ作る）
SAVE_DIR = "movi_a_data"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_VIDEOS = 9703

print(f"MOVi-A から {NUM_VIDEOS} 本だけ取り出して保存します...")

ds = tfds.load(
    "movi_a/128x128",
    data_dir="gs://kubric-public/tfds",
    split="train",
)
ds = ds.take(NUM_VIDEOS)

# 1本ずつ処理する
for i, example in enumerate(tfds.as_numpy(ds)):
    video = example["video"]              # (24,128,128,3) uint8
    segmentations = example["segmentations"]  # (24,128,128,1)
    forward_flow = example["forward_flow"]    # (24,128,128,2)

    # この1本を .npz ファイルに保存する
    save_path = os.path.join(SAVE_DIR, f"video_{i:03d}.npz")
    np.savez_compressed(
        save_path,
        video=video,
        segmentations=segmentations,
        forward_flow=forward_flow,
    )
    print(f"  保存しました: {save_path}  （動画の形: {video.shape}）")

print("=" * 50)
print("保存おわり。次に、保存したものを読み直して確認します...")
print("=" * 50)

# 整合性チェック：保存した1本目を読み直して、形が保たれているか確認
check_path = os.path.join(SAVE_DIR, "video_000.npz")
loaded = np.load(check_path)
print("読み直した動画の形:", loaded["video"].shape)
print("読み直した正解の形:", loaded["segmentations"].shape)
print("読み直した動きの形:", loaded["forward_flow"].shape)

# おまけ：読み直したものをPyTorchテンソルに変換して並べ替えまでできるか確認
video_t = torch.from_numpy(loaded["video"]).permute(0, 3, 1, 2)
print("PyTorch流に並べ替えた形:", tuple(video_t.shape))
print("整合性チェック完了。")