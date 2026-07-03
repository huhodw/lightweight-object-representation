import tensorflow_datasets as tfds

print("MOVi-A を1本だけ取り出します（少し時間がかかります）...")

# 訓練用データから「1本だけ」取り出す
ds = tfds.load(
    "movi_a/128x128",
    data_dir="gs://kubric-public/tfds",
    split="train",
)
ds = ds.take(1)  # 1本だけに絞る

# 取り出した1本の中身を確認する
for example in tfds.as_numpy(ds):
    video = example["video"]
    segmentations = example["segmentations"]
    forward_flow = example["forward_flow"]

    print("=" * 50)
    print("動画(video)の形:", video.shape, "／ 中身の型:", video.dtype)
    print("正解の物体分け(segmentations)の形:", segmentations.shape)
    print("動きの地図(forward_flow)の形:", forward_flow.shape)
    print("=" * 50)
    print("動画の画素値の範囲: 最小", video.min(), "〜 最大", video.max())
    print("この動画に含まれる物体の番号:", sorted(set(segmentations.flatten().tolist())))
    print("=" * 50)

print("確認おわり。")