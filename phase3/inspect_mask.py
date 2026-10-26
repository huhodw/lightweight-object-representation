import numpy as np

# 1本だけ読む（本番データには触れず、確認のみ）
path = "movi_a_data/video_970.npz"
data = np.load(path)

print("キー一覧:", list(data.files))

video = data["video"]
seg = data["segmentations"]

print("video: shape", video.shape, "dtype", video.dtype,
      "min", video.min(), "max", video.max())
print("segmentations: shape", seg.shape, "dtype", seg.dtype,
      "min", seg.min(), "max", seg.max())

# マスクに実際どんな値(物体ID)が入っているか（0=背景, 1,2,...=物体）
unique_ids = np.unique(seg)
print("マスクに含まれる値(物体ID):", unique_ids)
print("物体IDの種類数:", len(unique_ids))