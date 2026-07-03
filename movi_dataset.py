import numpy as np
import torch
from torch.utils.data import Dataset
import os
import glob


class MoviDataset(Dataset):
    def __init__(self, data_dir):
        # data_dir の中の .npz ファイルを全部探して、名前順に並べる
        pattern = os.path.join(data_dir, "video_*.npz")
        self.file_paths = sorted(glob.glob(pattern))

        # 何本見つかったかを表示する（確認用）
        print(f"{data_dir} の中に {len(self.file_paths)} 本の動画が見つかりました")

    def __len__(self):
        # 全部で何本あるかを返す
        return len(self.file_paths)

    def __getitem__(self, idx):
        # idx 番目のファイルのパスを取り出す
        path = self.file_paths[idx]

        # .npz を読み込む
        data = np.load(path)

        # 中身を取り出す（保存したときの名前で取り出せる）
        video = data["video"]                  # (24, 128, 128, 3) uint8
        segmentations = data["segmentations"]  # (24, 128, 128, 1)

        # NumPy配列 → PyTorchテンソルに変換
        video = torch.from_numpy(video)
        segmentations = torch.from_numpy(segmentations)

        # 色を最後から前に並べ替える（TF流 → PyTorch流）
        # (コマ, 縦, 横, 色) → (コマ, 色, 縦, 横)
        video = video.permute(0, 3, 1, 2)

        # 画素値を 0〜255 から 0〜1 の範囲に変換する（floatにする）
        video = video.float() / 255.0

        # 結果を辞書にまとめて返す
        return {
            "video": video,
            "segmentations": segmentations,
        }
    
    # ===== 以下は動作確認用 =====
if __name__ == "__main__":
    # Datasetを作る
    dataset = MoviDataset("movi_a_data")

    # 全部で何本あるか
    print("データ数:", len(dataset))

    # 0番目を取り出してみる
    sample = dataset[0]
    print("動画の形:", tuple(sample["video"].shape))
    print("正解の形:", tuple(sample["segmentations"].shape))
    print("動画の値の範囲: 最小", sample["video"].min().item(), "〜 最大", sample["video"].max().item())