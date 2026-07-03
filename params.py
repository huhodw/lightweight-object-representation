from typing import Optional
from typing import Tuple
import attr


@attr.s(auto_attribs=True)
class SAViParams:
    # 学習率
    lr: float = 0.0002

    # バッチサイズ
    batch_size: int = 32            # 最初は小さく。動作確認後に大きくする
    val_batch_size: int = 4

    # 解像度: 君のMOViデータは128×128
    resolution: Tuple[int, int] = (128, 128)

    # スロット数: 論文通り11個
    num_slots: int = 11

    # Correctorの繰り返し回数: MOVi/MOVi++は1回
    num_iterations: int = 1

    # 君のデータフォルダのパス
    data_root: str = "movi_a_data"

    # GPU: 1枚(RTX 4500)
    gpus: int = 1

    # 訓練エポック数(あとで変える、まずは動作確認用)
    max_epochs: int = 100

    # 検証ステップ数
    num_sanity_val_steps: int = 1

    # 学習率スケジューラの設定
    scheduler_gamma: float = 0.5
    weight_decay: float = 0.0

    # データ本数の制限(Noneは全部使う)
    num_train_videos: Optional[int] = None
    num_val_videos: Optional[int] = 128

    # DataLoaderのワーカー数(Windowsは0が安全)
    num_workers: int = 0

    # ログ関連
    n_samples: int = 1
    is_logger_enabled: bool = False    # まずwandbなしで動作確認
    is_verbose: bool = True

    # warmupとdecayの割合(論文通り)
    warmup_steps_pct: float = 0.025
    decay_steps_pct: float = 0.2

    # CNNの隠れ層チャンネル数
    hidden_dims: Tuple[int, ...] = (32, 32, 32, 32)

    # デコーダの隠れ層チャンネル数
    decoder_hidden_dims: Tuple[int, ...] = (128, 64, 64, 64, 64)

    # FG-ARIのログ
    log_ari: bool = True

    # 訓練時のシーケンス長(論文通り6コマ)
    seq_len: int = 6
