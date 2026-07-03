import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "savi-pytorch"))

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from model import SAViModel
from params import SAViParams
from movi_dataset import MoviDataset

# ▼▼▼ デバッグ用: 環境変数 SAVI_MAX_STEPS が設定されていれば、そのステップ数で早期終了する ▼▼▼
# 例: PowerShell で $env:SAVI_MAX_STEPS=30 としてから実行すると 30ステップで止まる。
# 何も設定しなければ None になり、本番挙動（最後まで訓練）と完全に同じ。
_DEBUG_MAX_STEPS = os.environ.get("SAVI_MAX_STEPS")
_DEBUG_MAX_STEPS = int(_DEBUG_MAX_STEPS) if _DEBUG_MAX_STEPS is not None else None
# ▲▲▲


def get_lr_scheduler(optimizer, warmup_steps, total_steps, decay_steps_pct):
    decay_start = int(total_steps * (1.0 - decay_steps_pct))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= decay_start:
            progress = (step - decay_start) / max(1, total_steps - decay_start)
            return max(0.0, 1.0 - progress)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, loss, path):
    """
    モデルの重みと訓練状態をまとめて保存する。
    重みだけでなくoptimizerとschedulerも保存するので、
    途中から再開するときに完全に同じ状態から続けられる。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
    }, path)
    print(f"チェックポイント保存: {path}  (epoch {epoch+1}, step {global_step})")


def main():
    params = SAViParams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ===== データの準備 =====
    full_dataset = MoviDataset(params.data_root)
    total = len(full_dataset)

    val_size = min(params.num_val_videos or 250, total // 10)
    train_size = total - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"訓練データ: {train_size}本 / 検証データ: {val_size}本")

    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ===== モデルの準備 =====
    model = SAViModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"総パラメータ数: {total_params:,}")
    print(f"訓練可能パラメータ数: {trainable_params:,}")

    # ===== 最適化の準備 =====
    optimizer = Adam(model.parameters(), lr=params.lr)

    steps_per_epoch = math.ceil(train_size / params.batch_size)
    total_steps = steps_per_epoch * params.max_epochs
    warmup_steps = int(total_steps * params.warmup_steps_pct)

    scheduler = get_lr_scheduler(
        optimizer, warmup_steps, total_steps, params.decay_steps_pct
    )

    print(f"1エポックのステップ数: {steps_per_epoch}")
    print(f"総ステップ数: {total_steps}")
    print(f"warmupステップ数: {warmup_steps}")
    print("=" * 50)

    # ===== 訓練ループ =====
    model.train()
    global_step = 0
    latest_loss = 0.0

    for epoch in range(params.max_epochs):
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            video = batch["video"].to(device)
            # (B, 24, 3, H, W) → 最初の seq_len コマだけ使う
            video = video[:, :params.seq_len, :, :, :]

            optimizer.zero_grad()
            loss_dict = model.loss_function(video)
            loss = loss_dict["loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.05)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            latest_loss = loss.item()
            global_step += 1

            # ▼ デバッグ用の早期終了（本番では _DEBUG_MAX_STEPS が None なので発動しない）
            if _DEBUG_MAX_STEPS is not None and global_step >= _DEBUG_MAX_STEPS:
                print(f"[デバッグ] {global_step} ステップで早期終了します（動作確認用）")
                return   # main() をここで抜ける = 訓練を打ち切る

            if global_step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0] * params.lr
                print(f"step {global_step:5d} | loss: {loss.item():.4f} | lr: {current_lr:.6f}")

        # ===== エポック終了時の処理 =====
        avg_loss = epoch_loss / len(train_loader)
        print(f"epoch {epoch+1:4d} 完了 | 平均損失: {avg_loss:.4f}")

        # 10エポックごとに中間チェックポイントを保存
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, global_step, avg_loss,
                f"checkpoints_phase3/savi_epoch{epoch+1:04d}.pth"
            )

    # ===== 訓練完了時に最終モデルを保存 =====
    save_checkpoint(
        model, optimizer, scheduler,
        params.max_epochs - 1, global_step, latest_loss,
        "checkpoints_phase3/savi_final.pth"
    )
    print("訓練完了。")


if __name__ == "__main__":
    main()