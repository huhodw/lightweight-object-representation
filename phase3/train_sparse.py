"""
train_sparse.py

画像を k フレームに 1 回だけ入力する「間引き訓練」の訓練スクリプト。
Phase 3 の train_savi.py をもとに、次の点だけを変えている。
  1. モデル   : model_sparse.SparseSAViModel（フレームのループだけが違う。部品とパラメータは SAViModel と同じ）
  2. 保存先   : checkpoints_sparse_k{k}/（Phase 3 の checkpoints_phase3/ には書き込まない）
  3. 乱数     : 重みの初期値・データの順番・初期スロットの乱数をシードで固定し、チェックポイントに保存
  4. 再開     : 途中で止まっても、終わったバッチを飛ばして、同じ順番・同じ総ステップ数で続きから再開する
  5. 記録     : 画像を入力したフレームと、入力しないフレームの再構成誤差を分けて CSV に記録
  6. 検証     : 10 エポックごとに検証データで FG-ARI を測って記録（モデルの選択には使わない）
訓練条件（エポック数、バッチサイズ、学習率とスケジューラ、勾配クリップ、6 フレーム、損失）は Phase 3 と同じ。

置き場所: savi_phase3 フォルダの直下（model_sparse.py と同じ場所）
実行例:
  python train_sparse.py --max-steps 100    # 試運転（checkpoints_sparse_k3_trial に保存）
  python train_sparse.py                    # 本番（checkpoints_sparse_k3 に保存）
"""
import sys
import os
import argparse
import csv
import math
import random
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "savi-pytorch"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from params import SAViParams
from movi_dataset import MoviDataset
from evaluate_movi import adjusted_rand_index
from model_sparse import SparseSAViModel

GRAD_CLIP = 0.05              # train_savi.py と同じ
SAVE_EVERY_STEPS = 100        # train_savi.py と同じ（再開用の latest.pth を上書きする間隔）
SNAPSHOT_EVERY_EPOCHS = 10    # train_savi.py と同じ（記録用のチェックポイントを残す間隔）
LOG_EVERY_STEPS = 10
SPLIT_SEED = 42               # train_savi.py / evaluate_movi.py と同じ分割にするため、変えない
# 予備実験（Phase 3 のモデル＋コピー、6 フレーム）での、画像を入力しないフレームの FG-ARI
COPY_BASELINE_T6 = {2: 0.5576, 3: 0.5085, 6: 0.4141}

TRAIN_HEADER = ["global_step", "epoch", "loss", "mse_observed", "mse_unobserved", "lr", "data_sec", "compute_sec"]
VAL_HEADER = ["global_step", "epoch", "n_videos", "fg_ari_all", "fg_ari_observed", "fg_ari_unobserved",
              "mse_observed", "mse_unobserved"]


def get_lr_scheduler(optimizer, warmup_steps, total_steps, decay_steps_pct):
    """train_savi.py と同じ（warmup で上げる → 一定 → 最後の 20% で 0 まで直線的に下げる）"""
    decay_start = int(total_steps * (1.0 - decay_steps_pct))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step >= decay_start:
            progress = (step - decay_start) / max(1, total_steps - decay_start)
            return max(0.0, 1.0 - progress)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


class EpochPermutationSampler(Sampler):
    """エポックごとに (シード, エポック番号) だけで決まる順番でデータを並べる。
    同じシード・同じエポックなら必ず同じ順番になるので、途中から再開するときに
    終わったバッチを正確に飛ばせる。"""

    def __init__(self, n, seed):
        self.n = n
        self.seed = seed
        self.epoch = 0
        self.skip = 0

    def set_epoch(self, epoch, skip_samples=0):
        self.epoch = epoch
        self.skip = skip_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 100003 + self.epoch)
        perm = torch.randperm(self.n, generator=g).tolist()
        return iter(perm[self.skip:])

    def __len__(self):
        return self.n - self.skip


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def make_config(args, params, train_size, val_size):
    """訓練の設定。チェックポイントに保存し、再開のときに一致するかを確かめる。"""
    return {
        "observe_every": args.observe_every,
        "seed": args.seed,
        "trial": args.max_steps is not None,
        "batch_size": params.batch_size,
        "lr": params.lr,
        "max_epochs": params.max_epochs,
        "seq_len": params.seq_len,
        "num_slots": params.num_slots,
        "num_iterations": params.num_iterations,
        "warmup_steps_pct": params.warmup_steps_pct,
        "decay_steps_pct": params.decay_steps_pct,
        "grad_clip": GRAD_CLIP,
        "train_size": train_size,
        "val_size": val_size,
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, batches_done, global_step, loss, config, best_val):
    """重み・optimizer・scheduler・乱数の状態・設定をまとめて保存する。
    一度 .tmp に書いてから置き換えるので、保存中に止まっても前のファイルが壊れない。"""
    state = {
        "epoch": epoch,                  # 次に続けるエポック（0 始まり）
        "batches_done": batches_done,    # そのエポックで終わったバッチ数
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "seed": config["seed"],
        "config": config,
        "best_val_fg_ari": best_val,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    last_err = None
    for _ in range(10):                  # Windows でウイルス対策ソフトが一瞬ファイルを掴むことがあるので再試行
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(1.0)
    raise last_err


def load_for_resume(path, model, optimizer, scheduler, config):
    """latest.pth があれば読み込んで、(エポック, 終わったバッチ数, global_step, best) を返す。無ければ None。"""
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)   # 乱数の状態は CPU に置く必要がある
    saved = ck.get("config")
    if saved != config:
        keys = sorted(set(saved or {}) | set(config))
        diffs = {key: ((saved or {}).get(key), config.get(key)) for key in keys
                 if (saved or {}).get(key) != config.get(key)}
        raise SystemExit(
            f"止めました: {path} は設定が違う訓練のものです（保存時, 今）= {diffs}\n"
            "別の訓練の続きにしないため、保存先を変えるか、そのフォルダを確認してください。"
        )
    model.load_state_dict(ck["model_state_dict"])
    optimizer.load_state_dict(ck["optimizer_state_dict"])
    scheduler.load_state_dict(ck["scheduler_state_dict"])
    torch.set_rng_state(ck["rng_cpu"])
    if ck["rng_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["rng_cuda"])
    np.random.set_state(ck["rng_numpy"])
    random.setstate(ck["rng_python"])
    return ck["epoch"], ck["batches_done"], ck["global_step"], ck.get("best_val_fg_ari")


def init_csv(path, header, resume_step):
    """CSV を用意する。再開のときは、再開するステップより後の行を消す（同じステップが2回記録されないように）。"""
    rows = []
    if resume_step is not None and os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if int(r["global_step"]) <= resume_step]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def append_csv(path, header, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=header, extrasaction="ignore").writerow(row)


@torch.no_grad()
def run_validation(model, val_dataset, n_videos, device, T):
    """検証データの先頭 n_videos 本で、画像を入力したフレーム・入力しないフレームの FG-ARI と再構成誤差を測る。
    初期スロットの乱数は動画ごとに固定（sparse_observation_eval.py の既定と同じ値）。
    専用の乱数生成器を使うので、訓練側の乱数の流れには影響しない。"""
    was_training = model.training
    model.eval()
    n = min(n_videos, len(val_dataset))
    loader = DataLoader(Subset(val_dataset, range(n)), batch_size=1, shuffle=False,
                        num_workers=0, generator=torch.Generator())
    ari_obs, ari_un, mse_obs, mse_un = [], [], [], []
    t0 = time.perf_counter()
    for vid_idx, batch in enumerate(loader):
        video = batch["video"][:, :T].to(device)
        seg = batch["segmentations"][:, :T].to(device)
        gt = seg.squeeze(0).permute(0, 3, 1, 2)                 # [T, 1, H, W]（evaluate_movi.py と同じ）
        g = torch.Generator().manual_seed(vid_idx)
        noise = torch.randn((1, model.num_slots, model.slot_size), generator=g).to(device)
        recon, _, masks, _, observed = model(video, noise=noise)
        pm = masks.squeeze(0).permute(1, 0, 2, 3, 4)            # [T, K, 1, H, W]
        ari = adjusted_rand_index(gt, pm).tolist()
        mse = ((recon - video) ** 2).mean(dim=(0, 2, 3, 4)).tolist()
        for t in range(T):
            if observed[t]:
                ari_obs.append(ari[t])
                mse_obs.append(mse[t])
            else:
                ari_un.append(ari[t])
                mse_un.append(mse[t])
    model.train(was_training)
    return {
        "n_videos": n,
        "fg_ari_all": mean(ari_obs + ari_un),
        "fg_ari_observed": mean(ari_obs),
        "fg_ari_unobserved": mean(ari_un),
        "mse_observed": mean(mse_obs),
        "mse_unobserved": mean(mse_un),
        "sec": time.perf_counter() - t0,
    }


def print_validation(v, k, T):
    print(f"  検証（{v['n_videos']}本）: FG-ARI 全体 {v['fg_ari_all']:.4f} | 入力あり {v['fg_ari_observed']:.4f} | "
          f"入力なし {v['fg_ari_unobserved']:.4f} | {v['sec']:.0f}秒")
    if T == 6 and k in COPY_BASELINE_T6:
        print(f"  （参考: 入力なしフレームの目標 = Phase 3 のモデル＋コピー {COPY_BASELINE_T6[k]:.4f}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observe-every", type=int, default=3,
                    help="k: 何フレームに 1 回画像を入力するか（既定 3）")
    ap.add_argument("--seed", type=int, default=0, help="乱数のシード（既定 0）")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="試運転: このステップ数で止める。保存先は checkpoints_sparse_k{k}_trial")
    ap.add_argument("--ckpt-dir", default=None, help="保存先（savi_phase3 からの相対パス）")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="データ読み込みの並列数（省略時は params.py の値 = Phase 3 と同じ）")
    ap.add_argument("--val-every-epochs", type=int, default=10, help="何エポックごとに検証するか")
    ap.add_argument("--val-videos", type=int, default=None,
                    help="検証に使う本数（省略時: 本番は全部、試運転は 8 本）")
    args = ap.parse_args()

    params = SAViParams()
    T = params.seq_len
    k = args.observe_every
    if not 1 <= k <= T:
        raise SystemExit(f"--observe-every は 1〜{T} にしてください")
    trial = args.max_steps is not None

    ckpt_rel = args.ckpt_dir or (f"checkpoints_sparse_k{k}" + ("_trial" if trial else ""))
    ckpt_dir = os.path.abspath(os.path.join(_HERE, ckpt_rel))
    if "phase3" in os.path.basename(ckpt_dir).lower():
        raise SystemExit("止めました: 保存先に Phase 3 のフォルダは使えません（比較相手のモデルを上書きしないため）")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 乱数の固定 =====
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ===== データ（train_savi.py と同じ分割）=====
    full_dataset = MoviDataset(os.path.join(_HERE, params.data_root))
    total = len(full_dataset)
    val_size = min(params.num_val_videos or 250, total // 10)
    train_size = total - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )
    num_workers = params.num_workers if args.num_workers is None else args.num_workers
    sampler = EpochPermutationSampler(train_size, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params.batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        generator=torch.Generator().manual_seed(args.seed),   # 訓練側の乱数の流れを乱さないための専用の生成器
    )

    # ===== モデル =====
    model = SparseSAViModel(
        resolution=params.resolution,
        num_slots=params.num_slots,
        num_iterations=params.num_iterations,
        observe_every=k,
    ).to(device)

    # model_sparse.py の forward が書かれているか（乱数は使わない）
    expected_obs = [t % k == 0 for t in range(T)]
    try:
        with torch.no_grad():
            dummy = torch.zeros((1, T, 3, *params.resolution), device=device)
            dummy_noise = torch.zeros((1, model.num_slots, model.slot_size), device=device)
            obs_check = model(dummy, noise=dummy_noise)[4]
    except NotImplementedError:
        raise SystemExit("止めました: model_sparse.py の forward がまだ書かれていません。"
                         "先に書いて、test_model_sparse.py をすべて合格にしてください。")
    if [bool(x) for x in obs_check] != expected_obs:
        raise SystemExit(f"止めました: 画像を入力するフレームが想定と違います（{obs_check}）。"
                         "test_model_sparse.py を先にすべて合格にしてください。")

    # ===== 最適化（train_savi.py と同じ）=====
    optimizer = Adam(model.parameters(), lr=params.lr)
    steps_per_epoch = math.ceil(train_size / params.batch_size)
    total_steps = steps_per_epoch * params.max_epochs
    warmup_steps = int(total_steps * params.warmup_steps_pct)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps, params.decay_steps_pct)

    # ===== 再開 =====
    config = make_config(args, params, train_size, val_size)
    latest_path = os.path.join(ckpt_dir, "latest.pth")
    resumed = load_for_resume(latest_path, model, optimizer, scheduler, config)
    if resumed is not None:
        start_epoch, batches_done, global_step, best_val = resumed
    else:
        start_epoch, batches_done, global_step, best_val = 0, 0, 0, None
    if batches_done >= steps_per_epoch:
        start_epoch, batches_done = start_epoch + 1, 0

    train_log = os.path.join(ckpt_dir, "train_log.csv")
    val_log = os.path.join(ckpt_dir, "val_log.csv")
    init_csv(train_log, TRAIN_HEADER, global_step if resumed else None)
    init_csv(val_log, VAL_HEADER, global_step if resumed else None)

    print(f"使用デバイス: {device}")
    print(f"間引き訓練: k={k}（画像を入力するフレーム {[t for t in range(T) if expected_obs[t]]} / {T}、"
          f"エンコーダの呼び出し {sum(expected_obs)}/{T}）")
    print(f"{'試運転（' + str(args.max_steps) + ' ステップで停止）' if trial else '本番'} / 保存先: {ckpt_dir}")
    print(f"シード: {args.seed} / データ読み込みの並列数: {num_workers}")
    print(f"訓練データ: {train_size}本 / 検証データ: {val_size}本（train_savi.py と同じ分割）")
    print(f"1エポックのステップ数: {steps_per_epoch} / 総ステップ数: {total_steps} / warmup: {warmup_steps}")
    if resumed is not None:
        print(f"再開: step {global_step} まで完了済み → エポック {start_epoch + 1} の "
              f"{batches_done + 1} バッチ目から続けます")
    else:
        print("新規に開始します")
    print("=" * 78)

    if trial and global_step >= args.max_steps:
        print(f"この保存先はすでに step {global_step} まで進んでいます。--max-steps を増やしてください。")
        return

    # ===== 訓練ループ =====
    model.train()
    obs_idx = [t for t in range(T) if expected_obs[t]]
    un_idx = [t for t in range(T) if not expected_obs[t]]
    acc = {"loss": [], "obs": [], "un": [], "data": [], "comp": []}
    timings = []
    latest_loss = float("nan")
    stop = False

    for epoch in range(start_epoch, params.max_epochs):
        skip = batches_done if epoch == start_epoch else 0
        sampler.set_epoch(epoch, skip * params.batch_size)
        epoch_losses = []
        t_prev = time.perf_counter()

        for i, batch in enumerate(train_loader):
            b_done = skip + i + 1
            video = batch["video"][:, :T].to(device, non_blocking=True)   # 先頭 T フレームだけ使う（Phase 3 と同じ）
            t_data = time.perf_counter()

            optimizer.zero_grad()
            recon, _, _, _, _ = model(video)
            loss = F.mse_loss(recon, video)          # train_savi.py と同じ損失（全フレームの再構成誤差の平均）
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            lr_used = optimizer.param_groups[0]["lr"]   # このステップで実際に使った学習率
            optimizer.step()
            scheduler.step()
            global_step += 1

            with torch.no_grad():                    # 記録用：入力あり／なしのフレームに分けた誤差
                fm = ((recon.detach() - video) ** 2).mean(dim=(0, 2, 3, 4))
                mse_obs = fm[obs_idx].mean().item()
                mse_un = fm[un_idx].mean().item() if un_idx else float("nan")
            latest_loss = loss.item()
            t_end = time.perf_counter()

            # CSV には毎ステップ記録する（再開しても同じ行になる）。画面には 10 ステップの平均を出す
            append_csv(train_log, TRAIN_HEADER, {
                "global_step": global_step, "epoch": epoch + 1, "loss": latest_loss,
                "mse_observed": mse_obs, "mse_unobserved": mse_un, "lr": lr_used,
                "data_sec": t_data - t_prev, "compute_sec": t_end - t_data})
            epoch_losses.append(latest_loss)
            acc["loss"].append(latest_loss)
            acc["obs"].append(mse_obs)
            acc["un"].append(mse_un)
            acc["data"].append(t_data - t_prev)
            acc["comp"].append(t_end - t_data)
            timings.append((t_data - t_prev, t_end - t_data))
            t_prev = t_end

            if global_step % LOG_EVERY_STEPS == 0:
                print(f"step {global_step:6d} | epoch {epoch + 1:3d} | loss {mean(acc['loss']):.5f} | "
                      f"入力あり {mean(acc['obs']):.5f} | 入力なし {mean(acc['un']):.5f} | lr {lr_used:.2e} | "
                      f"読込 {mean(acc['data']):.2f}s 計算 {mean(acc['comp']):.2f}s /step")
                acc = {key: [] for key in acc}

            if global_step % SAVE_EVERY_STEPS == 0:
                save_checkpoint(latest_path, model, optimizer, scheduler, epoch, b_done, global_step,
                                latest_loss, config, best_val)

            if trial and global_step >= args.max_steps:
                save_checkpoint(latest_path, model, optimizer, scheduler, epoch, b_done, global_step,
                                latest_loss, config, best_val)
                stop = True
                break

        if stop:
            break

        # ===== エポック終了 =====
        print(f"epoch {epoch + 1:4d} 完了 | 平均損失: {mean(epoch_losses):.5f}")
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch + 1, 0, global_step,
                        latest_loss, config, best_val)
        if (epoch + 1) % SNAPSHOT_EVERY_EPOCHS == 0:
            save_checkpoint(os.path.join(ckpt_dir, f"epoch{epoch + 1:04d}.pth"), model, optimizer, scheduler,
                            epoch + 1, 0, global_step, latest_loss, config, best_val)
        if (epoch + 1) % args.val_every_epochs == 0 or epoch + 1 == params.max_epochs:
            v = run_validation(model, val_dataset, args.val_videos or val_size, device, T)
            print_validation(v, k, T)
            append_csv(val_log, VAL_HEADER, {"global_step": global_step, "epoch": epoch + 1, **v})
            score = v["fg_ari_unobserved"] if un_idx else v["fg_ari_observed"]
            if best_val is None or score > best_val:
                best_val = score
                save_checkpoint(os.path.join(ckpt_dir, "best_val.pth"), model, optimizer, scheduler,
                                epoch + 1, 0, global_step, latest_loss, config, best_val)
                print("  検証の最高値を更新 → best_val.pth（参考用。比較には final.pth を使う）")
            save_checkpoint(latest_path, model, optimizer, scheduler, epoch + 1, 0, global_step,
                            latest_loss, config, best_val)

    if stop:
        # ===== 試運転のまとめ =====
        print("=" * 78)
        print(f"試運転: step {global_step} で停止（再開用に {latest_path} を保存）")
        n_val = args.val_videos or 8
        v = run_validation(model, val_dataset, n_val, device, T)
        print_validation(v, k, T)
        print("  ※ 試運転なので FG-ARI はまだ意味のある値ではありません（検証のコードが動くかの確認）")
        steady = timings[3:] if len(timings) > 3 else timings
        d = mean([x[0] for x in steady])
        c = mean([x[1] for x in steady])
        sec_per_step = d + c
        n_val_checks = math.ceil(params.max_epochs / args.val_every_epochs)
        val_hours = v["sec"] / max(1, v["n_videos"]) * val_size * n_val_checks / 3600
        train_hours = total_steps * sec_per_step / 3600
        print(f"1ステップの平均（最初の3ステップを除く）: 読込 {d:.2f}秒 + 計算 {c:.2f}秒 = {sec_per_step:.2f}秒")
        if device.type == "cuda":
            print(f"GPU メモリの最大使用量: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        print(f"本番（{total_steps} ステップ）の推定時間: 訓練 約 {train_hours:.1f} 時間 + 検証 約 {val_hours:.1f} 時間"
              f" = 約 {(train_hours + val_hours) / 24:.1f} 日")
        return

    # ===== 訓練完了 =====
    save_checkpoint(os.path.join(ckpt_dir, "final.pth"), model, optimizer, scheduler,
                    params.max_epochs, 0, global_step, latest_loss, config, best_val)
    print(f"訓練完了（総ステップ数 {global_step}）。比較には {os.path.join(ckpt_dir, 'final.pth')} を使います。")


if __name__ == "__main__":
    main()
