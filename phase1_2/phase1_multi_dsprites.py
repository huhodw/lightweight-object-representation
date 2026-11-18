# phase1_multi_dsprites.py
"""
Phase 1: multi_dsprites での Slot Attention 再現。
64×64画像対応のモデル定義 + 訓練ループ + FG-ARI評価。

依存モジュール:
  - multi_dsprites_dataset.py (MultiDSpritesDataset)
  - metrics.py (compute_fg_ari)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
import argparse
from datetime import datetime


# ============================================================
# モデル定義（既存と同じ、変更なし）
# ============================================================
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_dim, resolution):
        super().__init__()
        self.embedding = nn.Linear(4, hidden_dim, bias=True)
        self.register_buffer('grid', self._build_grid(resolution))
    
    def _build_grid(self, resolution):
        H, W = resolution
        ranges = [torch.linspace(0.0, 1.0, steps=r) for r in resolution]
        grid_y, grid_x = torch.meshgrid(ranges[0], ranges[1], indexing='ij')
        grid = torch.stack([grid_x, 1.0 - grid_x, grid_y, 1.0 - grid_y], dim=-1)
        return grid.unsqueeze(0)
    
    def forward(self, inputs):
        pos_emb = self.embedding(self.grid)
        pos_emb = pos_emb.permute(0, 3, 1, 2)
        return inputs + pos_emb


class Encoder(nn.Module):
    def __init__(self, resolution, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.pos_embed = SoftPositionEmbed(hidden_dim, resolution)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pos_embed(x)
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)
        x = self.mlp(x)
        return x


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
    
    def forward(self, inputs, return_attn=False):
        B, N, D = inputs.shape
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        inputs = self.norm_input(inputs)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        attn = None
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn = attn_logits.softmax(dim=1) + self.eps
            attn_weights = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bkn,bnd->bkd', attn_weights, v)
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        if return_attn:
            return slots, attn
        return slots


class Decoder64(nn.Module):
    """64×64出力のDecoder（transpose conv 3層）"""
    def __init__(self, hidden_dim=64, decoder_init_resolution=(8, 8)):
        super().__init__()
        self.decoder_init_resolution = decoder_init_resolution
        self.pos_embed = SoftPositionEmbed(hidden_dim, decoder_init_resolution)
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv_out = nn.Conv2d(hidden_dim, 4, kernel_size=3, padding=1)
    
    def forward(self, slots):
        B, K, D = slots.shape
        x = slots.reshape(B * K, D)
        H0, W0 = self.decoder_init_resolution
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = x.expand(-1, -1, H0, W0)
        x = self.pos_embed(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv_out(x)
        x = x.reshape(B, K, 4, x.shape[-2], x.shape[-1])
        recons = x[:, :, :3, :, :]
        masks = x[:, :, 3:4, :, :]
        return recons, masks


class SlotAttentionAutoEncoder64(nn.Module):
    def __init__(self, resolution=(64, 64), num_slots=7, num_iterations=3, hidden_dim=64):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.encoder = Encoder(resolution=resolution, hidden_dim=hidden_dim)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=hidden_dim, iters=num_iterations)
        self.decoder = Decoder64(hidden_dim=hidden_dim, decoder_init_resolution=(8, 8))
    
    def forward(self, image, return_slots=False):
        features = self.encoder(image)
        slots = self.slot_attention(features)
        recons, masks = self.decoder(slots)
        masks_softmax = masks.softmax(dim=1)
        recon_combined = (recons * masks_softmax).sum(dim=1)
        if return_slots:
            return recon_combined, recons, masks_softmax, slots
        return recon_combined


# ============================================================
# 学習率スケジューラ（Warmup + Exponential Decay）
# ============================================================
def get_lr(step, base_lr, warmup_steps, decay_steps, decay_rate):
    """Slot Attention論文のスケジュール"""
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        return base_lr * (decay_rate ** ((step - warmup_steps) / decay_steps))


# ============================================================
# 訓練ステップ
# ============================================================
def train_step(model, batch, optimizer, grad_clip, device):
    """1バッチを処理"""
    model.train()
    images = batch['image'].to(device, non_blocking=True)
    
    optimizer.zero_grad()
    recon = model(images)
    loss = F.mse_loss(recon, images)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()
    
    return loss.item()


# ============================================================
# 検証ループ（FG-ARI計算）
# ============================================================
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None):
    """検証セットでMSEとFG-ARIを計算"""
    from metrics import compute_fg_ari
    
    model.eval()
    losses = []
    aris = []
    
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        
        images = batch['image'].to(device, non_blocking=True)
        gt_masks = batch['mask'].to(device, non_blocking=True)
        
        # モデル予測
        recon, recons, pred_masks, slots = model(images, return_slots=True)
        
        # MSE
        loss = F.mse_loss(recon, images).item()
        losses.append(loss)
        
        # FG-ARI
        # pred_masks: (B, K, 1, H, W) -> compute_fg_ariが処理
        ari = compute_fg_ari(pred_masks, gt_masks)
        aris.append(ari)
    
    model.train()
    
    return {
        'val_loss': np.mean(losses),
        'val_fg_ari': np.mean(aris),
        'val_fg_ari_std': np.std(aris),
    }


# ============================================================
# 訓練ループ本体
# ============================================================
def train_phase1(
    num_steps=500000,
    batch_size=64,
    base_lr=4e-4,
    warmup_steps=10000,
    decay_steps=100000,
    decay_rate=0.5,
    grad_clip=1.0,
    num_slots=7,
    num_iterations=3,
    hidden_dim=64,
    eval_every=5000,
    save_every=5000,
    log_every=100,
    val_max_batches=20,  # 検証時、何バッチまで使うか（20 × 64 = 1280サンプル）
    npz_path="data/multi_dsprites_70k.npz",
    save_dir="./checkpoints_phase1",
    resume_from=None,
):
    """Phase 1の訓練を実行"""
    # ============================================
    # セットアップ
    # ============================================
    print("=" * 60)
    print("Phase 1: multi_dsprites Slot Attention 訓練")
    print(f"  開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[デバイス] {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # 保存ディレクトリ
    os.makedirs(save_dir, exist_ok=True)
    
    # データ
    print(f"\n[データセット] {npz_path}")
    from multi_dsprites_dataset import MultiDSpritesDataset
    train_dataset = MultiDSpritesDataset(npz_path=npz_path, split='train')
    val_dataset = MultiDSpritesDataset(npz_path=npz_path, split='val')
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    print(f"  train batches: {len(train_loader)}")
    print(f"  val batches:   {len(val_loader)}")
    
    # モデル
    print(f"\n[モデル]")
    model = SlotAttentionAutoEncoder64(
        resolution=(64, 64),
        num_slots=num_slots,
        num_iterations=num_iterations,
        hidden_dim=hidden_dim,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  パラメータ数: {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"  スロット数:   {num_slots}")
    print(f"  反復回数:     {num_iterations}")
    
    # 最適化
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    
    # 履歴
    history = {
        'train_loss': [],     # (step, loss) のリスト
        'val_loss': [],       # (step, loss)
        'val_fg_ari': [],     # (step, ari)
        'val_fg_ari_std': [],
        'lr': [],
    }
    
    # ============================================
    # チェックポイントから再開
    # ============================================
    start_step = 0
    best_ari = -1.0
    if resume_from is not None and os.path.exists(resume_from):
        print(f"\n[再開] {resume_from}")
        ckpt = torch.load(resume_from)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        best_ari = ckpt.get('best_ari', -1.0)
        # 履歴も復元
        if 'history' in ckpt:
            history = ckpt['history']
        print(f"  ステップ {start_step} から再開、ベストARI = {best_ari:.4f}")
    
    # ============================================
    # 訓練ループ
    # ============================================
    print(f"\n[訓練開始] {num_steps} ステップ\n")
    
    train_iter = iter(train_loader)
    start_time = time.time()
    recent_losses = []
    
    for step in range(start_step, num_steps):
        # データ取得
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # 学習率
        lr = get_lr(step, base_lr, warmup_steps, decay_steps, decay_rate)
        for g in optimizer.param_groups:
            g['lr'] = lr
        
        # 訓練ステップ
        loss = train_step(model, batch, optimizer, grad_clip, device)
        recent_losses.append(loss)
        
        # ログ
        if (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            avg_loss = np.mean(recent_losses[-log_every:])
            steps_per_sec = (step + 1 - start_step) / elapsed
            eta_sec = (num_steps - step - 1) / steps_per_sec
            eta_hr = eta_sec / 3600
            
            print(f"Step {step+1:6d}/{num_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"{steps_per_sec:.1f} steps/s | "
                  f"ETA: {eta_hr:.1f}h")
            
            history['train_loss'].append((step + 1, avg_loss))
            history['lr'].append((step + 1, lr))
        
        # 評価
        if (step + 1) % eval_every == 0:
            print(f"\n[評価] step {step+1}")
            metrics = evaluate(model, val_loader, device, max_batches=val_max_batches)
            print(f"  val_loss   = {metrics['val_loss']:.4f}")
            print(f"  val_fg_ari = {metrics['val_fg_ari']:.4f} ± {metrics['val_fg_ari_std']:.4f}")
            
            history['val_loss'].append((step + 1, metrics['val_loss']))
            history['val_fg_ari'].append((step + 1, metrics['val_fg_ari']))
            history['val_fg_ari_std'].append((step + 1, metrics['val_fg_ari_std']))
            
            # ベストモデル保存
            if metrics['val_fg_ari'] > best_ari:
                best_ari = metrics['val_fg_ari']
                best_path = os.path.join(save_dir, 'best.pth')
                torch.save({
                    'step': step + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_ari': best_ari,
                    'metrics': metrics,
                    'history': history,
                }, best_path)
                print(f"  💾 ベストモデル更新: FG-ARI = {best_ari:.4f}")
            
            print()
        
        # 定期チェックポイント
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(save_dir, f'step_{step+1}.pth')
            torch.save({
                'step': step + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_ari': best_ari,
                'history': history,
            }, ckpt_path)
            
            # 古いチェックポイント削除（3つ前を消す）
            old_step = step + 1 - save_every * 3
            old_ckpt = os.path.join(save_dir, f'step_{old_step}.pth')
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
            
            # 履歴をJSONで保存（毎回上書き）
            history_path = os.path.join(save_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
    
    # ============================================
    # 訓練完了
    # ============================================
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"訓練完了")
    print(f"  総時間: {total_time/3600:.2f} 時間")
    print(f"  ベスト FG-ARI: {best_ari:.4f}")
    print("=" * 60)
    
    return model, history


# ============================================================
# メイン部分
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_phase1')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    train_phase1(
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        base_lr=args.lr,
        eval_every=args.eval_every,
        save_every=args.save_every,
        save_dir=args.save_dir,
        resume_from=args.resume,
    )