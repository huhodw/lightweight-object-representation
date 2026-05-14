# phase2_dinov2_slot.py
"""
Phase 2: DINOv2 ViT-B + Slot Attention の最小動作モデル。

Phase 1 からの変更点:
- CNN Encoder → DINOv2 ViT-B (frozen)
- 入力サイズ 64×64 → 224×224 へリサイズ
- ViT 出力 (256パッチ × 768次元) を Slot Attention に投入
- Slot Attention 内部次元: 64 → 384 (768/2、ViT特徴と整合性)
- Decoder は Phase 1 と同じ（8→16→32→64出力）

DINOv2 仕様（公式確認済み）:
- patch_size = 14
- hidden_size = 768
- 入力 224×224 → パッチ数 (224/14)² = 256
- 出力 last_hidden_state: (B, 1+256, 768) = (B, 257, 768)
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

from transformers import AutoModel


# ============================================================
# Phase 1 から流用（変更なし）
# ============================================================
class SoftPositionEmbed(nn.Module):
    """座標情報を特徴マップに足し込む"""
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
    """64×64出力のDecoder（Phase 1 と同じ）"""
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


# ============================================================
# DINOv2 Encoder ラッパー
# ============================================================
class DINOv2Encoder(nn.Module):
    """
    DINOv2 ViT-B を Encoder として使うラッパー。
    
    入力: (B, 3, 224, 224)
    出力: (B, num_patches=256, 768) - パッチトークンのみ（CLS除く）
    """
    def __init__(self, model_name='facebook/dinov2-base', freeze=True):
        super().__init__()
        self.dinov2 = AutoModel.from_pretrained(model_name)
        
        self.frozen = freeze
        if freeze:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        
        self.feature_dim = self.dinov2.config.hidden_size  # 768
        self.patch_size = self.dinov2.config.patch_size     # 14
    
    def train(self, mode=True):
        """凍結時は dinov2 を常に eval モードに保つ"""
        super().train(mode)
        if self.frozen:
            self.dinov2.eval()
        return self
    
    def forward(self, x):
        # 形状チェック
        H, W = x.shape[-2], x.shape[-1]
        assert H % self.patch_size == 0, \
            f"画像高さ {H} は patch_size {self.patch_size} の倍数である必要あり"
        assert W % self.patch_size == 0, \
            f"画像幅 {W} は patch_size {self.patch_size} の倍数である必要あり"
        
        if self.frozen:
            with torch.no_grad():
                outputs = self.dinov2(pixel_values=x)
        else:
            outputs = self.dinov2(pixel_values=x)
        
        # CLSトークンを除いてパッチトークンだけ返す
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        return patch_tokens


# ============================================================
# 全体モデル: DINOv2 + Slot Attention + Decoder
# ============================================================
class SlotAttentionDINOv2(nn.Module):
    def __init__(
        self,
        num_slots=7,
        slot_dim=384,
        decoder_dim=64,
        num_iterations=3,
        freeze_vit=True,
        input_size=64,
        vit_input_size=224,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.vit_input_size = vit_input_size
        
        # Encoder（DINOv2）
        self.encoder = DINOv2Encoder(
            model_name='facebook/dinov2-base',
            freeze=freeze_vit
        )
        vit_dim = self.encoder.feature_dim
        
        # ViT出力 → Slot Attention 次元へ投影
        self.vit_to_slot = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )
        
        # Slot Attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=slot_dim,
            iters=num_iterations,
            hidden_dim=128,
        )
        
        # Slot Attention 出力 → Decoder 入力次元へ
        self.slot_to_decoder = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, decoder_dim),
        )
        
        # Decoder
        self.decoder = Decoder64(
            hidden_dim=decoder_dim,
            decoder_init_resolution=(8, 8)
        )
        
        self.num_slots = num_slots
    
    def forward(self, image, return_slots=False):
        B = image.shape[0]
        
        # 1. 224×224 にリサイズ
        if image.shape[-1] != self.vit_input_size:
            image_vit = F.interpolate(
                image, size=(self.vit_input_size, self.vit_input_size),
                mode='bilinear', align_corners=False
            )
        else:
            image_vit = image
        
        # 2. DINOv2 で特徴抽出
        features = self.encoder(image_vit)  # (B, 256, 768)
        
        # 3. Slot Attention 次元へ投影
        features = self.vit_to_slot(features)  # (B, 256, 384)
        
        # 4. Slot Attention
        slots = self.slot_attention(features)  # (B, K, 384)
        
        # 5. Decoder 次元へ投影
        slots_for_decoder = self.slot_to_decoder(slots)  # (B, K, 64)
        
        # 6. Decoder で再構成
        recons, masks = self.decoder(slots_for_decoder)
        
        # 7. α合成
        masks_softmax = masks.softmax(dim=1)
        recon_combined = (recons * masks_softmax).sum(dim=1)
        
        if return_slots:
            return recon_combined, recons, masks_softmax, slots
        return recon_combined


# ============================================================
# ImageNet 正規化
# ============================================================
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_for_dinov2(images, device):
    """[0, 1] 範囲の画像を ImageNet 統計で正規化"""
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (images - mean) / std


# ============================================================
# 学習率スケジュール（Phase 1 と同じ）
# ============================================================
def get_lr(step, base_lr, warmup_steps, decay_steps, decay_rate):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        return base_lr * (decay_rate ** ((step - warmup_steps) / decay_steps))


# ============================================================
# 訓練ステップ
# ============================================================
def train_step(model, batch, optimizer, grad_clip, device):
    model.train()
    
    images = batch['image'].to(device, non_blocking=True)
    images_normalized = normalize_for_dinov2(images, device)
    
    optimizer.zero_grad()
    recon = model(images_normalized)
    loss = F.mse_loss(recon, images_normalized)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()
    
    return loss.item()


# ============================================================
# 検証ループ
# ============================================================
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None):
    from metrics import compute_fg_ari
    
    model.eval()
    losses = []
    aris = []
    
    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break
        
        images = batch['image'].to(device, non_blocking=True)
        gt_masks = batch['mask'].to(device, non_blocking=True)
        
        images_normalized = normalize_for_dinov2(images, device)
        
        recon, recons, pred_masks, slots = model(images_normalized, return_slots=True)
        
        loss = F.mse_loss(recon, images_normalized).item()
        losses.append(loss)
        
        ari = compute_fg_ari(pred_masks, gt_masks)
        aris.append(ari)
    
    return {
        'val_loss': np.mean(losses),
        'val_fg_ari': np.mean(aris),
        'val_fg_ari_std': np.std(aris),
    }


# ============================================================
# 訓練ループ本体
# ============================================================
def train_phase2(
    num_steps=10000,
    batch_size=64,
    base_lr=4e-4,
    warmup_steps=1000,
    decay_steps=10000,
    decay_rate=0.5,
    grad_clip=1.0,
    num_slots=7,
    slot_dim=384,
    decoder_dim=64,
    num_iterations=3,
    eval_every=1000,
    save_every=2000,
    log_every=100,
    val_max_batches=20,
    npz_path="data/multi_dsprites_70k.npz",
    save_dir="./checkpoints_phase2",
    resume_from=None,
):
    print("=" * 60)
    print("Phase 2: DINOv2 + Slot Attention 訓練")
    print(f"  開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[デバイス] {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
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
    print(f"\n[モデル] DINOv2 + Slot Attention")
    print(f"  DINOv2 を読み込み中（初回はダウンロードあり）...")
    model = SlotAttentionDINOv2(
        num_slots=num_slots,
        slot_dim=slot_dim,
        decoder_dim=decoder_dim,
        num_iterations=num_iterations,
        freeze_vit=True,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  全パラメータ:       {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"  訓練可能パラメータ: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    
    # 訓練可能パラメータだけ optimizer に渡す
    trainable_param_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_param_list, lr=base_lr)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_fg_ari': [],
        'val_fg_ari_std': [],
        'lr': [],
    }
    
    start_step = 0
    best_ari = -1.0
    if resume_from is not None and os.path.exists(resume_from):
        print(f"\n[再開] {resume_from}")
        ckpt = torch.load(resume_from)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        best_ari = ckpt.get('best_ari', -1.0)
        if 'history' in ckpt:
            history = ckpt['history']
        print(f"  ステップ {start_step} から再開、ベストARI = {best_ari:.4f}")
    
    print(f"\n[訓練開始] {num_steps} ステップ\n")
    
    train_iter = iter(train_loader)
    start_time = time.time()
    recent_losses = []
    
    for step in range(start_step, num_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        lr = get_lr(step, base_lr, warmup_steps, decay_steps, decay_rate)
        for g in optimizer.param_groups:
            g['lr'] = lr
        
        loss = train_step(model, batch, optimizer, grad_clip, device)
        recent_losses.append(loss)
        
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
        
        if (step + 1) % eval_every == 0:
            print(f"\n[評価] step {step+1}")
            metrics = evaluate(model, val_loader, device, max_batches=val_max_batches)
            print(f"  val_loss   = {metrics['val_loss']:.4f}")
            print(f"  val_fg_ari = {metrics['val_fg_ari']:.4f} ± {metrics['val_fg_ari_std']:.4f}")
            
            history['val_loss'].append((step + 1, metrics['val_loss']))
            history['val_fg_ari'].append((step + 1, metrics['val_fg_ari']))
            history['val_fg_ari_std'].append((step + 1, metrics['val_fg_ari_std']))
            
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
        
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(save_dir, f'step_{step+1}.pth')
            torch.save({
                'step': step + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_ari': best_ari,
                'history': history,
            }, ckpt_path)
            
            old_step = step + 1 - save_every * 3
            old_ckpt = os.path.join(save_dir, f'step_{old_step}.pth')
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
            
            history_path = os.path.join(save_dir, 'history.json')
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n" + "=" * 60)
    print(f"訓練完了")
    print(f"  総時間: {total_time/3600:.2f} 時間")
    print(f"  ベスト FG-ARI: {best_ari:.4f}")
    print("=" * 60)
    
    return model, history


# ============================================================
# 動作確認関数
# ============================================================
def test_pipeline():
    """訓練前のパイプライン動作確認"""
    print("=" * 60)
    print("Phase 2 パイプライン動作確認")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nデバイス: {device}")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # データ
    print("\n[1] データセット準備")
    from multi_dsprites_dataset import MultiDSpritesDataset
    train_dataset = MultiDSpritesDataset(npz_path="data/multi_dsprites_70k.npz", split='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    print(f"  train: {len(train_dataset)} samples")
    
    # モデル
    print("\n[2] モデル構築")
    print("  DINOv2 を読み込み中（初回はダウンロードあり）...")
    model = SlotAttentionDINOv2(num_slots=7, slot_dim=384, decoder_dim=64).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  全パラメータ:       {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"  訓練可能パラメータ: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    
    # モード確認
    print(f"\n[2.5] モード確認")
    print(f"  model.training (before): {model.training}")
    model.train()
    print(f"  model.training (after .train()): {model.training}")
    print(f"  encoder.dinov2.training: {model.encoder.dinov2.training}")
    print(f"  → ViT凍結なので dinov2.training は False になるべき")
    
    # Forward
    print("\n[3] forward 動作確認")
    batch = next(iter(train_loader))
    images = batch['image'].to(device)
    images_normalized = normalize_for_dinov2(images, device)
    print(f"  入力: {images_normalized.shape}")
    
    with torch.no_grad():
        recon, recons, masks, slots = model(images_normalized, return_slots=True)
    
    print(f"  recon:  {recon.shape}")
    print(f"  recons: {recons.shape}")
    print(f"  masks:  {masks.shape}")
    print(f"  slots:  {slots.shape}")
    
    if device.type == 'cuda':
        vram_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  VRAM 使用量（バッチ=8）: {vram_used:.2f} GB")
    
    # バッチ64でも試す
    print("\n[4] バッチサイズ64で動作確認")
    train_loader_64 = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    batch = next(iter(train_loader_64))
    images = batch['image'].to(device)
    images_normalized = normalize_for_dinov2(images, device)
    print(f"  入力: {images_normalized.shape}")
    
    try:
        with torch.no_grad():
            recon = model(images_normalized)
        print(f"  出力: {recon.shape}")
        
        if device.type == 'cuda':
            vram_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  VRAM 使用量: {vram_used:.2f} GB / 24 GB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ メモリ不足: {e}")
        print(f"  → バッチサイズを減らしてください")
    
    print("\n" + "=" * 60)
    print("✅ パイプライン動作確認完了")
    print("=" * 60)


# ============================================================
# メイン
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'train'])
    parser.add_argument('--num_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--eval_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_phase2')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    if args.mode == 'test':
        test_pipeline()
    else:
        train_phase2(
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            base_lr=args.lr,
            eval_every=args.eval_every,
            save_every=args.save_every,
            save_dir=args.save_dir,
            resume_from=args.resume,
        )