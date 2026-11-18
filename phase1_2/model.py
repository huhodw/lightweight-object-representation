import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Slot Attentionモジュール
    画像特徴量をnum_slots個の「スロット」に分解する
    """
    def __init__(self, num_slots, dim, iters=3, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.dim = dim
        self.iters = iters  # 反復回数
        self.eps = eps

        self.scale = dim ** -0.5  # アテンションのスケーリング係数

        # スロットの初期値を学習するパラメータ
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))

        # Q, K, V の線形変換
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        # スロット更新用GRU
        self.gru = nn.GRUCell(dim, dim)

        # フィードフォワードネットワーク
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

        # Layer Normalization
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    def forward(self, inputs):
        """
        inputs: (B, N, dim)  B=バッチ, N=特徴量の数, dim=次元
        """
        B, N, D = inputs.shape

        # 入力を正規化
        inputs = self.norm_input(inputs)

        # スロットをガウス分布からサンプリングして初期化
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_sigma.expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)  # (B, num_slots, dim)

        # K, V は入力から計算（ループ外で一度だけ）
        k = self.to_k(inputs)  # (B, N, dim)
        v = self.to_v(inputs)  # (B, N, dim)

        # Slot Attentionの反復
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Q はスロットから計算
            q = self.to_q(slots)  # (B, num_slots, dim)

            # アテンションスコアを計算
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            # (B, num_slots, N)

            # ★ここが核心：dim=1（スロット次元）でsoftmax
            # 各特徴量がどのスロットに属するかを競争させる
            attn = dots.softmax(dim=1) + self.eps  # (B, num_slots, N)

            # アテンションを正規化
            attn = attn / attn.sum(dim=-1, keepdim=True)  # (B, num_slots, N)

            # 加重平均でスロットを更新
            updates = torch.einsum('bij,bjd->bid', attn, v)
            # (B, num_slots, dim)

            # GRUでスロットを更新
            slots = self.gru(
                updates.reshape(B * self.num_slots, D),
                slots_prev.reshape(B * self.num_slots, D)
            ).reshape(B, self.num_slots, D)

            # 残差接続 + MLP
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class CNN(nn.Module):
    """
    特徴抽出用CNN（Phase 1で使用、Phase 2でViTに差し替える）
    入力: (B, 3, H, W)
    出力: (B, N, dim)  N=特徴量の数
    """
    def __init__(self, dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # (B, 32, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),  # (B, 32, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(32, dim, 4, stride=2, padding=1), # (B, dim, H/16, W/16)
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)          # (B, dim, H', W')
        B, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1)   # (B, H', W', dim)
        x = x.reshape(B, H * W, D)  # (B, N, dim)  N=H'*W'
        return x


class SlotAttentionAutoEncoder(nn.Module):
    """
    Slot Attention AutoEncoder
    CNN + SlotAttention + Decoder の組み合わせ
    """
    def __init__(self, num_slots=4, dim=64, iters=3):
        super().__init__()
        self.num_slots = num_slots

        # エンコーダ（Phase 2でViTに差し替え）
        self.encoder = CNN(dim=dim)

        # Slot Attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=dim,
            iters=iters
        )

        # デコーダ（各スロットから画像を再構成）
        self.decoder = nn.Sequential(
            nn.Linear(dim, 32 * 4 * 4),
            nn.ReLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 4, stride=2, padding=1),   # 64x64, 4ch(RGB+マスク)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 特徴抽出
        features = self.encoder(x)           # (B, N, dim)

        # Slot Attention
        slots, attn = self.slot_attention(features)  # (B, num_slots, dim)

        # 各スロットをデコード
        B, S, D = slots.shape
        slots_flat = slots.reshape(B * S, D)

        x_dec = self.decoder(slots_flat)         # (B*S, 32*4*4)
        x_dec = x_dec.reshape(B * S, 32, 4, 4)  # (B*S, 32, 4, 4)
        x_dec = self.decoder_conv(x_dec)         # (B*S, 4, 64, 64)
        x_dec = x_dec.reshape(B, S, 4, H, W)    # (B, S, 4, H, W)

        # RGBとマスクに分離
        recons = x_dec[:, :, :3, :, :]          # (B, S, 3, H, W)
        masks = x_dec[:, :, 3:, :, :]           # (B, S, 1, H, W)
        masks = masks.softmax(dim=1)             # スロット間でsoftmax

        # マスクで重み付けして合成
        recon_combined = (recons * masks).sum(dim=1)  # (B, 3, H, W)

        return recon_combined, recons, masks, slots


if __name__ == '__main__':
    # 動作確認
    model = SlotAttentionAutoEncoder(num_slots=4, dim=64, iters=3)
    x = torch.randn(2, 3, 64, 64)  # バッチサイズ2
    recon, recons, masks, slots = model(x)
    print(f"入力: {x.shape}")
    print(f"再構成画像: {recon.shape}")       # (2, 3, 64, 64)
    print(f"各スロット画像: {recons.shape}") # (2, 4, 3, 64, 64)
    print(f"マスク: {masks.shape}")           # (2, 4, 1, 64, 64)
    print(f"スロット: {slots.shape}")         # (2, 4, 64)
    print("モデルOK!")