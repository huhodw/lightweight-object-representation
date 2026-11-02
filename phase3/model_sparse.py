"""
model_sparse.py

画像を k フレームに 1 回だけ入力する「間引き訓練」用のモデル。
SAViModel を継承していて、部品（エンコーダ・補正・予測器・デコーダ）とパラメータは
SAViModel とまったく同じ。違うのは forward の中のフレームのループだけ。
そのため、訓練したチェックポイントは普通の SAViModel にそのまま読み込める
（評価は sparse_observation_eval.py を修正なしで使える）。

★ forward の「ここから書く」〜「ここまで」の部分は 琦哥 が書く。
  書いたら test_model_sparse.py を実行して、すべて「合格」になることを確認する。

置き場所: savi_phase3 フォルダの直下（diagnose_predictor.py と同じ場所）
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "savi-pytorch"))

import torch
import torch.nn.functional as F

from model import SAViModel
# 元のモデルと完全一致することを確認済みの部品（diagnose_predictor.py の中にある）
#   encode_frame(model, x)        : 1フレームの画像 x [B,3,H,W] をエンコーダに通す -> [B, H*W, 64]
#   run_predictor(c, s)           : 予測器。s は norm_slots を通したスロット [B,K,D] -> [B,K,D]
#   run_correction(c, enc, s)     : 補正（Slot Attention）。画像の特徴 enc でスロット s を直す -> [B,K,D]
#   decode_slots(model, s)        : スロットを画像に戻す -> [B*K, 4, H, W]
from diagnose_predictor import encode_frame, run_predictor, run_correction, decode_slots


class SparseSAViModel(SAViModel):
    def __init__(self, *args, observe_every: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        if observe_every < 1:
            raise ValueError("observe_every は 1 以上にしてください")
        # k: フレーム t は t % k == 0 のときだけ画像を入力する（k=3 なら 0, 3）
        self.observe_every = observe_every

    def forward(self, batch, noise=None):
        """
        入力
          batch : [B, T, 3, H, W]  動画（値は 0〜1）
          noise : [B, K, D] または None。初期スロットを作る乱数。
                  None のときは、元の SAViModel と同じ作り方で内部で作る。
                  （テストと検証で、元のモデルと同じ初期スロットを使うための引数）
        出力（タプル）
          recon_combined : [B, T, 3, H, W]     再構成画像（全フレーム）
          recons         : [B, K, T, 3, H, W]  スロットごとの再構成画像
          masks          : [B, K, T, 1, H, W]  マスク（スロット方向に softmax 済み）
          slots_all      : [B, T, K, D]        各フレームのスロット
          observed       : list[bool]（長さ T） フレーム t で画像を入力したら True
        """
        c = self.corrector
        B, T = batch.shape[0], batch.shape[1]

        # 初期スロット（元の SAViModel.forward の 237〜239行と同じ）
        if noise is None:
            noise = torch.randn((B, self.num_slots, self.slot_size)).to(batch.device)
        prev = c.slots_mu + c.slots_log_sigma.exp() * noise

        slot_list = []   # 各フレームのスロット [B, K, D] を T 個ためる
        observed = []    # 各フレームで画像を入力したか（True / False）を T 個ためる

        # ===================== ここから書く =====================
        # フレーム t = 0, 1, ..., T-1 について、次のルールでそのフレームのスロット s_t を作り、
        # slot_list と observed に追加し、prev を s_t に更新する。
        #
        # ・画像を入力するフレーム（t % self.observe_every == 0）
        #     元のモデルとまったく同じ処理にする:
        #       1. 画像 batch[:, t] をエンコーダに通す
        #       2. 前のスロットを c.norm_slots で正規化する
        #       3. 予測器にかける（ただし t == 1 のときはかけない。元のコードの `if i != 1` と同じ）
        #       4. 補正（Slot Attention）で、画像の特徴を使って直す
        # ・画像を入力しないフレーム
        #     エンコーダも補正も呼ばない。前のスロットを c.norm_slots で正規化して、予測器だけにかける。
        #
        # 参考: sparse_observation_eval.py の forward_sparse（mode="predictor" のとき）。
        #       評価と訓練で同じ計算になっている必要がある（test_model_sparse.py で確認する）。
        #
        # 書き終わったら、すぐ下の raise の行を消すこと。

        for t in range(T):
            obs = t % self.observe_every == 0
            if obs:
                enc = encode_frame(self, batch[:, t])
                s_in = c.norm_slots(prev)
                s_pred = run_predictor(c, s_in) if t != 1 else s_in
                s_t = run_correction(c, enc, s_pred)
            else:
                s_t = run_predictor(c, c.norm_slots(prev))
            slot_list.append(s_t)
            observed.append(obs)
            prev = s_t
        # ===================== ここまで =====================

        recon_combined, recons, masks, slots_all = self._decode_all(slot_list)
        return recon_combined, recons, masks, slots_all, observed

    def _decode_all(self, slot_list):
        """全フレームのスロットを画像に戻して合成する（元の SAViModel.forward の 262〜281行と同じ計算）。
        画像を入力しなかったフレームもデコードするので、そのフレームの本物の画像との誤差が損失に入る。"""
        B, K, D = slot_list[0].shape
        T = len(slot_list)
        outs = [decode_slots(self, s) for s in slot_list]   # 各 [B*K, 4, H, W]
        out = torch.stack(outs, dim=1)                       # [B*K, T, 4, H, W]
        H, W = out.shape[-2], out.shape[-1]
        C = self.in_channels
        out = out.view(B, K, T, C + 1, H, W)
        recons = out[:, :, :, :C, :, :]
        masks = F.softmax(out[:, :, :, C:, :, :], dim=1)     # スロット方向の softmax
        recon_combined = torch.sum(recons * masks, dim=1)
        slots_all = torch.stack(slot_list, dim=1)            # [B, T, K, D]
        return recon_combined, recons, masks, slots_all
