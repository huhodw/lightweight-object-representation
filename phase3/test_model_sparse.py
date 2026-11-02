"""
test_model_sparse.py

琦哥 が書いた model_sparse.py の forward が正しいかを確かめるテスト。
データもチェックポイントも使わない（ランダムな重みとランダムな動画で確かめる）。

確かめること
  1. パラメータの名前と数が SAViModel と同じか（訓練したモデルを評価スクリプトで読み込めるか）
  2. k = 1, 2, 3, 6 のそれぞれで
       - 出力の形
       - 画像を入力したフレームが t % k == 0 のフレームになっているか
       - エンコーダを呼んだ回数が、画像を入力したフレームの数と同じか
       - 評価スクリプト sparse_observation_eval.py の forward_sparse（mode="predictor"）と同じ計算か
     k = 1 では、元の SAViModel.forward とも同じ計算か（初期スロットの乱数の作り方も含む）
  3. k = 3 で、画像を入力しないフレームの誤差から、予測器・デコーダ・エンコーダに勾配が流れるか
  4. 全フレームの誤差で逆伝播ができるか

置き場所: savi_phase3 フォルダの直下（model_sparse.py と同じ場所）
実行: python test_model_sparse.py
"""
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "savi-pytorch"))

import torch
import torch.nn.functional as F

from model import SAViModel
from params import SAViParams
from sparse_observation_eval import forward_sparse
from model_sparse import SparseSAViModel

TOL = 1e-5
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'合格' if ok else '不合格'}] {name}" + (f"  ({detail})" if detail else ""))


def build(p, k, ref, device):
    m = SparseSAViModel(
        resolution=p.resolution, num_slots=p.num_slots,
        num_iterations=p.num_iterations, observe_every=k,
    ).to(device)
    m.load_state_dict(ref.state_dict())
    return m


def grad_norm(param):
    return 0.0 if param.grad is None else param.grad.norm().item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"使用デバイス: {device}")

    p = SAViParams()
    T = p.seq_len
    H, W = p.resolution
    K = p.num_slots

    torch.manual_seed(0)
    ref = SAViModel(resolution=p.resolution, num_slots=p.num_slots,
                    num_iterations=p.num_iterations).to(device).eval()
    D = ref.slot_size

    # ----- 1. パラメータの名前と数 -----
    m1 = build(p, 1, ref, device)
    same_keys = list(ref.state_dict().keys()) == list(m1.state_dict().keys())
    check("パラメータの名前と数が SAViModel と同じ（評価スクリプトでそのまま読み込める）", same_keys)

    video = torch.rand((2, T, 3, H, W), generator=torch.Generator().manual_seed(1)).to(device)
    noise = torch.randn((2, K, D), generator=torch.Generator().manual_seed(2)).to(device)

    # forward が書かれているか
    try:
        with torch.no_grad():
            m1.eval()
            m1(video[:1], noise=noise[:1])
    except NotImplementedError as e:
        print()
        print(f"止めました: {e}")
        print("model_sparse.py の「ここから書く」〜「ここまで」を書いて、raise の行を消してから、もう一度実行してください。")
        sys.exit(1)

    # ----- 2. k ごとの確認 -----
    with torch.no_grad():
        for k in (1, 2, 3, 6):
            print(f"--- k={k} ---")
            m = build(p, k, ref, device).eval()

            calls = [0]

            def count_call(module, inp, out):
                calls[0] += 1

            hook = m.encoder.register_forward_hook(count_call)
            out = m(video, noise=noise)
            hook.remove()

            if not (isinstance(out, tuple) and len(out) == 5):
                check(f"k={k}: 出力が5個のタプル (recon_combined, recons, masks, slots_all, observed)", False,
                      f"実際: {type(out).__name__}, 長さ {len(out) if isinstance(out, tuple) else '-'}")
                continue
            recon, recons, masks, slots_all, observed = out

            shapes_ok = (tuple(recon.shape) == (2, T, 3, H, W)
                         and tuple(recons.shape) == (2, K, T, 3, H, W)
                         and tuple(masks.shape) == (2, K, T, 1, H, W)
                         and tuple(slots_all.shape) == (2, T, K, D))
            check(f"k={k}: 出力の形", shapes_ok,
                  f"recon {tuple(recon.shape)}, masks {tuple(masks.shape)}, slots_all {tuple(slots_all.shape)}")

            expected = [t % k == 0 for t in range(T)]
            got = [bool(x) for x in observed]
            check(f"k={k}: 画像を入力したフレーム = {[t for t in range(T) if expected[t]]}",
                  got == expected, f"observed = {got}")

            check(f"k={k}: エンコーダの呼び出し回数 = {sum(expected)}/{T}",
                  calls[0] == sum(expected), f"実際 {calls[0]} 回")

            r2, m2, _ = forward_sparse(m, video, noise, k, "predictor")
            diff = max((recon - r2).abs().max().item(), (masks - m2).abs().max().item())
            check(f"k={k}: 評価スクリプト（forward_sparse）と同じ計算", diff <= TOL, f"最大差 {diff:.2e}")

            if k == 1:
                torch.manual_seed(3)
                ref_out = ref(video)
                torch.manual_seed(3)
                my_out = m(video)
                diff1 = max((my_out[0] - ref_out[0]).abs().max().item(),
                            (my_out[2] - ref_out[2]).abs().max().item())
                check("k=1: 元の SAViModel.forward と同じ計算（初期スロットの乱数の作り方も含む）",
                      diff1 <= TOL, f"最大差 {diff1:.2e}")

    # ----- 3. 勾配の確認（k=3） -----
    print("--- 勾配（k=3）---")
    m = build(p, 3, ref, device).train()
    v1, n1 = video[:1], noise[:1]
    recon, _, _, _, observed = m(v1, noise=n1)
    unobs = [t for t in range(T) if not observed[t]]
    loss_un = F.mse_loss(recon[:, unobs], v1[:, unobs])
    m.zero_grad()
    loss_un.backward()
    g_pred = grad_norm(m.corrector.multihead_attn.in_proj_weight) + grad_norm(m.corrector.mlp_attn[0].weight)
    g_dec = grad_norm(m.decoder[0][0].weight)
    g_enc = grad_norm(m.encoder[0][0].weight)
    check(f"画像を入力しないフレーム {unobs} の誤差から、予測器に勾配が流れる", g_pred > 0, f"勾配の大きさ {g_pred:.2e}")
    check("同じく、デコーダに勾配が流れる", g_dec > 0, f"勾配の大きさ {g_dec:.2e}")
    check("同じく、エンコーダに勾配が流れる（フレーム0の画像 → 予測器 → 後のフレーム）", g_enc > 0,
          f"勾配の大きさ {g_enc:.2e}")

    # ----- 4. 全フレームの誤差で逆伝播 -----
    m.zero_grad()
    recon, *_ = m(v1, noise=n1)
    loss = F.mse_loss(recon, v1)
    loss.backward()
    finite = all(torch.isfinite(q.grad).all().item() for q in m.parameters() if q.grad is not None)
    check("全フレームの誤差で逆伝播ができる（勾配に NaN や無限大がない）", torch.isfinite(loss).item() and finite,
          f"損失 {loss.item():.4f}")

    print("=" * 60)
    n_bad = results.count(False)
    if n_bad == 0:
        print(f"すべて合格（{len(results)} 項目）")
    else:
        print(f"不合格 {n_bad} / {len(results)} 項目。不合格の項目を見て model_sparse.py を直してください。")
        sys.exit(1)


if __name__ == "__main__":
    main()
