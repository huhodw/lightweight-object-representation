import torch
import numpy as np
from sklearn.metrics import adjusted_rand_score


# ========== 1. 学習率スケジューラ ==========
def get_lr(step, base_lr=4e-4, warmup_steps=10000, decay_steps=100000, decay_rate=0.5):
    """
    Slot Attention原著の学習率スケジュール。
    
    なぜ必要か:
    - Warmup: 訓練初期は勾配が不安定。徐々に学習率を上げることで、
      スロット初期化の崩壊を防ぐ。
    - Decay: 収束後に学習率が高いままだと、損失曲面の平坦な領域で
      モデルがふらふらしてスロット分業が崩れる（あなたの200エポック問題の原因）。
    """
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    decay_factor = decay_rate ** ((step - warmup_steps) / decay_steps)
    return base_lr * decay_factor


# ========== 2. FG-ARI 計算関数 ==========
def compute_fg_ari(pred_masks, true_masks):
    """
    pred_masks: (B, N_slots, H, W) - モデルが出力したスロットマスク
    true_masks: (B, H, W) - 正解インスタンスマスク（背景=0、物体=1,2,3...）
    
    FG-ARI = Foreground Adjusted Rand Index
    背景を除外した物体領域のみで「分離精度」を測る指標。
    """
    pred_assignments = pred_masks.argmax(dim=1)  # (B, H, W)
    
    aris = []
    for b in range(pred_masks.shape[0]):
        # 「FG」= Foreground、つまり物体領域のみ評価
        # 背景を含めると「全部背景」と予測しても高スコアになるので除外
        fg_mask = true_masks[b] > 0
        
        if fg_mask.sum() == 0:
            continue
        
        pred_flat = pred_assignments[b][fg_mask].cpu().numpy().flatten()
        true_flat = true_masks[b][fg_mask].cpu().numpy().flatten()
        
        ari = adjusted_rand_score(true_flat, pred_flat)
        aris.append(ari)
    
    return np.mean(aris) if aris else 0.0


# ========== 3. 改善された訓練ループ ==========
def train_with_monitoring(model, train_loader, val_loader, num_epochs=100, 
                           eval_every=5, patience=4):
    """
    あなたの現状を踏まえた訓練ループ。
    
    主な改善点:
    1. 学習率スケジューラを追加（収束後のドリフトを抑制）
    2. eval_every エポックごとに FG-ARI を測定（推移を可視化）
    3. ベストモデルを保存（FG-ARI最大時点）
    4. Early stopping（FG-ARI が改善しなくなったら停止）
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)
    
    best_ari = -1.0
    no_improve_count = 0
    global_step = 0
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'fg_ari': []}
    
    for epoch in range(num_epochs):
        # ----- 訓練フェーズ -----
        model.train()
        train_losses = []
        for batch in train_loader:
            # 学習率を毎ステップ更新（バッチ単位、エポック単位ではない！）
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            images = batch['image'].cuda()
            output = model(images)
            loss = output['loss']  # reconstruction loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            global_step += 1
        
        avg_train_loss = np.mean(train_losses)
        
        # ----- 評価フェーズ（eval_every エポックごと） -----
        if (epoch + 1) % eval_every == 0:
            model.eval()
            val_losses = []
            all_aris = []
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].cuda()
                    true_masks = batch['mask'].cuda()
                    
                    output = model(images)
                    val_losses.append(output['loss'].item())
                    
                    pred_masks = output['masks']  # (B, N_slots, H, W)
                    ari = compute_fg_ari(pred_masks, true_masks)
                    all_aris.append(ari)
            
            avg_val_loss = np.mean(val_losses)
            avg_ari = np.mean(all_aris)
            
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['fg_ari'].append(avg_ari)
            
            print(f"Epoch {epoch+1:3d} | LR: {lr:.6f} | "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"FG-ARI: {avg_ari:.4f}")
            
            # ベストモデル保存
            if avg_ari > best_ari:
                best_ari = avg_ari
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'fg_ari': avg_ari,
                }, 'best_model.pt')
                no_improve_count = 0
                print(f"  → 新ベスト保存: FG-ARI = {avg_ari:.4f}")
            else:
                no_improve_count += 1
                
            # Early stopping
            # patience回連続で改善しなければ停止（無駄な訓練を避ける）
            if no_improve_count >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(FG-ARI not improving for {patience} evals)")
                break
    
    print(f"\n訓練終了。ベスト FG-ARI: {best_ari:.4f}")
    return history