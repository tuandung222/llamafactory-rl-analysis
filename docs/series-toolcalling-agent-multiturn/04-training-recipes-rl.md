# Training Recipes cho Tool-Calling Multi-Turn

## 1) Chiến lược huấn luyện theo pha
1. SFT baseline (không trình bày chi tiết ở đây).
2. RM trên pairwise tool-calling.
3. DPO/KTO để alignment offline.
4. PPO để tối ưu online theo reward.

## 2) RM recipe (khuyến nghị trước)
- Stage: `rm`
- Mục tiêu: học scoring function cho trajectory.
- Lưu ý: dev set bắt buộc có hard failures.

## 3) DPO/ORPO/SimPO recipe
- Stage: `dpo`
- `pref_loss=sigmoid` trước.
- Sau đó benchmark thêm `orpo` hoặc `simpo`.
- Giữ LR thấp (`1e-6` tới `5e-6`) khi data noisy.

## 4) KTO recipe
- Stage: `kto`
- Dùng khi feedback dạng positive/negative dễ thu thập hơn pairwise chuẩn.
- Điều chỉnh `kto_chosen_weight`/`kto_rejected_weight` nếu lệch class.

## 5) PPO recipe
- Stage: `ppo`
- Bắt buộc `reward_model`.
- Chọn `reward_model_type` theo hạ tầng:
  - `lora`/`full` cho local,
  - `api` cho reward service.

## 6) Hyperparameter tuning order
1. Cố định dataset + prompt rendering.
2. Tune learning rate.
3. Tune beta/ftx hoặc ppo_target/ppo_epochs.
4. Tune batch/grad accumulation.

## 7) Minimal KPI board
- Task success rate.
- Correct tool selection rate.
- Valid argument rate.
- Avg tool calls / successful task.
- Recovery success rate.
- Reward mean/std (nếu PPO).
