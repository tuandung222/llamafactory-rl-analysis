# Phân tích GRPO từ mã nguồn hiện tại

## Kết luận nhanh
Trong source LLaMA-Factory ở commit `f80e15db`, không thấy:
- `stage: grpo`
- thư mục `train/grpo/`
- trainer/workflow/config args cho GRPO.

Vì vậy không thể “phân tích implementation GRPO hiện tại” theo kiểu line-by-line như PPO/DPO/KTO/ORPO.

## 1) Đã kiểm tra gì?
- Tìm từ khoá `grpo` toàn repo: không có code path train tương ứng.
- Router stage (`tuner.py`) chỉ có: `pt/sft/rm/ppo/dpo/kto`.

## 2) Nếu thêm GRPO thì cần gì về mặt thuật toán?
Giả sử mỗi prompt có `K` rollout:
1. Tính reward từng rollout: `r_i`.
2. Group baseline/normalization: 
   - ví dụ `A_i = (r_i - mean(r_group)) / (std(r_group)+eps)`.
3. Tối ưu policy với clipped objective giống PPO nhưng advantage là group-relative.
4. Thêm KL penalty với reference policy để giữ ổn định.

## 3) Kiến trúc code cần bổ sung trong LLaMA-Factory
1. `FinetuningArguments`: thêm hyperparams GRPO.
2. `tuner.py`: thêm stage `grpo`.
3. `train/grpo/workflow.py`: load data/model/ref/reward.
4. `train/grpo/trainer.py`: rollout nhóm + group advantage + update.
5. logging metric đặc trưng:
   - group reward mean/std
   - policy entropy
   - KL to ref
   - clip fraction

## 4) Khuyến nghị cho team
- Ngắn hạn: dùng DPO/KTO/PPO đã có để đạt baseline ổn định.
- Trung hạn: nếu cần GRPO thật, phát triển branch riêng và test A/B với PPO.
