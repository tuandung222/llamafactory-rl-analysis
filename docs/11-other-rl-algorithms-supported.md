# LLaMA-Factory còn hỗ trợ thuật toán RL nào?

Theo source hiện tại (commit phân tích):

## 1) Nhóm preference/RLHF có trong training stage
- `rm` (Reward Modeling)
- `ppo`
- `dpo`
- `kto`

## 2) Biến thể objective có trong DPO trainer
Thông qua `pref_loss`:
- `sigmoid` (DPO chuẩn)
- `hinge`
- `ipo`
- `kto_pair`
- `orpo`
- `simpo`

## 3) Những gì KHÔNG thấy trong source stage train
- `grpo` stage/trainer/workflow

## 4) Nên hiểu “hỗ trợ thuật toán” thế nào?
- Có objective enum nhưng chưa chắc có stage riêng.
- Có stage riêng mới có pipeline đầy đủ (data/ref/reward/logging/checkpoint) cho production experiment.

## 5) Lộ trình mở rộng thực tế
1. Dùng DPO/KTO/PPO để đạt baseline tool-calling.
2. Nếu cần GRPO, bổ sung stage mới như tài liệu `09-grpo-from-current-source.md`.
3. Giữ chuẩn đánh giá thống nhất để so sánh công bằng giữa objective.
