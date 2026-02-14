# Ví dụ: Huấn luyện Qwen với PPO cho tool-calling multi-turn

Lưu ý: LLaMA-Factory có code PPO trong `src/llamafactory/train/ppo/*`, nhưng upstream chưa có YAML PPO mẫu trong `examples/`.

## 1) Giả định
- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Bạn đã có reward model hoặc reward API
- Dataset prompt pool tool-calling multi-turn

## 2) YAML gợi ý
```yaml
### model
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

### method
stage: ppo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### rlhf
reward_model: /abs/path/to/qwen-toolcall-reward
reward_model_type: lora   # full | lora | api
ppo_buffer_size: 1
ppo_epochs: 4
ppo_target: 6.0
ppo_score_norm: false
ppo_whiten_rewards: false

### dataset
dataset: your_toolcalling_prompt_pool
template: qwen3_nothink
cutoff_len: 4096
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /abs/path/to/saves/qwen3-4b/lora/ppo-toolcall
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true
report_to: tensorboard

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

## 3) Chạy lệnh
```bash
llamafactory-cli train /abs/path/to/qwen3_ppo_toolcall.yaml
```

## 4) Điều chỉnh hyperparameter
1. Reward variance cao -> bật `ppo_score_norm=true`.
2. Policy drift quá nhanh -> giảm LR, giảm `ppo_epochs`.
3. Output dài lặp vô nghĩa -> siết generation config (max_new_tokens, stop tokens).

## 5) KPI nên theo dõi
- Tỉ lệ gọi đúng tool
- Tỉ lệ args hợp lệ
- Số bước trung bình/task
- Success@task
- Reward mean/std theo thời gian
