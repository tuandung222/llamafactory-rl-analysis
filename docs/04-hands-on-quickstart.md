# Hands-on Quickstart

## 0) Install and Verify
```bash
cd /Users/admin/TuanDung/repos/LLaMA-Factory
pip install -e .
```

## 1) Reward Model
```bash
llamafactory-cli train examples/train_lora/qwen3_lora_reward.yaml
```

## 2) DPO
```bash
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml
```

Switch objective:
```bash
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml pref_loss=orpo
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml pref_loss=simpo
```

## 3) KTO
```bash
llamafactory-cli train examples/train_lora/qwen3_lora_kto.yaml
```

## 4) PPO (manual template since no official sample YAML in examples)
```yaml
# ppo_example.yaml
model_name_or_path: Qwen/Qwen3-4B-Instruct-2507
trust_remote_code: true

stage: ppo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

# required for PPO
reward_model: /path/to/reward-model
reward_model_type: lora  # or full, api

dataset: alpaca_en_demo
template: qwen3_nothink
cutoff_len: 2048

output_dir: saves/qwen3-4b/lora/ppo
logging_steps: 10
save_steps: 200
overwrite_output_dir: true

per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 5.0e-6
num_train_epochs: 1.0
bf16: true

# PPO knobs
ppo_buffer_size: 1
ppo_epochs: 4
ppo_target: 6.0
ppo_score_norm: false
ppo_whiten_rewards: false
```

Run:
```bash
llamafactory-cli train ppo_example.yaml
```
