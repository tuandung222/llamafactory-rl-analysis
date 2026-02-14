# Ví dụ: Huấn luyện Gemma với DPO cho tool-calling multi-turn

## 1) Mục tiêu
Dùng pairwise preference để ưu tiên trajectory tool-calling tốt hơn.

## 2) YAML gợi ý
```yaml
### model
model_name_or_path: google/gemma-3-4b-it
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid   # thử orpo/simpo ở lượt sau
pref_ftx: 0.0

### optional ref
# ref_model: google/gemma-3-4b-it

### dataset
dataset: gemma_toolcall_pairwise_multiturn
template: gemma
cutoff_len: 4096
max_samples: 200000
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /abs/path/to/saves/gemma3-4b/lora/dpo-toolcall
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
report_to: tensorboard

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-6
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
```

## 3) Chạy lệnh
```bash
llamafactory-cli train /abs/path/to/gemma_dpo_toolcall.yaml
```

## 4) Biến thể ORPO/SimPO
Giữ nguyên config và đổi:
- `pref_loss: orpo`
- hoặc `pref_loss: simpo`

Lưu ý:
- Khi `orpo/simpo`, hệ thống chạy reference-free theo logic hiện tại.
- Không set `dpo_label_smoothing > 0` với non-sigmoid.

## 5) Đánh giá
Bộ eval nên có:
- easy tool-call
- ambiguous tool selection
- failure recovery
- long-context multi-turn
