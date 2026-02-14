# Phân tích ý nghĩa training config (RL focus)

Tài liệu này giải thích theo cấu trúc YAML mà bạn thường truyền vào `llamafactory-cli train ...yaml`, đối chiếu với logic kiểm tra trong `src/llamafactory/hparams/finetuning_args.py` và workflow trong `src/llamafactory/train/*/workflow.py`.

## 1) Nhóm `model`
Các field thường gặp:
- `model_name_or_path`: model gốc (HF path hoặc local).
- `trust_remote_code`: bật khi model repo yêu cầu custom modeling code.
- `adapter_name_or_path`: dùng khi resume hoặc ghép adapter.
- `quantization_bit`: 4/8-bit cho QLoRA hoặc inference path.

Ý nghĩa vận hành:
- Quyết định backbone + tokenizer + khả năng load adapter/value head.
- Tác động trực tiếp memory footprint và tốc độ.

Pitfall:
- Tokenizer mismatch giữa policy/reward/ref model gây reward sai (PPO).
- Dùng quantization không tương thích với stage hoặc backend.

## 2) Nhóm `method` (quan trọng nhất với RL)
Field lõi:
- `stage`: `rm | ppo | dpo | kto` (ngoài ra có `pt`, `sft`).
- `finetuning_type`: `lora | oft | freeze | full`.

Field preference/RLHF:
- `pref_beta`: scale của reward/log-ratio trong DPO/KTO/ORPO/SimPO.
- `pref_ftx`: trộn thêm supervised loss (FTX term).
- `pref_loss`: `sigmoid | hinge | ipo | kto_pair | orpo | simpo`.
- `dpo_label_smoothing`: chỉ hợp lệ khi `pref_loss=sigmoid`.
- `kto_chosen_weight`, `kto_rejected_weight`: trọng số desirable/undesirable cho KTO.
- `simpo_gamma`: margin term trong SimPO.

Field PPO:
- `reward_model`: bắt buộc ở stage PPO.
- `reward_model_type`: `lora | full | api`.
- `ppo_buffer_size`, `ppo_epochs`, `ppo_target`, `ppo_score_norm`, `ppo_whiten_rewards`.

Field ref model:
- `ref_model`, `ref_model_adapters`, `ref_model_quantization_bit`.

Ràng buộc từ source:
- PPO thiếu `reward_model` -> lỗi ngay.
- `reward_model_type=lora` chỉ hợp lệ khi `finetuning_type=lora`.
- `reward_model_type=oft` chỉ hợp lệ khi `finetuning_type=oft`.
- DPO + non-sigmoid mà `dpo_label_smoothing>0` -> lỗi.
- DPO với `orpo/simpo` sẽ `use_ref_model=False`.

## 3) Nhóm `dataset`
Field phổ biến:
- `dataset`, `eval_dataset`
- `template`
- `cutoff_len`
- `max_samples`
- `preprocessing_num_workers`, `dataloader_num_workers`
- `val_size`, `streaming`, `mix_strategy`

Ý nghĩa:
- `stage` quyết định processor:
  - `rm` -> `PairwiseDatasetProcessor`
  - `kto` -> `FeedbackDatasetProcessor`
  - `ppo` -> `UnsupervisedDatasetProcessor`
- Cùng dataset nhưng schema sai stage sẽ bị reject.

Pitfall với tool-calling:
- Không đồng nhất format tool message (assistant/function/observation).
- Prompt-turn lẻ/chẵn sai chuẩn bị drop mẫu.
- Dữ liệu chosen/rejected chứa noise (cả 2 đều tệ) làm DPO/ORPO học lệch.

## 4) Nhóm `output`
- `output_dir`, `logging_steps`, `save_steps`, `plot_loss`, `overwrite_output_dir`, `report_to`.

Ý nghĩa:
- Kiểm soát checkpoint cadence, logging, và khả năng resume/debug.

Pitfall:
- `save_steps` quá lớn với run ngắn dễ mất checkpoint tốt.
- `overwrite_output_dir=true` khi resume nhầm run có thể mất artifact cũ.

## 5) Nhóm `train`
- `per_device_train_batch_size`
- `gradient_accumulation_steps`
- `learning_rate`
- `num_train_epochs` hoặc `max_steps`
- `lr_scheduler_type`, `warmup_ratio`
- `bf16/fp16`
- `ddp_timeout`

Ý nghĩa:
- Quyết định stability và throughput.

Pitfall RL:
- LR cao trong DPO/PPO thường làm policy collapse nhanh.
- Batch quá nhỏ làm reward signal noisy.
- PPO cần cân bằng buffer/epochs để không overfit reward.

## 6) Mẫu checklist trước khi chạy RL
1. Đúng `stage` chưa?
2. Dataset schema đúng stage chưa?
3. PPO có `reward_model` và đúng `reward_model_type` chưa?
4. DPO non-sigmoid đã tắt `dpo_label_smoothing` chưa?
5. Tokenizer giữa policy/ref/reward có đồng bộ chưa?
6. Logging/checkpoint đã đủ dày để debug chưa?
