# Phân tích chi tiết DPO, KTO, ORPO từ mã nguồn hiện tại

## 1) DPO (và biến thể ORPO/SimPO)

### Vị trí code
- `src/llamafactory/train/dpo/workflow.py`
- `src/llamafactory/train/dpo/trainer.py`

### Luồng
1. Load pairwise dataset (chosen/rejected).
2. Build policy model.
3. Optional build reference model (trừ ORPO/SimPO).
4. Tính log-prob cho chosen/rejected.
5. Tính preference loss + metric reward margins.

### Toán học trực giác DPO
- Mục tiêu: tăng xác suất preferred response tương đối so với rejected, có neo theo reference.
- Dùng log-ratio policy vs reference để tránh policy drift.

### ORPO trong code
- Chạy qua `pref_loss=orpo` trong cùng trainer DPO.
- Loss dùng odds-ratio + SFT term:
  - `orpo_loss = sft_loss + beta * odds_ratio_loss`
- Reference-free theo logic `use_ref_model=False`.

### SimPO trong code
- `pref_loss=simpo`.
- Dùng margin `simpo_gamma` trên log-ratio chosen/rejected.

## 2) KTO

### Vị trí code
- `src/llamafactory/train/kto/workflow.py`
- `src/llamafactory/train/kto/trainer.py`

### Dữ liệu khác DPO thế nào?
- KTO dùng desirable/undesirable tags (`kto_tags`) + batch KL riêng (`kl_*`).
- Processor/collator chuyên biệt cho cấu trúc này.

### Trực giác toán học
- KTO cân bằng hai loại phản hồi (desirable/undesirable) với trọng số riêng.
- Tận dụng KL-guided signal để giữ policy ổn định.

## 3) Metrics quan trọng trong source
- DPO/KTO đều log:
  - rewards/chosen
  - rewards/rejected
  - rewards/margins
  - logps/logits
- KTO có thêm aggregate theo chosen/rejected count và KL.

## 4) Khi nào chọn thuật toán nào?
- DPO: dữ liệu pairwise sạch, muốn baseline mạnh và đơn giản.
- ORPO: muốn reference-free, giảm complexity ref model.
- SimPO: cần margin-based preference behavior.
- KTO: dữ liệu feedback dạng desirable/undesirable tự nhiên hơn pairwise hoàn chỉnh.

## 5) Gợi ý training recipe cho người mới
1. Bắt đầu DPO sigmoid.
2. So sánh ORPO để giảm phụ thuộc ref model.
3. Thử KTO nếu pipeline dữ liệu phù hợp tags.
4. Chỉ sau đó mới vào PPO online optimization.
