# Kiến trúc mã nguồn LLaMA-Factory và so sánh với HuggingFace TRL

## 1) Toàn cảnh kiến trúc

```mermaid
flowchart TB
  subgraph Entry
    A[llamafactory-cli train] --> B[src/train.py]
    B --> C[src/llamafactory/train/tuner.py]
  end

  C --> D{stage}
  D --> RM[run_rm]
  D --> DPO[run_dpo]
  D --> KTO[run_kto]
  D --> PPO[run_ppo]

  subgraph SharedInfra
    E[data loader + parser + processor]
    F[model loader + adapter + value head]
    G[trainer_utils: ref/reward/optimizer/scheduler]
  end

  RM --> E
  DPO --> E
  KTO --> E
  PPO --> E
  RM --> F
  DPO --> F
  KTO --> F
  PPO --> F
  RM --> G
  DPO --> G
  KTO --> G
  PPO --> G
```

Điểm chính:
- LLaMA-Factory dùng TRL như một dependency cho một số trainer (PPO/DPO/KTO), nhưng có thêm lớp orchestration và custom behavior quanh data/model/training.
- Có khả năng multimodal, nhiều backend, nhiều optimizer ngoài chuẩn HF.

## 2) Code path cụ thể
- Entry CLI: `src/train.py`
- Router: `src/llamafactory/train/tuner.py`
- Stage workflow:
  - RM: `src/llamafactory/train/rm/workflow.py`
  - DPO: `src/llamafactory/train/dpo/workflow.py`
  - KTO: `src/llamafactory/train/kto/workflow.py`
  - PPO: `src/llamafactory/train/ppo/workflow.py`

## 3) LLaMA-Factory khác TRL ở đâu?

### 3.1 Mức orchestration
- TRL: tập trung trainer + objective.
- LLaMA-Factory: thêm full pipeline:
  - dataset abstraction + parser/converter
  - template/chat-format
  - model loading with LoRA/OFT/freeze/full
  - distributed/ray/webui integration

### 3.2 Dữ liệu và collator
- TRL thường giả định input schema gần chuẩn trainer.
- LLaMA-Factory có processor riêng theo stage:
  - pairwise, feedback(KTO), unsupervised(PPO)
- Hỗ trợ multimodal token mapping + cross attention mask.

### 3.3 Objective extensions
- DPO trainer của LLaMA-Factory mở rộng:
  - ORPO, SimPO, BCO mixing, FTX mixing, LD-DPO alpha.

### 3.4 Reward model handling trong PPO
- Hỗ trợ `reward_model_type`:
  - `full`
  - `lora/oft` (switch adapter + swap value head buffer)
  - `api` (server scoring)

### 3.5 Khả năng sản xuất (production-ish)
- WebUI + CLI thống nhất.
- Nhiều optimizer tích hợp: GaLore/APOLLO/BAdam/Adam-mini/Muon.
- Tương thích đa phần model zoo lớn.

## 4) Minh hoạ sequence chạy `stage=dpo`

```mermaid
sequenceDiagram
  participant U as User
  participant CLI as llamafactory-cli
  participant T as tuner.run_exp
  participant W as dpo.workflow
  participant DL as data.loader
  participant TR as CustomDPOTrainer

  U->>CLI: train dpo.yaml
  CLI->>T: parse args
  T->>W: run_dpo(...)
  W->>DL: get_dataset(stage="rm")
  W->>W: load_model + create_ref_model
  W->>TR: init trainer
  TR->>TR: train/eval loop
  TR-->>U: metrics + checkpoints
```

## 5) Kết luận kiến trúc
- LLaMA-Factory = “training system” hoàn chỉnh hơn TRL ở tầng workflow.
- TRL vẫn là thành phần quan trọng ở tầng algorithm trainer.
- Nếu mục tiêu là vận hành nhiều bài toán thực nghiệm nhanh, LLaMA-Factory phù hợp hơn việc tự ghép rời rạc TRL + HF scripts.
