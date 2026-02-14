# RL Stages Deep Dive

## 1) Reward Modeling (RM)
- Files:
  - `src/llamafactory/train/rm/workflow.py`
  - `src/llamafactory/train/rm/trainer.py`
- Core idea:
  - Train value head to score chosen > rejected completion.
- Loss:
  - pairwise logistic: `-log(sigmoid(chosen_score - rejected_score))`
- Output:
  - reward model checkpoint used later by PPO.

## 2) DPO / ORPO / SimPO
- Files:
  - `src/llamafactory/train/dpo/workflow.py`
  - `src/llamafactory/train/dpo/trainer.py`
- `pref_loss` controls objective:
  - `sigmoid`: DPO
  - `orpo`: odds-ratio policy optimization
  - `simpo`: simple preference optimization
- Ref model behavior:
  - DPO uses reference model by default.
  - ORPO/SimPO are reference-free in this implementation (`use_ref_model=false`).
- Additional knobs:
  - `pref_ftx`: SFT loss mixing
  - `pref_bco_weight`: BCO blending
  - `dpo_label_smoothing`: only for sigmoid DPO

## 3) KTO
- Files:
  - `src/llamafactory/train/kto/workflow.py`
  - `src/llamafactory/train/kto/trainer.py`
- Uses desirable/undesirable feedback split plus KL guidance.
- Data collator injects both normal and `kl_*` tensors.
- Logs include reward/logp/logit summaries and KL stats.

## 4) PPO
- Files:
  - `src/llamafactory/train/ppo/workflow.py`
  - `src/llamafactory/train/ppo/trainer.py`
  - `src/llamafactory/train/ppo/ppo_utils.py`
- Runtime loop:
  1. Generate responses from current policy.
  2. Score responses with reward model (full, lora/oft adapter, or API endpoint).
  3. Run PPO step and log reward/loss.
- Important compatibility constraint:
  - Requires `trl>=0.8.6,<=0.9.6`.
- Current gap:
  - no official PPO YAML in `examples/` even though PPO code is implemented.

## 5) Ref and Reward Model Construction
- File: `src/llamafactory/train/trainer_utils.py`
- `create_ref_model()`:
  - load explicit `ref_model` if given,
  - otherwise fallback rules based on finetuning mode.
- `create_reward_model()`:
  - `api`: HTTP scoring server
  - `lora`/`oft`: switch adapter + value head buffers
  - `full`: load separate full reward model with value head
