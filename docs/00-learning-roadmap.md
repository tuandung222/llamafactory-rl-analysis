# Learning Roadmap

## Week 1: Build Correct Mental Model
1. Read architecture flow in `docs/01-codebase-architecture.md`.
2. Understand stage router in `src/llamafactory/train/tuner.py`.
3. Inspect argument validation in `src/llamafactory/hparams/finetuning_args.py`.
4. Learn data processors for `rm`, `dpo`, `kto`, `ppo`.

Outcome:
- You can explain what happens from CLI to training loop.

## Week 2: Run and Modify Baselines
1. Run reward model training (`stage: rm`).
2. Run preference training (`stage: dpo` with `pref_loss=sigmoid`).
3. Switch to `pref_loss=orpo` and `pref_loss=simpo` to compare behavior.
4. Run `stage: kto` and inspect logged metrics.

Outcome:
- You can execute and compare 3+ preference algorithms.

## Week 3: PPO and Reward Integration
1. Build or select a reward model.
2. Configure PPO with correct `reward_model` and `reward_model_type`.
3. Validate reward path (local model, adapter, or API server).

Outcome:
- You can run PPO end-to-end and debug reward pipeline issues.

## Week 4: Planning and Orchestration Adaptation
1. Convert planner trajectories into preference datasets.
2. Define success/failure reward heuristics for planning tasks.
3. Fine-tune and evaluate on orchestration metrics.

Outcome:
- You can align model behavior for multi-step planning and tool orchestration.
