# End-to-End Runbook: Train Agent Tool-Calling Multi-Turn

## 1) Chuẩn bị
- Chốt model base.
- Chốt tool schema.
- Chốt benchmark/KPI.
- Freeze dataset version.

## 2) Train sequence
1. Train RM.
2. Train DPO hoặc KTO.
3. (Optional) PPO online optimization.
4. Evaluate full benchmark.

## 3) Artifact quản lý
Mỗi run cần lưu:
- config YAML,
- git commit hash,
- tokenizer/template version,
- checkpoint path,
- metrics snapshot.

## 4) Cấu trúc experiment tracking
- Project: `toolcall-agent-multiturn`
- Groups: `rm`, `dpo`, `kto`, `ppo`
- Tags: model, dataset_version, template, objective

## 5) Go/No-Go checklist
1. Task success tăng ổn định qua 2 benchmark rounds.
2. Không tụt mạnh ở failure recovery.
3. Không tăng quá mức avg tool calls.
4. Trajectory có thể giải thích được bằng rubric.

## 6) Deployment preflight
- Inference template trùng training template.
- Guardrails bật ở production.
- Canary rollout theo traffic nhỏ.
- Monitor lỗi tool args theo thời gian thực.

## 7) Kế hoạch iteration
- Thu thập failure trajectories tuần tự.
- Ưu tiên hard-negative mining cho DPO/KTO.
- Re-train định kỳ theo data freshness.
