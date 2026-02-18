# LLaMA-Factory RL Handbook (Vietnamese)

Bộ tài liệu chi tiết để đọc hiểu, cấu hình, và vận hành RL training cho LLM bằng LLaMA-Factory, tập trung vào bài toán tool-calling multi-turn và planning/orchestration.

## Upstream Snapshot
- Mã nguồn phân tích: `https://github.com/hiyouga/LLaMA-Factory`
- Commit tham chiếu: `f80e15db` (2026-02-12)
- Phạm vi thuật toán RL trong source hiện tại:
  - Có: `RM`, `PPO`, `DPO`, `ORPO`, `SimPO`, `KTO`
  - Không thấy implementation: `GRPO`

## Mục lục tài liệu
- `docs/00-index.md`
- `docs/01-training-config-meaning.md`
- `docs/02-source-architecture-vs-trl.md`
- `docs/03-rl-framework-architecture.md`
- `docs/04-tool-calling-setup-notes.md`
- `docs/05-example-qwen-ppo-toolcalling-multiturn.md`
- `docs/06-example-gemma-dpo-toolcalling-multiturn.md`
- `docs/07-example-gpt-oss-grpo-toolcalling-multiturn.md`
- `docs/08-ppo-from-source.md`
- `docs/09-grpo-from-current-source.md`
- `docs/10-dpo-kto-orpo-from-source.md`
- `docs/11-other-rl-algorithms-supported.md`
- `docs/series-toolcalling-agent-multiturn/` (series moi: train agent tool-calling multi-turn, tu data den training)

## Đọc nhanh theo nhu cầu
- Muốn chạy ngay: đọc `docs/04`, `docs/05`, `docs/06`.
- Muốn hiểu sâu code: đọc `docs/02`, `docs/08`, `docs/10`.
- Muốn thiết kế framework RL tổng quát: đọc `docs/03`.
- Muốn GRPO: đọc `docs/09` + `docs/07`.
- Muốn series full tu data den production runbook cho tool-calling agent: doc thu muc `docs/series-toolcalling-agent-multiturn/`.
