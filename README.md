# LLaMA-Factory RL Analysis Playbook

Practical onboarding docs for RL training in [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), with focus on:
- Preference learning (RM, DPO, ORPO, SimPO, KTO, PPO)
- How source code is organized and executed
- How to become productive quickly as a new engineer
- How to adapt this stack for LLM planning and orchestration tasks

## Who This Is For
- New engineers exploring RL training for LLMs
- Applied researchers who need code-level understanding, not only theory
- Teams building planner/orchestrator style LLM systems

## Repository Structure
- `docs/00-learning-roadmap.md`: 2-4 week learning path
- `docs/01-codebase-architecture.md`: end-to-end architecture walk-through
- `docs/02-rl-stages-deep-dive.md`: RM/DPO/KTO/PPO internals
- `docs/03-data-and-collators.md`: dataset schema and processor details
- `docs/04-hands-on-quickstart.md`: practical commands and configs
- `docs/05-common-failures.md`: troubleshooting guide
- `docs/06-planning-orchestration.md`: applying RL to planning/orchestration behavior

## Upstream Snapshot Analyzed
- Upstream repo: `hiyouga/LLaMA-Factory`
- Branch: `main`
- Commit analyzed: `f80e15db` (2026-02-12)

## Fast Start
1. Read `docs/00-learning-roadmap.md`
2. Skim `docs/01-codebase-architecture.md`
3. Run one RM + one DPO experiment from `docs/04-hands-on-quickstart.md`
4. Use `docs/06-planning-orchestration.md` to map RL setup to your planner tasks

## Scope Note
Upstream currently ships ready YAML examples for `rm`, `dpo`, `kto`, but not an explicit `ppo` YAML in `examples/`. PPO is still implemented in code (`src/llamafactory/train/ppo`).
