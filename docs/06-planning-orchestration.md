# RL for Planning and Orchestration

## Goal
Use preference learning to teach LLMs to:
- produce better multi-step plans,
- select better tool-call sequences,
- recover from failed branches.

## Data Strategy
Create pairwise samples where chosen answer has:
- clearer decomposition,
- fewer redundant tool calls,
- better constraint satisfaction,
- better recovery strategy.

Rejected answer should represent realistic planner mistakes:
- missing dependencies,
- wrong ordering,
- premature finalization,
- low-signal or irrelevant tool calls.

## Mapping to LLaMA-Factory Stages
1. `rm`:
- Train a reward model that scores planning trajectories.

2. `dpo/orpo/simpo`:
- Directly push model toward better planner outputs from pairwise data.

3. `kto`:
- Useful when feedback naturally appears as acceptable/unacceptable traces.

4. `ppo`:
- Use reward model (or reward API) to optimize generated trajectories online.

## Reward Design Ideas for Planning
- + score for valid dependency ordering
- + score for tool efficiency (fewer unnecessary actions)
- + score for constraint coverage
- - penalty for hallucinated resources
- - penalty for dead-end plans

## Practical Evaluation
Track both text quality and planner metrics:
- task success rate
- average tool calls per successful task
- recovery success after first failure
- latency and token cost

## Suggested Iteration Loop
1. Start with DPO on curated pairwise planner data.
2. Train RM from same or expanded preference set.
3. Run PPO with online-generated hard cases.
4. Mine new failure traces and feed back into DPO/KTO.
