# Paper-v4 architecture

## Closed loop

1. **Task interface and initialization.** The environment card, task
   specification, masked transition interface, and retrieved reward-design
   evidence ground an initial executable reward `R_0`.
2. **Fresh policy training and native evaluation.** Every candidate reward
   trains a fresh PPO policy. Selection uses the unchanged environment reward,
   so CREATE cannot grade itself with its generated shaping reward.
3. **Observe.** The agent receives the native score, training trace, per-term
   reward statistics (`episode_sum_mean`, `signed_share`, `magnitude_share`, and
   `active_rate`), the active reward, and its lineage record.
4. **Diagnose and act.** The reward editor forms one diagnosis and chooses one
   primary target. It applies an L1 parameter tune, L2 structural refactor, or
   L3 reward redesign.
5. **Validate, remember, and archive.** The candidate is checked for interface
   and safety violations. The transition record is appended to persistent
   memory; the best archive changes only when native evaluation improves.
6. **Repeat or stop.** The validated reward becomes `R_{t+1}`. Search stops at
   the task threshold or the fixed evaluation budget.

This stateful observe-diagnose-act-update cycle is the reason CREATE is an
agent rather than a sequence of unrelated LLM calls.

## Method-to-code map

| Paper mechanism | Primary implementation |
|---|---|
| Iterative closed loop | `pipeline/run_iterative_experiment.py` |
| Structured training evidence | `training/train_sb3_wrapper.py` |
| Diagnosis and bounded L1/L2/L3 edit | `pipeline/run_reflection_agent.py`, `prompts/reflection_agent_prompt.md` |
| Optional investigator signal | `pipeline/subagent_investigator.py` |
| Reward validation and repair guard | validation logic in `pipeline/run_iterative_experiment.py` |
| Persistent lineage memory | `pipeline/run_06_update_reward_memory.py` and experiment records under `runs/` |
| Best-reward archive | best-retention logic in `pipeline/run_iterative_experiment.py` |
| Expert reward context | `rag/`, `knowledge_base/`, and `pipeline/run_02_build_expert_context.py` |
| Fresh PPO and native evaluation | `training/train_sb3_wrapper.py` |

## Paper-aligned configurations

- `configs/env001_paper_v4.yaml`: reported CREATE condition, five lineages,
  structured evidence, persistent memory, bounded reflection, and investigator.
- `configs/env001_ablation_score_only_v4.yaml`: removes component evidence.
- `configs/env001_ablation_eureka_feedback_v4.yaml`: supplies score and coarse
  component means rather than CREATE's structured diagnostic table.
- `configs/env001_ablation_unconstrained_v4.yaml`: retains evidence and memory
  but removes the bounded L1/L2/L3 reflection contract.
- `scripts/run_independent_baseline.sh`: uses the same initial-generation path,
  task context, LLM configuration, PPO budget, and ten evaluations per lineage,
  but no repair feedback or lineage memory.
- `scripts/run_official_baseline.sh`: PPO trained on the unchanged environment
  reward; its reward adapter is `baselines/official_reward.py`.

## Semantic boundaries used by the manuscript

- **Training reward:** editable program used to train PPO.
- **Native outcome:** unchanged environment score used for selection and
  reporting.
- **Persistent memory:** chronological edit-diagnosis-outcome lineage; it is not
  the best archive.
- **Best archive:** incumbent reward with the highest native outcome; it is not
  automatically replaced by the latest candidate.
- **Bounded repair:** one primary diagnosed target and one declared L1/L2/L3
  intervention, followed by executable validation.
- **Self-evolution:** verified evolution of the reward artifact and its lineage,
  with fixed LLM weights.
