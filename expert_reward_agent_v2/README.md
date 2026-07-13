# Expert Reward Agent v2

This isolated package implements an evidence-driven reward self-evolution agent. It does not modify or import the existing experiment pipeline.

## Implemented in phase 1

- Explicit state machine from task perception to confirmed stopping.
- Structured environment, candidate, evidence, intervention, and stop schemas.
- Working, episodic, semantic, and cross-run case memory.
- Phase-scoped tool registry with append-only audit records.
- Three-tier behavior adapter design: verified plugin, restricted generated DSL, generic fallback.
- Expert intervention levels: scale repair, mathematical transformation, structural rebuild.
- Separate engineering retries and scientific iteration budgets.
- Deterministic mock backend for end-to-end verification.

## Run

From the repository root:

```powershell
python -m expert_reward_agent_v2.cli --run-dir runs_v2/mock_run
python -m unittest discover -s expert_reward_agent_v2/tests -v
```

## DeepSeek API check

```powershell
$env:DEEPSEEK_API_KEY="your-key"
python -m expert_reward_agent_v2.cli --mode api-check --config expert_reward_agent_v2/configs/env001_api_smoke.yaml
```

## Env_001 production smoke run

The supplied configuration runs at most three scientific iterations, each with 300k SB3 steps and 20 fixed-seed evaluation episodes. A solved candidate is evaluated again with a disjoint seed range.

```powershell
$env:DEEPSEEK_API_KEY="your-key"
python -m expert_reward_agent_v2.cli --mode run --config expert_reward_agent_v2/configs/env001_api_smoke.yaml --run-dir runs_v2/env001_api_smoke_seed0
```

The formal single-seed configuration uses at most ten iterations with 1M SB3 steps per iteration:

```powershell
python -m expert_reward_agent_v2.cli --mode run --config expert_reward_agent_v2/configs/env001_formal_1m.yaml --run-dir runs_v2/env001_formal_seed0
```

The production backend reuses `training.train_sb3_wrapper`, but the v2 controller owns planning, validation, iteration accounting, best retention, confirmation, memory, and stopping.

## Prompt sources

- `prompts/planner_system.md`: initial planning and evidence-based reflection policy.
- `prompts/generator_system.md`: reward-code contract and safety constraints.
- `prompts/expert_reasoning_core.md`: mandatory expert questions, functional roles, mathematical forms, and their risks.
- `knowledge/ontology.py`: machine-validated names for expert dimensions, component roles, mathematical forms, and structural transformations.

User prompts are assembled from the current contract, evidence, memory, and intervention. Every fully rendered prompt is preserved in the run directory.

## Run artifacts

Each production run has the following inspectable layout:

```text
<run-dir>/
  agent_state.json
  inputs/
    perception.json
    environment_contract.json
    retrieved_knowledge.json
  llm_calls/
    001_initial_plan_attempt_0/
      system_prompt.md
      user_prompt.md
      raw_response.txt
      parsed_response.json
    002_diagnosis_attempt_0/
      ...
  artifacts/
    generations/candidate_01/
      system_prompt.md
      user_prompt_attempt_0.md
      raw_response_attempt_0.txt
      response_attempt_0.json
      validation_attempt_0.json
      reward.py
      candidate.json
    iter_01/
      reward.py
      training.stdout.log
      training.stderr.log
      training/
        model.zip
        training_summary.json
        eval_result.json
        training_feedback.md
        component_stats.md
        monitor/
        confirmation_eval.json
  memory/
    working.json
    episodes.jsonl
  audit/
    tool_calls.jsonl
```

`confirmation_eval.json` is created only after a candidate first reaches the target. TensorBoard event files use the directory configured by the referenced legacy training config; their exact path is also recorded in `training_summary.json`.
