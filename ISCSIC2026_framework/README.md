# CREATE — ISCSIC 2026 reproducibility package

This directory is the clean, paper-aligned snapshot of **CREATE: A Closed-Loop
Reward Editing Agent with Training Evidence for Reinforcement Learning**. It
contains the five-page manuscript, the paper-v4 implementation, prompts,
environment interface, knowledge base, and runnable experiment configurations.

## Central claim

CREATE turns failed policy-training runs into structured observations for a
persistent reward-editing agent. Component-level evidence localizes a likely
reward defect; one bounded L1/L2/L3 edit makes the intervention testable;
persistent lineage memory connects edits to outcomes; and a best archive keeps
later failures from overwriting a successful reward.

In this repository, **self-evolution** has a deliberately precise meaning: the
reward program and its diagnostic lineage evolve through verified
`reward -> train -> evaluate -> diagnose -> edit` transitions. The LLM weights
remain fixed. The claim is therefore reward-program self-evolution mediated by
an agent, not autonomous model self-improvement.

## Layout

- `paper/`: complete LaTeX source, five-page PDF, editable framework SVG, vector
  plots, and submission material.
- `pipeline/`: closed-loop orchestration, reflection, validation, memory, and
  archive logic.
- `training/`: fresh PPO training and unchanged native-task evaluation.
- `prompts/`: environment analysis, initial reward generation, and repair-agent
  prompts.
- `configs/`: the main paper-v4 condition and current ablations.
- `envs/`: anonymized task specification and masked environment interface.
- `knowledge_base/` and `rag/`: reward-design evidence supplied to the agent.
- `scripts/`: five-seed experiment entry points.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the method-to-code map and
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) before rerunning the ablations. The
complete bilingual System/User prompt audit is in
[PROMPTS_BILINGUAL.md](PROMPTS_BILINGUAL.md).

## Setup and main run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY="..."
bash scripts/run_paper_v4.sh
```

Run the prompt- and budget-matched independent-generation comparison with
`bash scripts/run_independent_baseline.sh`.

Run commands from this directory. Outputs are intentionally excluded from Git
and are written under `runs/env_001/`; TensorBoard events are written under
`runs/env_001/tensorboard/`.

## Rebuild the paper

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
