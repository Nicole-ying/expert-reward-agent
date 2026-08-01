# CREATE — ISCSIC 2026 论文与 paper-v4 冻结框架

本目录保存论文 **CREATE: A Closed-Loop Reward Editing Agent with Training Evidence for Reinforcement Learning**，以及与历史 `paper_v4` 主实验一致的实现快照。

## 冻结基准

- 历史源提交：`cafceeb9`（`feat: paper v4 framework — structured diagnosis as core, remove bloat`）
- 主配置：`configs/env001_paper_v4.yaml`
- 主入口：`run_paper_v4.sh`
- 5 个独立 seed：0–4
- 每条搜索 lineage 最多 10 次 reward evaluation
- 每个 reward candidate 从头训练 PPO：1,000,000 timesteps
- 每次 native evaluation：20 episodes

本快照不包含 paper-v4 之后探索的 Subagent investigator、Component delta、Formula switching guide 或附加累计记录段。它们没有参与归档 `paper_v4` 实验，因此不属于本论文方法证据。

## 论文主张

CREATE 将一次策略训练产生的外部任务结果和 reward-component 统计转化为下一轮奖励修复的结构化观察。它在持久 lineage memory 的支持下执行一次可验证的 L1 参数调整、L2 数学结构重构或 L3 奖励骨架重设，并始终用未修改的环境原生目标评价候选奖励。这里的“自进化”指奖励程序及其诊断 lineage 在 `reward → train → evaluate → diagnose → edit` 闭环中演化；LLM 权重不发生更新。

## 目录

- `paper/`：LaTeX 源文件、PDF、可编辑 SVG 和论文图表。
- `pipeline/`：paper-v4 的生成、反思、验证、记忆与控制器逻辑。
- `training/`：Stable-Baselines3 PPO 训练和 native evaluation。
- `prompts/`：历史固定 System Prompt。
- `configs/`：历史主配置与当时保留的消融配置。
- `envs/`：匿名任务接口与 masked step source。
- `knowledge_base/`、`rag/`：初始生成和反思可调用的专家知识。
- `baselines/`、`scripts/`：历史提交中已有的辅助入口。

详细语义与代码对应见 [ARCHITECTURE.md](ARCHITECTURE.md)，全部 Prompt 结构的中英文归档见 [PROMPTS_BILINGUAL.md](PROMPTS_BILINGUAL.md)，实验复现边界见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

## 运行主实验

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY="..."
bash run_paper_v4.sh
```

输出写入 `runs/env_001/`，不纳入本投稿源码包。

## 编译论文

```bash
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
