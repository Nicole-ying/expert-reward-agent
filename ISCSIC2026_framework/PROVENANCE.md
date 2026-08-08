# Paper-v4 来源证明

- 冻结源提交：`cafceeb9`
- 提交说明：`feat: paper v4 framework — structured diagnosis as core, remove bloat`
- 恢复范围：`configs/`、`envs/`、`knowledge_base/`、`llm_clients/`、`pipeline/`、`prompts/`、`rag/`、`training/`、`baselines/`、`scripts/`、`run_paper_v4.sh`、`run_unconstrained_ablation.sh`、`requirements_bridge.txt`
- `requirements_bridge.txt` 在投稿包中重命名为 `requirements.txt`，文件内容不变。

三个实际固定 System Prompt 与归档 `runs/env_001/paper_v4` Prompt record 的内容哈希一致：

| Prompt | SHA-256 前 16 位 |
|---|---|
| `01_environment_analyzer_prompt.md` | `c086d7c5635ea4e7` |
| `02_reward_generator_prompt.md` | `c7629de893bb9083` |
| `reflection_agent_prompt.md` | `69c223915de8fa41` |

归档记录中没有 `subagent_trace_*.json` 或 `subagent_signal_*.md`，因此后续 Subagent 机制被排除在本快照之外。
