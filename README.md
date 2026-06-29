# expert_eureka_env001_bridge_v9_direct_generator

v9 删除 Reward Architect / 奖励骨架 LLM，改成两次 LLM 调用：

```text
Environment Analyzer LLM
  → environment_card.md

RAG 压缩器
  → expert_reward_context.md

Reward Generator LLM
  输入：environment_card.md + expert_reward_context.md
  输出：reward_v1.py + reward_v1.md
```

随后进入训练：

```text
reward_v1.py
  → RewardOverrideWrapper
  → PPO 训练
  → 原始 LunarLander-v3 external evaluation
  → training_feedback.md / training_summary.json
```

## 当前流程

- `prompt_records/` 保存 `.md`，不再保存 JSON prompt。
- `response_records/` 保存 `.md`，方便直接阅读 LLM 原始输出。
- 新 run 不再自动生成 `human_review/`、`raw_outputs/`、`final_outputs/`。
- `expert_reward_context.md` 不再把知识库原始 YAML 整段塞进 prompt。
- Reward Generator 使用 role-based component budget，不再用“最多两个组件”这种硬限制。
- 推荐 reward_v1 使用 2~4 个组件：1 个主学习信号 + 0~2 个稳定/安全约束 + 0~1 个任务完成 proxy。
- 新的推荐返回格式是：`return float(total_reward), components`。
- wrapper 已兼容旧格式 `return total_reward` 和新格式 `return float(total_reward), components`。
- wrapper 已加入 reward 异常兜底、NaN/inf 检查、reward clipping 和错误计数。
- 训练完成后会用原始环境 reward 做 external evaluation。

## 一键完整实验

### 10k smoke test

```bash
export DEEPSEEK_API_KEY="你的key"
bash scripts/run_full_experiment.sh \
  configs/env001_deepseek_rag.yaml \
  deepseek_full_smoke_001 \
  ppo_full_smoke_001 \
  10000 \
  5
```

这个命令会完整执行：

```text
环境理解 LLM
→ RAG 压缩
→ Reward Generator LLM
→ reward_v1 validator
→ PPO 训练
→ 原环境 external evaluation
```

### 1e6 正式实验

```bash
export DEEPSEEK_API_KEY="你的key"
bash scripts/run_full_experiment.sh \
  configs/env001_deepseek_rag.yaml \
  deepseek_full_run_001 \
  ppo_full_run_001 \
  1000000 \
  10
```

### mock 全流程测试

不调用 DeepSeek，只测试流程是否能跑通：

```bash
bash scripts/run_full_experiment.sh \
  configs/env001_deepseek_rag.yaml \
  mock_full_run_001 \
  ppo_mock_full_001 \
  10000 \
  5 \
  --mock
```

## 人类主要看哪些文件

生成阶段重点看：

```text
runs/env_001/<GEN_RUN>/reward_v1.py
runs/env_001/<GEN_RUN>/reward_v1.md
runs/env_001/<GEN_RUN>/prompt_records/02_reward_generator.prompt_stats.md
runs/env_001/<GEN_RUN>/validations/reward_v1.validation.json
```

训练阶段重点看：

```text
runs/env_001/training_runs/<TRAIN_RUN>/training_feedback.md
```

跨轮迭代重点看：

```text
runs/env_001/memory/reward_memory.md
knowledge_base/iteration/reward_misalignment_cards.md
```

机器日志可以保留，但不要作为 LLM 主输入：

```text
training_summary.json
eval_result.json
monitor/
model.zip
```

## 迭代文件原则

- 不把 reward 路径作为 memory 的核心信息；路径只适合人工追溯，LLM 更需要 reward 结构和关键证据。
- `reward_memory.md` 只记录：组件结构、外部得分、episode length、关键组件信号、诊断、下一步动作。
- 不复制完整 reward 代码、完整日志、完整 summary 到 memory。
- failure / reward-misalignment 知识必须压缩成短卡片。
- 每轮只把命中的 2~4 张专家卡片放入迭代上下文，不传整份知识库。
- 迭代不是机械填充骨架，而是根据训练反馈决定 keep / weaken / revise / consider_add / still_defer。

## 迭代阶段建议结构

```text
training_feedback.md
+ reward_memory.md
+ matched compressed expert cards
→ iteration_context.md
→ reward_v2.py / reward_v2.md
```

`iteration_context.md` 后续应成为奖励迭代 LLM 的主输入，包含：

```text
previous_reward_summary
training_feedback_summary
matched_failure_and_misalignment_cards
skeleton_revision_plan
generation_constraints
```

## 单独生成 reward

```bash
python -m pipeline.run_direct_generation_pipeline \
  --config configs/env001_deepseek_rag.yaml \
  --run-name deepseek_run_002
```

## 单独训练已有 reward

```bash
python -m training.train_sb3_wrapper \
  --config configs/env001_deepseek_rag.yaml \
  --reward runs/env_001/deepseek_run_002/reward_v1.py \
  --run-name ppo_reward_v1_001 \
  --total-timesteps 1000000 \
  --eval-episodes 10
```

## 是否让 Reward Generator 看 step 函数

默认不看。  
如果测试后仍然越界使用信号，可在 config 中打开：

```yaml
context:
  include_masked_step_in_reward_generator: true
```
