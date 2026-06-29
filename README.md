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

第一次训练后新增的迭代流程：

```text
training_feedback.md
  → Iteration Context Builder（确定性脚本，不调用 LLM）
  → iteration_context.md
  → Reward Revision LLM
  → reward_v2.py / reward_v2.md
  → PPO 训练
```

第二次训练后重复同样流程，生成 `reward_v3.py / reward_v3.md` 并训练。

## 当前流程

- `prompt_records/` 保存 `.md`，不再保存 JSON prompt。
- `response_records/` 保存 `.md`，方便直接阅读 LLM 原始输出。
- 新 run 不再自动生成 `human_review/`、`raw_outputs/`、`final_outputs/`。
- `expert_reward_context.md` 不再把知识库原始 YAML 整段塞进 prompt。
- Reward Generator 使用 role-based component budget，不再用“最多两个组件”这种硬限制。
- Reward Revision LLM 使用 `iteration_context.md`，只根据训练证据做继承式修订。
- 新的推荐返回格式是：`return float(total_reward), components`。
- wrapper 已兼容旧格式 `return total_reward` 和新格式 `return float(total_reward), components`。
- wrapper 已加入 reward 异常兜底、NaN/inf 检查、reward clipping 和错误计数。
- 训练完成后会用原始环境 reward 做 external evaluation。

## 一键 3 轮训练 / 2 轮迭代

### 10k smoke test

```bash
export DEEPSEEK_API_KEY="你的key"
bash scripts/run_three_round_experiment.sh \
  configs/env001_deepseek_rag.yaml \
  smoke3 \
  10000 \
  5
```

### 1e6 正式实验

```bash
export DEEPSEEK_API_KEY="你的key"
bash scripts/run_three_round_experiment.sh \
  configs/env001_deepseek_rag.yaml \
  exp3 \
  1000000 \
  10
```

这个脚本会执行：

```text
v1: Environment Analyzer LLM → RAG → Reward Generator LLM → train
v2: Iteration Context Builder → Reward Revision LLM → train
v3: Iteration Context Builder → Reward Revision LLM → train
```

### mock 三轮流程测试

```bash
bash scripts/run_three_round_experiment.sh \
  configs/env001_deepseek_rag.yaml \
  mock3 \
  10000 \
  5 \
  --mock
```

## 单轮完整实验

```bash
export DEEPSEEK_API_KEY="你的key"
bash scripts/run_full_experiment.sh \
  configs/env001_deepseek_rag.yaml \
  deepseek_full_run_001 \
  ppo_full_run_001 \
  1000000 \
  10
```

## 人类主要看哪些文件

生成阶段重点看：

```text
runs/env_001/<GEN_RUN>/reward_v1.py 或 reward_v2.py / reward_v3.py
runs/env_001/<GEN_RUN>/reward_v1.md 或 reward_v2.md / reward_v3.md
runs/env_001/<GEN_RUN>/prompt_records/*.prompt_stats.md
runs/env_001/<GEN_RUN>/validations/*.validation.json
```

训练阶段重点看：

```text
runs/env_001/training_runs/<TRAIN_RUN>/training_feedback.md
```

跨轮迭代重点看：

```text
runs/env_001/memory/reward_memory.md
runs/env_001/iterations/<ITER_NAME>/iteration_context.md
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

## 生成迭代上下文

当前实验进入下一轮时，先生成一个唯一的迭代上下文文件：

```bash
python -m pipeline.run_04_build_iteration_context \
  --train-run-dir runs/env_001/training_runs/ppo_full_run_002 \
  --out runs/env_001/iterations/iter_003/iteration_context.md
```

这个文件合并：

```text
training_feedback.md
+ reward_memory.md
+ matched compressed expert cards
→ iteration_context.md
```

`iteration_context.md` 是后续奖励迭代 LLM 的主输入，包含：

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
