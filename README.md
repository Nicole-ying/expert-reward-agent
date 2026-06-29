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
  → reward_memory.md 自动更新
```

第一次训练后新增的迭代流程：

```text
training_feedback.md + reward_memory.md
  → Iteration Context Builder（确定性脚本，不调用 LLM）
  → iteration_context.md
  → Reward Revision LLM
  → reward_v2.py / reward_v2.md
  → PPO 训练
  → reward_memory.md 自动更新
```

后续轮次重复同样流程，直到达到 `iteration.total_rounds`。

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
- 每轮训练结束后会自动更新 `runs/env_001/memory/reward_memory.md`。

## 配置驱动的任意轮迭代

迭代轮数写在 config 中：

```yaml
iteration:
  total_rounds: 3        # 想跑 10 轮就改成 10
  experiment_prefix: exp_iter
  experiment_root: runs/env_001/experiments
  memory_path: runs/env_001/memory/reward_memory.md
  reset_memory_at_start: true
  cards_top_k: 4
  stop_on_invalid_reward: true
  use_mock_llm: false
```

训练规模也写在 config 中：

```yaml
training:
  total_timesteps: 1000000
  eval_episodes: 10
```

启动任意轮迭代实验：

```bash
export DEEPSEEK_API_KEY="你的key"
bash scripts/run_iterative_experiment.sh configs/env001_deepseek_rag.yaml exp_iter
```

也可以直接运行 Python orchestrator：

```bash
python -m pipeline.run_iterative_experiment \
  --config configs/env001_deepseek_rag.yaml \
  --prefix exp_iter
```

如果临时覆盖轮数，而不改 config：

```bash
python -m pipeline.run_iterative_experiment \
  --config configs/env001_deepseek_rag.yaml \
  --prefix exp10 \
  --rounds 10
```

mock 流程测试：

```bash
bash scripts/run_iterative_experiment.sh configs/env001_deepseek_rag.yaml mock_iter --mock
```

旧入口 `scripts/run_three_round_experiment.sh` 已废弃，只作为兼容包装器。

## 方法借鉴

本项目借鉴 `frde-hdrc` 的方法，而不是照搬其内容：

- 用历史 reward 和分数指导下一轮，不每轮从零生成；
- 根据 score trend、episode length、组件结构判断 tune / add / delete / rebuild；
- 强制检查组件方向、尺度和冗余；
- 只记录 compact memory，而不是全量日志；
- 当前项目额外加入 RAG：每轮只给命中的 failure / reward-misalignment 专家卡片。

## 输出目录组织

每轮输出按统一目录组织：

```text
runs/env_001/experiments/<PREFIX>/iter_01/
runs/env_001/experiments/<PREFIX>/iter_02/
runs/env_001/experiments/<PREFIX>/iter_03/
...
runs/env_001/experiments/<PREFIX>/iter_10/
```

每轮训练输出在：

```text
runs/env_001/training_runs/experiments/<PREFIX>/iter_XX/training/
```

每轮执行逻辑：

```text
iter_01: Environment Analyzer LLM → RAG → Reward Generator LLM → train → update memory
iter_02+: Build iteration_context → Reward Revision LLM → train → update memory
```

## 人类主要看哪些文件

每轮 reward 生成或修订结果：

```text
runs/env_001/experiments/<PREFIX>/iter_XX/generation/reward_vX.py
runs/env_001/experiments/<PREFIX>/iter_XX/generation/reward_vX.md
runs/env_001/experiments/<PREFIX>/iter_XX/generation/prompt_records/*.prompt_stats.md
runs/env_001/experiments/<PREFIX>/iter_XX/generation/validations/*.validation.json
```

每轮训练反馈：

```text
runs/env_001/training_runs/experiments/<PREFIX>/iter_XX/training/training_feedback.md
```

跨轮迭代重点看：

```text
runs/env_001/memory/reward_memory.md
runs/env_001/experiments/<PREFIX>/iter_XX/iteration_context.md
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
- 每轮只把命中的 `cards_top_k` 张专家卡片放入迭代上下文，不传整份知识库。
- 迭代不是机械填充骨架，而是根据训练反馈决定 keep / weaken / revise / consider_add / still_defer。

## 单独生成迭代上下文

```bash
python -m pipeline.run_04_build_iteration_context \
  --train-run-dir runs/env_001/training_runs/experiments/exp_iter/iter_01/training \
  --memory runs/env_001/memory/reward_memory.md \
  --cards knowledge_base/iteration/reward_misalignment_cards.md \
  --top-k 4 \
  --out runs/env_001/experiments/exp_iter/iter_02/iteration_context.md
```

## 单独更新 memory

```bash
python -m pipeline.run_06_update_reward_memory \
  --iter 1 \
  --train-run-dir runs/env_001/training_runs/experiments/exp_iter/iter_01/training \
  --memory runs/env_001/memory/reward_memory.md
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
