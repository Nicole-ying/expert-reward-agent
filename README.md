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
  → eval_result.json / training_summary.json
```

## 当前流程

- `prompt_records/` 保存 `.md`，不再保存 JSON prompt。
- `response_records/` 保存 `.md`，方便直接阅读 LLM 原始输出。
- 新 run 不再自动生成 `human_review/`、`raw_outputs/`、`final_outputs/`。
- `expert_reward_context.md` 不再把知识库原始 YAML 整段塞进 prompt。
- Reward Generator 使用 role-based component budget，不再用“最多两个组件”这种硬限制。
- 推荐 reward_v1 使用 2~4 个组件：1 个主学习信号 + 0~2 个稳定/安全约束 + 0~1 个任务完成 proxy。
- 新的推荐返回格式是：`return float(total_reward), components`。
- wrapper 已兼容旧格式 `return total_reward` 和新格式 `return total_reward, components`。
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

## 完整实验输出

生成阶段输出：

```text
runs/env_001/<GEN_RUN>/
├── environment_card.md
├── expert_reward_context.md
├── reward_v1.py
├── reward_v1.md
├── llm_inputs/
│   └── 02_reward_generator.input.md
├── prompt_records/
│   ├── 01_environment_analyzer.md
│   └── 02_reward_generator.md
├── response_records/
│   ├── 01_environment_analyzer.md
│   └── 02_reward_generator.md
└── validations/
    └── reward_v1.validation.json
```

训练阶段输出：

```text
runs/env_001/training_runs/<TRAIN_RUN>/
├── model.zip
├── train_config_used.yaml
├── eval_result.json
├── training_summary.json
└── monitor/
    ├── monitor_0.csv
    ├── monitor_1.csv
    ├── monitor_2.csv
    └── monitor_3.csv
```

TensorBoard：

```text
runs/env_001/tensorboard/
```

建议重点看：

```text
runs/env_001/<GEN_RUN>/reward_v1.py
runs/env_001/<GEN_RUN>/validations/reward_v1.validation.json
runs/env_001/training_runs/<TRAIN_RUN>/eval_result.json
runs/env_001/training_runs/<TRAIN_RUN>/training_summary.json
runs/env_001/training_runs/<TRAIN_RUN>/monitor/
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
