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

## 为什么这么改

v8 的问题是中间产物太多，Reward Architect 过度保守，而且 JSON 输入太长。  
v9 改成 Markdown 环境卡片 + Markdown 专家上下文，直接给 Reward Generator 生成奖励函数。

## 当前最新调整

- `prompt_records/` 保存 `.md`，不再保存 JSON prompt。
- `response_records/` 保存 `.md`，方便直接阅读 LLM 原始输出。
- 已删除新 run 中的 `human_review/`、`raw_outputs/`、`final_outputs/` 自动生成逻辑。
- 核心输出文件直接放在 run 根目录，避免同一内容多处重复。
- `expert_reward_context.md` 不再把知识库原始 YAML 整段塞进 prompt。
- RAG 检索结果会被整理成“奖励骨架摘要”：角色、数学形态、需要信号、本轮建议、风险、后续迭代。
- Reward Generator 使用 role-based component budget，不再用“最多两个组件”这种硬限制。
- 推荐 reward_v1 使用 2~4 个组件：1 个主学习信号 + 0~2 个稳定/安全约束 + 0~1 个任务完成 proxy。
- 新的推荐返回格式是：`return float(total_reward), components`。
- wrapper 已兼容旧格式 `return total_reward` 和新格式 `return total_reward, components`。
- validator 会检查 import、未声明 info 字段、obs 切片、components dict 和 tuple 返回格式。

## 运行 mock

```bash
bash run_mock_pipeline.sh
```

## 真实 DeepSeek 运行

```bash
export DEEPSEEK_API_KEY="你的key"
python -m pipeline.run_direct_generation_pipeline \
  --config configs/env001_deepseek_rag.yaml \
  --run-name deepseek_run_002
```

## 新 run 的核心结构

```text
runs/env_001/deepseek_run_002/
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

建议主要看：

```text
reward_v1.py
reward_v1.md
validations/reward_v1.validation.json
expert_reward_context.md
prompt_records/02_reward_generator.md
```

## 是否让 Reward Generator 看 step 函数

默认不看。  
如果测试后仍然越界使用信号，可在 config 中打开：

```yaml
context:
  include_masked_step_in_reward_generator: true
```
