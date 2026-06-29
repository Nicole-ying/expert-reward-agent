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

## 这次最新调整

- `prompt_records/` 不再保存 JSON，改为保存同名 `.md` 文件。
- `response_records/` 也改为 `.md`，方便人工阅读。
- `expert_reward_context.md` 不再把知识库原始 YAML 整段塞进 prompt。
- RAG 检索结果现在会被整理成“奖励骨架摘要”：角色、数学形态、需要信号、本轮建议、风险、后续迭代。
- 例如 `potential_based_shaping` 不再传完整文献和长 YAML，只保留：数学形态、需要 Phi、本轮不首选、风险和后续用途。

## 运行 mock

```bash
bash run_mock_pipeline.sh
```

## 真实 DeepSeek 运行

```bash
export DEEPSEEK_API_KEY="你的key"
python -m pipeline.run_direct_generation_pipeline \
  --config configs/env001_deepseek_rag.yaml \
  --run-name deepseek_run_001
```

## 核心文件

```text
runs/env_001/deepseek_run_001/environment_card.md
runs/env_001/deepseek_run_001/expert_reward_context.md
runs/env_001/deepseek_run_001/llm_inputs/02_reward_generator.input.md
runs/env_001/deepseek_run_001/reward_v1.py
runs/env_001/deepseek_run_001/reward_v1.md
runs/env_001/deepseek_run_001/validations/reward_v1.validation.json
```

## Prompt 记录

现在看这里：

```text
runs/env_001/deepseek_run_001/prompt_records/01_environment_analyzer.md
runs/env_001/deepseek_run_001/prompt_records/02_reward_generator.md
```

不再需要看 `.json` prompt 记录。

## 是否让 Reward Generator 看 step 函数

默认不看。  
如果测试后仍然越界使用信号，可在 config 中打开：

```yaml
context:
  include_masked_step_in_reward_generator: true
```
