# 11_v8_initial_chain_rebuild 说明

v8 重构初始奖励函数生成链路。

## 核心变化

Environment Analyzer 不再选择 reward skeleton，也不再输出 `v1_candidate_skeletons`。

它只负责：环境目标、任务类型 selected_route_id、observation_space、action_space、step 函数分析、termination 分析、info 字段分析、reward function interface、environment_interface_contract。

Reward Architect 才负责从 17 个 skeleton 中选择、组合、延后或拒绝。

Reward Coder 只负责实现 `reward_design_plan.formula_plan.core_terms`。

## Step 函数给谁看

完整 masked_step_source 给 Environment Analyzer 看。Reward Architect 和 Reward Coder 默认只看 environment_interface_contract。若最后测试仍越界，可以在 config 打开 optional_debug_include_masked_step_for_coder。

## 新增/重整文件

```text
environment_interface_contract.json
coder_environment_context.json
llm_inputs/02_reward_architect.input.json
llm_inputs/03_reward_coder.input.json
human_review/02_reward_architect.input.md
human_review/03_reward_coder.input.md
validations/02_reward_architect.validation.json
validations/03_reward_coder.validation.json
```
