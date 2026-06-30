你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境背景；
2. expert_reward_context.md：RAG 检索并压缩后的专家知识；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务：
直接生成第一版奖励函数 `reward_v1.py`，并附带一份简短设计说明。

# 总体设计原则

- 从简单到复杂，但“简单”不等于只有一个组件。
- 不要用“最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 不要机械照抄 route 推荐公式。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 对 Env_001 这类二维任务，禁止把位置写成三维。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# role-based component budget

reward_v1 推荐使用 2~4 个组件，但每个组件必须有明确角色，不能为了显得完整而堆叠。

必须包含：
- 1 个主学习信号 learning guidance：通常选择 progress_delta_reward；如果无法计算 delta，才用 distance_reward。

允许包含：
- 0~2 个稳定/安全约束：例如速度、姿态角、角速度惩罚。
- 0~1 个任务完成 proxy：必须是 soft proxy，不能伪造 success flag；如果使用 contact，必须与 near_target、low_speed、stable_angle 等条件组合，并且权重较小。
- 0~1 个效率/动作代价：默认后续版本再加；如果 v1 使用，权重要很小，并说明风险。

默认不要在 v1 使用（如果环境条件不满足）：
- terminal_success_reward（需显式 success flag）
- terminal_failure_penalty（需显式 failure flag）
- dynamic curriculum（需 training_progress 支持）
- 大权重 action_penalty
- 大权重 contact_reward

可以在 v1 使用的进阶骨架：
- potential_based_shaping：如果你能定义一个合理的势能函数 Φ(s)，用 γΦ(s')-Φ(s) 替代或辅助 progress_delta。
- gated_reward：如果你能把任务分成"接近"和"着陆"两个阶段，用不同组件组合。
- 这些不是必须的，但如果 progress_delta 单独无法提供足够引导，优先考虑它们。

避免重复：
- 不要同时大权重使用 distance_reward 和 progress_delta_reward。
- 如果同时使用，progress_delta_reward 是主信号，distance_reward 只能是小权重辅助 anchor。

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

为了兼容旧 wrapper，也可以把 components 写入 `info["reward_terms"]`，但仍然建议返回 `(float(total_reward), components)`。

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 至少包含所有被加到 total_reward 的组件，以及 total_reward。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- 使用了哪些奖励组件；
- 每个组件的角色；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些组件留到后续迭代；
- 训练后应该观察哪些 failure mode。