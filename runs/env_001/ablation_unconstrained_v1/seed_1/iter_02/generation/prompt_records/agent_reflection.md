# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。用训练证据分析失败原因，提出修改方案并实施。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- 不得仅因任务描述出现"跳跃、着陆、抓取"等语义，就断言对应状态量是缺失奖励职责。新增职责必须有轨迹行为、终止分布、组件激活或历史干预结果支持；证据不足时明确写"未知"，优先保持best主结构并提出最小可证伪修改。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

# 分析要求

拿到训练反馈后，回答：
1. 这个agent发生了什么？用行为证据说明。
2. 哪个组件最值得干预？结合数学形态和统计数据判断。
3. 我之前改了什么？从Agent Memory检查上一轮动作与实际效果。

随后根据分析结果修改奖励函数。你可以调整系数、改变数学形态、增加或删除组件、或重构整体结构——修改范围由证据决定。current明显差于best时，以best代码为基础，但必须做一个新的、有证据的修改，不能原样复制best。

# 工具

以下工具可在诊断不确定时调用：

- `search_reward_design_knowledge(query)`：检索相似失败模式和经验修复。
- `get_skeleton_detail(skeleton_name)`：查看数学形态、原理和陷阱。
- `get_reward_transformation(query)`：查看结构变换的原理、风险和验证目标。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。需要指数形式时使用`2.718281828 ** exponent`，或改用`1/(1+k*x)`、`max(0,1-x/D)`等无需库的bounded表达式。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件，不放total_reward和中间调制器。

# 输出

先简要说明分析和修改理由，然后立即输出完整Python代码。

```

## User Prompt

```markdown
# Search objective
- target_score: 200.000000
- current_score: -108.688572
- gap_to_target: 308.688572
- target_achievement_ratio: -54.344%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -108.688572）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 主学习信号：接近目标的进度 ---
    # 使用当前和下一步到目标 (0,0) 的欧氏距离差
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_obs - dist_next  # 靠近为正，远离为负

    # --- 稳定和平滑约束 ---
    # 惩罚过大的速度、偏角和角速度，促使平稳着陆
    vel_penalty = 0.05 * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = 0.1 * abs(next_obs[4])
    angvel_penalty = 0.01 * abs(next_obs[5])
    stability_penalty = -(vel_penalty + angle_penalty + angvel_penalty)

    # --- 任务完成近似信号：软着陆奖励 (proxy) ---
    # 当双腿同时接触平台、位置靠近中心、速度足够小、姿态接近水平时给予一次性奖励
    landing_bonus = 0.0
    both_legs_down = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    near_center = (abs(next_obs[0]) < 0.2) and (abs(next_obs[1]) < 0.2)
    slow_enough = (abs(next_obs[2]) < 0.2) and (abs(next_obs[3]) < 0.2)
    upright = abs(next_obs[4]) < 0.1

    if both_legs_down and near_center and slow_enough and upright:
        landing_bonus = 10.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus

    # 组件字典（只包含被加到 total_reward 的项）
    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_bonus': landing_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-108.688572, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.714740, -90.133068]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -4.707838 | -53.1% | 53.1% | 100.0% |
| landing_bonus | 3.000000 | 33.8% | 33.8% | 0.4% |
| progress_reward | 1.119618 | 12.6% | 13.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 类飞行器着陆/姿态控制任务。  
主体从屏幕上方中部某位置出发，受到初始随机作用力。  
目标是**尽快到达并稳定降落在中央目标平台**上，同时**尽可能少使用引擎推力**。  
智能体需要学习：  
- 向目标平台靠近  
- 降低速度  
- 保持稳定姿态  
- 与平台安全接触（两条支撑腿同时着地）  

附属优化包括省燃料、时间短、姿态平稳，但不额外构成独立目标。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float32 或 float64（具体未知，但连续数值）
- obs[0]: x_position —— 水平坐标，相对于目标平台（零点为目标中心）
- obs[1]: y_position —— 垂直坐标，相对于平台高度（零点为平台表面高度）
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体朝向角度（可能用弧度）
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左支撑腿接触标志，1.0 表示接触，0.0 表示不接触
- obs[7]: right_support_contact —— 右支撑腿接触标志，1.0 表示接触，0.0 表示不接触

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 4
- action 0: no_engine —— 不启动任何引擎，无推力/力矩
- action 1: left_orientation_engine —— 启动左姿态引擎（产生顺时针或逆时针力矩，具体方向需经验判断）
- action 2: main_engine —— 启动主引擎（产生向上的主要推力）
- action 3: right_orientation_engine —— 启动右姿态引擎（产生与动作1相反的力矩）

动作空间是针对飞行器的简单推力与姿态控制，无油门大小，每个动作固定输出一个脉冲。

## 5. step 与终止条件分析
### 5.1 终止模式
环境在满足以下任一条件时终止（terminated = True）：
- **crash_or_body_contact** —— 主体部分（非支撑腿）发生碰撞或接触（可能摔到地面上、撞到障碍等）
- **horizontal_position_outside_viewport** —— 水平坐标超出视口范围（远离目标平台过远）
- **body_not_awake_or_settled** —— 身体不再运动或已稳定（包括成功着陆后稳定静止）

> 可见，最后一种可能既包含成功着陆稳定，也可能包含其他静止的状态（如卡住、惯性停止等）。需要结合上下文判断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false   （info 中没有 success 字段）
- explicit_failure_flag_available: false   （info 中没有 failure 字段）
- allowed_info_fields: （info 为空字典，无任何字段可用）
- forbidden_or_uncertain_info_fields: info 中所有字段不可用（因为 info 为空）；不可使用 terminated 标志

**注意**：终止条件（crash/出界/稳定）的真实语义不能直接从 `info` 获得，需要从 `next_obs` 的状态推理。

## 7. 可用于奖励函数的信号
- **位置（接近目标）**：`next_obs[0]`（相对目标平台的水平距离），`next_obs[1]`（高度偏移），可鼓励 x,y 趋向 0
- **速度（着陆柔和性）**：`next_obs[2]`, `next_obs[3]`，可惩罚过大的速度，尤其垂向着陆速度
- **姿态**：`next_obs[4]`（身体角度），鼓励接近 0（水平姿态）；`next_obs[5]`（角速度），惩罚剧烈旋转
- **接触**：`next_obs[6]` 和 `next_obs[7]`（左右支撑腿接触），两条腿同时接触标志稳定的成功着陆
- **动作/引擎使用**：从 `action` 可以判断是否使用了引擎（0为无引擎，其他为有引擎），可惩罚燃料消耗；也可以结合燃料效率同时使用 `obs` 与 `next_obs` 的变化判断推力作用

# Compact expert route context
- selected_route_id: navigation_goal_reaching
- recommended_design_roles: terminal_success_reward (终点成功奖励), terminal_failure_penalty (失败惩罚), time_penalty (每步时间惩罚), distance_reward (距离型密集奖励), progress_delta_reward (增量进步奖励), potential_based_shaping (势能塑形奖励), gated_reward (门控/层级奖励)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.
```
