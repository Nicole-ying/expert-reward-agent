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
- current_score: -64.501673
- gap_to_target: 264.501673
- target_achievement_ratio: -32.251%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -64.501673）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Next state
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 1. Bounded proximity: always provides gradient toward target
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    r_dist = 1.0 / (1.0 + distance)
    
    # 2. Stability penalty: discourage violent motion
    r_stability = -(
        0.01 * (abs(x_vel) + abs(y_vel)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )
    
    # 3. Continuous soft-landing proxy with widened activation ranges
    #    to provide gradient earlier and guide deceleration.
    proximity = max(0.0, 1.0 - distance / 1.0)            # activates when dist < 1.0
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.8)  # activates when |v| < 0.8
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)    # unchanged
    contact   = (left_contact + right_contact) / 2.0
    r_landing = 8.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)
    
    total_reward = r_dist + r_stability + r_landing
    
    components = {
        "distance_reward": r_dist,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }
    
    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Next state
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 1. Bounded proximity: always provides gradient toward target
    #    (0, 1], peaks at 1 when exactly on target
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    r_dist = 1.0 / (1.0 + distance)
    
    # 2. Stability penalty: discourage violent motion, promote gentle flight
    r_stability = -(
        0.01 * (abs(x_vel) + abs(y_vel)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )
    
    # 3. Continuous soft-landing proxy: product of four bounded factors [0,1]
    #    Each factor opens gradually — provides gradient before perfect conditions.
    proximity = max(0.0, 1.0 - distance / 0.5)          # activates when dist < 0.5
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.4)  # activates when |v| < 0.4
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)  # activates when |angle| < 0.3
    contact   = (left_contact + right_contact) / 2.0    # 0, 0.5, or 1.0
    # Floor of 0.1 ensures weak signal before any leg contact;
    # one leg = 0.55, both legs = 1.0 — gradient toward full touchdown.
    r_landing = 8.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)
    
    total_reward = r_dist + r_stability + r_landing
    
    components = {
        "distance_reward": r_dist,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }
    
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-64.501673, len=951.000000, terminated=6/20, truncated=14/20, reward_errors=0
score_range=[-146.576931, -4.212043]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | 649.250327 | 66.3% | 66.3% | 100.0% |
| landing_proxy | 323.740209 | 33.1% | 33.1% | 90.0% |
| stability_penalty | -5.867505 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个2D飞行器轨迹优化任务。  
主体从视口顶部中央附近出发，并受到随机初始力。  
目标是**尽可能快地飞到并稳定降落在中央目标垫上**，同时**消耗尽可能少的引擎推力**。  
智能体需要学会：接近目标、减速、保持姿态稳定、实现安全触地。

## 3. 观察空间 observation_space
- type: Box（连续）
- shape: [8]
- dtype: float64 （默认，部分字段为0/1浮点）
- obs[0]: x_position —— 相对目标垫的水平坐标
- obs[1]: y_position —— 相对目标垫高度的垂直坐标
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体朝向角
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左支撑腿接触标志（1.0 接触; 0.0 未接触）
- obs[7]: right_support_contact —— 右支撑腿接触标志（1.0 接触; 0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不点燃任何引擎
- action 1: left_orientation_engine —— 点燃左侧姿态引擎
- action 2: main_engine —— 点燃主引擎（产生推力）
- action 3: right_orientation_engine —— 点燃右侧姿态引擎（与左相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled —— 机体不再运动且稳定，可能对应成功着陆
- failure-like termination: crash_or_body_contact —— 机体重重撞击或非支撑部位触地（并非正常支撑腿触地）
- failure-like termination: horizontal_position_outside_viewport —— 飞出水平边界
- ambiguous termination: 无
- truncation: 源码中未展示步数限制，但通常环境存在episode截断，这里无信息，不依赖

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典）
- explicit_failure_flag_available: false
- allowed_info_fields: （无）
- forbidden_or_uncertain_info_fields: info.* （任何 info 字段都未声明，不可安全使用）

## 7. 可用于奖励函数的信号
- **位置**：obs[0] x_position, obs[1] y_position（相对目标垫的水平和垂直距离）
- **速度**：obs[2] x_velocity, obs[3] y_velocity（下降速度等）
- **姿态与角速度**：obs[4] body_angle, obs[5] angular_velocity
- **接触**：obs[6] left_support_contact, obs[7] right_support_contact（是否两条支撑腿着地）
- **动作/引擎**：当前动作类型（是否使用主引擎或姿态引擎）

# Compact expert route context
- selected_route_id: navigation_goal_reaching
- recommended_design_roles: terminal_success_reward (终点成功奖励), terminal_failure_penalty (失败惩罚), time_penalty (每步时间惩罚), distance_reward (距离型密集奖励), progress_delta_reward (增量进步奖励), potential_based_shaping (势能塑形奖励), gated_reward (门控/层级奖励)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.
```
