# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先用训练证据解释失败，再选择最小且可验证的干预。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率（旧日志中的`nonzero_rate`）。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- 不得仅因任务描述出现“跳跃、着陆、抓取”等语义，就断言对应状态量是缺失奖励职责。新增职责必须有轨迹行为、终止分布、组件激活或历史干预结果支持；证据不足时明确写“未知”，优先保持best主结构并提出最小可证伪修改。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

# 唯一决策流程

按顺序完成，不能因知识库或工具命中某个变换就跳级。

## 1. 行为与历史诊断

**第一步：阅读 user prompt 中的 #4 Subagent 行为诊断报告。**

**第二步：读 analyst 报告中的骨架演进表。** 表里并排了同一骨架各轮的 score、len、组件数值。看一眼就知道哪个组件变化引发了什么连锁反应。在此基础上回答两个问题：

1. **agent 学到了什么行为？**
2. **哪个组件最值得干预？** 不一定是变化最大的那个——如果 A 涨了 B 崩了，解释因果链。

随后确认当前失败是否来自该组件或某个缺失职责，一次只选择一个干预目标。current明显差于best时，以best代码为基础，但必须做一个新的、有证据的修改，不能原样复制best。

## 2. 信号完备性检查

检查当前奖励是否具有任务所需且可达的职责，而不是要求固定组件名称：

- 任务进展或可学习的过程引导；
- 必要的稳定、安全或动作约束；
- 当过程最优不等于任务完成时，能区分联合满足或完成状态的信号。

## 3. 选择干预层级

Subagent 诊断报告会明确告诉你哪个组件最需要干预。**它的推荐是你的起点。** 如果你不同意它的推荐，你必须在 response 里解释为什么不采纳。

在决定干预方向之前，先用 `episode_sum_mean ÷ len` 算出每个组件的 **per-step 量级**。直接对比两个组件的单步均值（如 `contact ≈ 4.15/step vs approach ≈ 0.017/step`），这比 share% 更直观地告诉你谁在真正驱动行为。

### Level 1：系数调整

**数据告诉你**：组件的职责和数学形态是对的，但量级失衡。Subagent 会指出”X 组件 per-step 均值太低了”或”Y 太强了”。

只调一个组件的系数。不改数学形态。

调哪个方向？看行为证据：如果强的组件在产生健康行为（score 涨、terminated 涨），保留它——提高弱组件。如果强的组件在驱动病态行为（score 跌、len 极长、exploit），削弱它。

### Level 2：数学形态修正

**数据告诉你**：有一个组件在教 agent 错误的行为——不是量级问题，是公式本身传达了错误的激励。训练反馈会显示对应的典型证据：

- active_rate 接近零 → 乘积塌缩或条件过严
- 持续状态值无界累积 → agent 赖着不走
- 全局惩罚全覆盖 → 压制探索

只改这一个组件的数学形态。参考以下变换方向（不穷举）：

| 证据 | 变换 |
|---|---|
| 事件几乎不触发 | 稀疏 → 连续 |
| 极端值支配 | 无界 → 有界 |
| 占据状态即获奖 | 状态值 → 改善量 |
| 乘积塌缩 | 乘积 → 加性/几何平均 |
| proxy与外部分数脱钩 | 代理 → 任务完成对齐 |

### Level 3：骨架重建

**数据告诉你**：所有历史迭代都没拿过好分数，当前骨架从根上错了。此时放弃在当前骨架上修补。

选 Level 3 之前，确认一个事实：**历史上有任何迭代拿到过可接受的分数吗？** 如果有——说明骨架核心成立，你需要的是 Level 1 或 Level 2 在 best 基础上定向修复，而不是 Level 3。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。需要指数形式时使用`2.718281828 ** exponent`，或改用`1/(1+k*x)`、`max(0,1-x/D)`等无需库的bounded表达式。
- 除Level 3重建外，每轮只修改一个目标组件，不顺带调整其他组件。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件，不放total_reward和中间调制器。

# 输出

先用以下固定字段各写一句，不复述输入表格：

1. `evidence`：支持判断的外部结果、组件证据和上一轮结果；
2. `behavior_diagnosis`：策略当前的失败行为；
3. `signal_completeness`：必要职责是否完备、可达；
4. `selected_level`：Level 1、Level 2或Level 3及触发条件；
5. `selected_intervention`：唯一目标组件及具体修改；
6. `falsifiable_hypothesis`：为什么该修改应改善策略；
7. `expected_next_round`：下一轮哪些指标应如何变化；
8. `main_risk`：最可能引入的新漏洞。

然后立即输出完整Python代码。预期必须能被下一轮反馈证伪。
```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: 103.628718
- gap_to_target: 96.371282
- target_achievement_ratio: 51.814%

# 2. 上一轮奖励函数代码（该轮得分: 103.628718）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward modulated by quality factors (WEIGHTED SUM)
    """
    
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    # === Component A: goal_approach_progress (main learning signal) ===
    # Use potential-based shaping: reward improvement in distance to target
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance  # positive when approaching
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    # === Component B: velocity_brake_near_target (safety constraint) ===
    # Quadratic velocity penalty, scaled by proximity to encourage slow approach
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)  # approaches 1 when close to target
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    # === Component C: orientation_stability (safety constraint) ===
    # Quadratic penalty on angle deviation and angular velocity
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    # === Component D: contact_landing_reward (dense contact guidance) ===
    # Continuous reward for contact, modulated by quality factors
    # Base: any contact (left or right)
    contact_active = max(next_left_contact, next_right_contact)
    
    # Quality modulator 1: velocity (lower is better, bounded to [0,1])
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    
    # Quality modulator 2: orientation (smaller angle is better, bounded to [0,1])
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    
    # Quality modulator 3: proximity to target (closer is better, bounded to [0,1])
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    # Combined: contact reward scaled by quality factors
    # Using WEIGHTED SUM instead of product to allow partial compensation
    quality_factor = 0.4 * velocity_quality + 0.3 * angle_quality + 0.3 * proximity_quality
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 984.85 | -82.86 | ✅ |
| 2 | 骨架变化: contact_landing_reward + goal_approach_progress +  | — | 721.10 | 142.32 | ✅ |
| 3 | 骨架变化: contact_landing_reward + goal_approach_progress +  | — | 930.15 | 36.97 | ❌ |
| 4 | 骨架变化: contact_landing_reward + goal_approach_progress +  | — | 736.95 | 103.63 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

# 诊断报告

## 1. 当前策略行为画像
agent 学会了接触着陆（13/20 终止），但回合过长（737 步）且得分低于历史最佳（103.63 vs 142.32）。contact_landing_reward 主导 90.7% 奖励份额，但仅 13.8% 激活率说明接触发生在末尾集中区间。

## 2. 组件健康度
- **contact_landing_reward**: 主导但错位（90.7% share，13.8% active，per-step 0.588）
- **goal_approach_progress**: 淹没（2.4% share，99.3% active，per-step 0.016）
- **velocity_brake_near_target**: 健康但被掩盖（-5.3% share，99.5% active，per-step -0.034）
- **orientation_stability**: 健康但被掩盖（-1.3% share，100% active，per-step -0.009）

## 3. 骨架演进表

```
| iter | score  | len    | term | contact_ep | contact_active | approach_ep | approach_active | velocity_ep | orient_ep |
|------|--------|--------|------|------------|----------------|-------------|-----------------|-------------|-----------|
| 2    | 142.32 | 721.10 | ?/20 | ~1517      | ?              | ~19.5       | ?               | ~-126       | ~-45      |
| 3    | 36.97  | 930.15 | 7/20 | 2031       | 52.3%          | 159         | 100%            | -221        | -61       |
| 4    | 103.63 | 736.95 | 13/20| 433        | 13.8%          | 12          | 99.3%           | -25         | -6        |
```

**演进解读**：iter3→4 将 contact 从加性权重和（weighted sum）改回几何平均（geometric mean^0.333），导致接触奖励暴跌 79%（2031→433），但终止率反升（7→13），说明质量门槛提高后反而促进了"正确接触"。然而 goal_approach_progress 同时崩塌 92%（159→12），distance shaping 被 contact 目标彻底压制，导致航行效率极低（len 仍高达 737）。

**交互效应**：contact 门槛收紧 + approach 信号失效 = agent 学会"磨蹭到目标附近才尝试接触"的低效策略，解释了高终止率但低分的矛盾现象。

## 4. 行为结论
**病态**。当前配置陷入"完成信号倒挂"陷阱：
- contact（稀疏完成信号）量级是 approach（稠密引导信号）的 **37 倍**（per-step 0.588 vs 0.016）
- approach 虽然 99.3% 全程激活，但单步贡献仅 0.016，在 contact 的 0.588 面前形同虚设

**最需干预组件**：**goal_approach_progress**。不是因为它"坏了"，而是它的 10.0 系数在 contact 5.0×quality 面前被结构性淹没。需要将 approach 系数提升至 50-100 量级，或将 contact 基础系数降至 1.0，才能让"航行→接近→着陆"的完整策略链条形成正确的梯度传递。

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander navigation and soft landing task.
    
    Task: Navigate from random perturbed initial position to target landing pad,
    achieve stable low-speed contact with upright orientation.
    
    Components:
    - goal_approach_progress: dense shaping via distance improvement delta
    - velocity_brake_near_target: quadratic velocity penalty scaled by proximity
    - orientation_stability: quadratic penalty on angle and angular velocity
    - contact_landing_reward: continuous contact reward modulated by quality factors
    """
    
    # Extract observations
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle, ang_vel = obs[4], obs[5]
    left_contact, right_contact = obs[6], obs[7]
    
    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_angle, next_ang_vel = next_obs[4], next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]
    
    components = {}
    
    # === Component A: goal_approach_progress (main learning signal) ===
    # Use potential-based shaping: reward improvement in distance to target
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (next_x_pos**2 + next_y_pos**2) ** 0.5
    distance_delta = current_distance - next_distance  # positive when approaching
    components["goal_approach_progress"] = 10.0 * distance_delta
    
    # === Component B: velocity_brake_near_target (safety constraint) ===
    # Quadratic velocity penalty, scaled by proximity to encourage slow approach
    velocity_magnitude = (next_x_vel**2 + next_y_vel**2) ** 0.5
    proximity_factor = 1.0 / (1.0 + next_distance)  # approaches 1 when close to target
    components["velocity_brake_near_target"] = -2.0 * proximity_factor * velocity_magnitude**2
    
    # === Component C: orientation_stability (safety constraint) ===
    # Quadratic penalty on angle deviation and angular velocity
    angle_penalty = next_angle**2
    ang_vel_penalty = next_ang_vel**2
    components["orientation_stability"] = -1.0 * (angle_penalty + 0.5 * ang_vel_penalty)
    
    # === Component D: contact_landing_reward (dense contact guidance) ===
    # Continuous reward for contact, modulated by landing quality factors
    # Base: any contact (left or right)
    contact_active = max(next_left_contact, next_right_contact)
    
    # Quality modulator 1: velocity (lower is better, bounded to [0,1])
    velocity_quality = max(0.0, 1.0 - velocity_magnitude / 1.0)
    
    # Quality modulator 2: orientation (smaller angle is better, bounded to [0,1])
    angle_quality = max(0.0, 1.0 - abs(next_angle) / 0.5)
    
    # Quality modulator 3: proximity to target (closer is better, bounded to [0,1])
    proximity_quality = max(0.0, 1.0 - next_distance / 1.0)
    
    # Combined: contact reward scaled by quality factors
    # Using product to require joint satisfaction, but normalized to avoid collapse
    quality_factor = (velocity_quality * angle_quality * proximity_quality) ** 0.333
    components["contact_landing_reward"] = 5.0 * contact_active * quality_factor
    
    total_reward = sum(components.values())
    
    return float(total_reward), components
```

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| contact_landing_reward | 2.183 | 2.131 | -0.052 | 52.3% | 50.5% | -1.8pp |
| goal_approach_progress | 0.171 | 0.032 | -0.139 | 100.0% | 100.0% | +0.0pp |
| orientation_stability | -0.066 | -0.072 | -0.005 | 100.0% | 100.0% | -0.0pp |
| velocity_brake_near_target | -0.238 | -0.223 | +0.015 | 100.0% | 100.0% | +0.0pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | 36.97 | 103.63 | +66.66 |
| 平均回合长度 | 930.1 | 737.0 | -193.2 |
| 终止回合数 | 7/20 | 13/20 | +6 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=103.628718, len=736.950000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[-48.804643, 219.162094]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_landing_reward | 433.442598 | 90.7% | 90.7% | 13.8% |
| velocity_brake_near_target | -25.202204 | -5.3% | 5.3% | 99.5% |
| goal_approach_progress | 11.665704 | 2.4% | 2.7% | 99.3% |
| orientation_stability | -6.340977 | -1.3% | 1.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主任务是控制一个二维刚体（类飞行器/着陆器）从顶部中央区域出发，在受到随机初始扰动力的情况下，尽快移动到画面中央的目标着陆区，并以低速、稳定姿态完成安全接触（“软着陆”）。次要目标是尽量减少引擎使用次数/推力消耗，追求燃料效率。不应将“存活/不坠毁”本身当作主目标——存活只是安全着陆的必要条件，而不是与到达目标并列的独立主目标；同时也不应把“悬停不动”误当作成功，因为任务要求最终稳定接触目标垫而非无限期悬空。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 未在源码中显式声明，推测为 float32（未确认，需以实际环境返回类型为准）
- obs[0]: x_position，相对目标垫的水平坐标，reward_usable: true
- obs[1]: y_position，相对目标垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体姿态角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左侧接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右侧接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不启动任何引擎
- action 1: left_orientation_engine，启动左侧姿态引擎（产生反向力矩/侧向修正）
- action 2: main_engine，启动主引擎（主要提供向上/减速推力）
- action 3: right_orientation_engine，启动右侧姿态引擎（与 action 1 相对的修正方向）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功标志，但 `body_not_awake_or_settled`（机体进入静止/休眠状态）在其发生时若伴随 obs[6]、obs[7] 均为 1 且位置接近目标（obs[0]、obs[1] 接近 0）、速度接近 0，可视为“成功着陆”的间接证据；仅当满足这些位置+速度+接触组合条件时才可推断为成功，不能仅凭该终止标志本身判定成功。
- failure-like termination: `crash_or_body_contact`（发生坠毁/异常碰撞）通常对应失败；若发生该终止但同时接触点状态异常（例如姿态角过大、速度过大）则更确定为失败。
- ambiguous termination: `horizontal_position_outside_viewport`（水平方向越界）本身既可能是失败（飞出边界），但也可能与正常物理仿真边界裁剪有关，需结合速度/位置综合判断，不能单独作为成功或失败的确定信号。
- truncation: 源码中固定返回 `False`，即当前 masked step 未展示基于时间步数的截断机制；若存在最大步数限制，需以外部 wrapper 确认，此处不能假设。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（`info={}`，源码未声明任何字段）
- forbidden_or_uncertain_info_fields: `info["success"]`、`info["failure"]`、`info["termination_reason"]` 及任何未在 step 源码中声明的字段均不可假设存在，禁止使用。

## 7. 可用于奖励函数的信号
- position: obs[0]（相对目标水平位置）、obs[1]（相对目标垫垂直位置）
- velocity: obs[2]（水平速度）、obs[3]（垂直速度）
- orientation: obs[4]（姿态角）、obs[5]（角速度）
- contact: obs[6]（左接触）、obs[7]（右接触）
- action/engine: action 本身（0=无动作，1/3=姿态引擎，2=主引擎），可用于估算引擎使用频率/能耗代理
- other: terminated 标志（由环境返回，可用于判断阶段结束，但需结合 obs 状态推断具体原因，不能直接读取“原因文本”）

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | goal_approach_progress + orientation_stability + terminal_soft_landing_bonus + velocity_brake_near_target | -82.86 | -82.86 | 0.00 | 984.85 | goal_approach_progress=0.031 orientation_stability=-0.071 terminal_soft_landing_bonus=14.147 velocity_brake_near_target=-0.200 | new_best |
| 2 | contact_landing_reward + goal_approach_progress + orientation_stability + velocity_brake_near_target | 142.32 | 142.32 | 0.00 | 721.10 | contact_landing_reward=2.104 goal_approach_progress=0.027 orientation_stability=-0.062 velocity_brake_near_target=-0.175 | new_best |
| 3 | contact_landing_reward + goal_approach_progress + orientation_stability + velocity_brake_near_target | 36.97 | 142.32 | -105.35 | 930.15 | contact_landing_reward=2.183 goal_approach_progress=0.171 orientation_stability=-0.066 velocity_brake_near_target=-0.238 | no_meaningful_improvement |
| 4 | contact_landing_reward + goal_approach_progress + orientation_stability + velocity_brake_near_target | 103.63 | 142.32 | -38.69 | 736.95 | contact_landing_reward=2.131 goal_approach_progress=0.032 orientation_stability=-0.072 velocity_brake_near_target=-0.223 | no_meaningful_improvement |

```
