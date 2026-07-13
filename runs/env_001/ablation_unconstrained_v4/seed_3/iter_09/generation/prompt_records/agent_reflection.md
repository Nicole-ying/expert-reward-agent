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
# ⚠️ 上一版代码验证失败
错误信息：Reward v9 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs\env_001\ablation_unconstrained_v4\seed_3\iter_09\generation\validations\reward_v9.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
首先进行分析，然后检索知识辅助设计。

## 分析

### 1. agent发生了什么？
上一轮代码引入的三个组件完全失效。`approach_reward`采用 `max(0, dist_obs - dist_next)` 只有在朝向目标移动时才给予奖励，但仅有5.9%的步激活，说明agent长时间在远离目标或无效徘徊。`soft_landing_penalty`的门控距离阈值为1.0，而agent从未接近至此范围（0%激活），因此完全没起作用。`upright_penalty`每步都在惩罚角度平方，尽管单步值小（约-0.0047），但episode长达587步，累计成巨大负值（-2.76每回合），主导了总得分。整体行为是agent在远处无目标漫游，概率性出现极少量正progress，但被持续的姿势惩罚淹没，最终因飞出视口或碰撞而大量终止。得分从best的-10急剧恶化到-5510。

### 2. 哪个组件最值得干预？
必须整体重构奖励函数。当前 `approach_reward` 的稀疏性是核心问题，应改为每步提供梯度的稠密距离信号。姿势惩罚需大幅降低权重或改为接近地面时激活。速度控制也需要在接近目标时确保激活。当前结构已无保留价值，应从best设计思想出发重新设计。

### 3. 上一轮改了什么？
在iter 7（`engine_penalty + landing_reward + lateral_penalty + progress_reward`，得分-122）基础上彻底替换为 `approach_reward + soft_landing_penalty + upright_penalty`，结果灾难性退化。因此，必须回到接近best（-10.16）的框架：接触奖励 + 门控目标奖励，并用稠密距离信号和改进的惩罚来修复问题。

接下来，检索外部知识辅助确定良好实践。

<search_reward_design_knowledge>
2D lunar lander reward shaping: dense distance signal, contact reward, soft velocity gate, boundary penalty, engine penalty scheduling
</search_reward_design_knowledge>

# Search objective
- target_score: 200.000000
- current_score: -5510.929150
- gap_to_target: 5710.929150
- target_achievement_ratio: -2755.465%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -5510.929150）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]

    # Distance to target pad (Euclidean)
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5

    # --- Component A: approach progress (improvement_delta, positive only) ---
    # Rewards any reduction in distance, encourages reaching the pad.
    progress = max(0.0, dist_obs - dist_next)
    w_progress = 2.0
    approach_reward = w_progress * progress

    # --- Component B: soft landing speed penalty (quadratic with gate) ---
    # Penalises high velocity only when already close to the target,
    # promoting deceleration for a stable touchdown.
    landing_dist_threshold = 1.0
    gate_landing = max(0.0, 1.0 - dist_obs / landing_dist_threshold)  # active when dist_obs < threshold
    w_speed = 1.0
    speed_sq = vx_next**2 + vy_next**2
    soft_landing_penalty = -w_speed * gate_landing * speed_sq

    # --- Component C: stable upright penalty (quadratic) ---
    # Discourages large body angles that could cause flip or single‑foot contact.
    w_angle = 0.1
    upright_penalty = -w_angle * (angle_next**2)

    # Total reward
    total_reward = approach_reward + soft_landing_penalty + upright_penalty

    components = {
        'approach_reward': approach_reward,
        'soft_landing_penalty': soft_landing_penalty,
        'upright_penalty': upright_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-5510.929150, len=586.850000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-10326.028906, -707.312998]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| upright_penalty | -2.760102 | -60.0% | 60.0% | 100.0% |
| approach_reward | 1.839699 | 40.0% | 40.0% | 5.9% |
| soft_landing_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个二维飞行器的轨迹优化问题。主体从视口上方中央附近以随机初始受力出发，需要在最短时间内到达并稳定在中央目标着陆垫上，同时尽量少用引擎推力。智能体必须学会向目标靠近，减小水平和垂直速度，保持稳定姿态，并实现安全接触（双支撑腿同时触垫）。次要目标是节约燃料（即减少引擎使用次数），但绝不能以牺牲成功着陆和安全为代价。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: 默认 float32（连续值，接触标志为 0.0 或 1.0）  
- 各维度含义：
  - obs[0]：`x_position`，相对目标垫的水平坐标（负/正代表左右），可用于奖励计算 `reward_usable: true`
  - obs[1]：`y_position`，相对垫高度的垂直坐标，可用于奖励计算 `reward_usable: true`
  - obs[2]：`x_velocity`，水平线速度，可用于奖励计算 `reward_usable: true`
  - obs[3]：`y_velocity`，垂直线速度，可用于奖励计算 `reward_usable: true`
  - obs[4]：`body_angle`，机体偏转角（以弧度计），可用于奖励计算 `reward_usable: true`
  - obs[5]：`angular_velocity`，角速度，可用于奖励计算 `reward_usable: true`
  - obs[6]：`left_support_contact`，左支撑腿接触标志（0.0/1.0），可用于奖励计算 `reward_usable: true`
  - obs[7]：`right_support_contact`，右支撑腿接触标志（0.0/1.0），可用于奖励计算 `reward_usable: true`

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - 0：`no_engine`，不做任何推力  
  - 1：`left_orientation_engine`，点燃左姿态引擎（通常产生顺时针力矩和微小侧向力）  
  - 2：`main_engine`，点燃主引擎（通常产生向上的主要推力）  
  - 3：`right_orientation_engine`，点燃右姿态引擎（通常产生逆时针力矩和微小侧向力）  

说明：动作是离散的推进器选择，每个时间步只能选择一种操作。因此奖励中无法直接获得连续推力值，但可基于动作类型判断是否使用了引擎，从而惩罚燃料消耗。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `termination_conditions` 原文，共三种终止原因：
1. **crash_or_body_contact（碰撞或身体接触）**  
   - 很可能是与地面或障碍物发生不当碰撞，通常代表失败。
2. **horizontal_position_outside_viewport（水平位置超出视口）**  
   - 机体横向飞出边界，代表失败。
3. **body_not_awake_or_settled（机体未醒或已稳定）**  
   - 语义存疑：可能表示机体已进入休眠状态（着陆后稳定静止）或未能稳定（翻倒后停止）。由于原文将它与前两类并列且未注明成功/失败，该终止条件本身是 **模糊的**。在典型同类任务中，机体成功着陆后会进入“睡眠”（awake=False），因而该条件可能是 **成功终止**。但在正式设计 reward 时，不能直接依赖该条件为成功标志，必须结合接触、位置、速度等观测自行判断。

### 5.2 步骤返回信息
`step()` 返回 `(state, masked_reward, terminated, False, {})`，其中 `info` 为空字典。因此：
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无
- **forbidden_or_uncertain_info_fields**: 所有 info 字段（因 info 为空，无可用信号）

## 7. 可用于奖励函数的信号
基于 `obs`、`action` 和 `next_obs`，可提取以下信号：
- **位置信号**：`obs[0]`、`obs[1]`（或 `next_obs[0]`、`next_obs[1]`），表示相对目标垫的水平和垂直偏差。
- **速度信号**：`obs[2]`、`obs[3]`，水平与垂直线速度。
- **姿态信号**：`obs[4]`（偏转角）、`obs[5]`（角速度）。
- **接触信号**：`obs[6]`、`obs[7]`，左右支撑腿是否触地。
- **动作/引擎信号**：`action` 值，用于判断是否使用了引擎（0为无推力，1、2、3均消耗燃料）。
- **附加信号**：还可以计算 `next_obs` 与目标的距离变化、速度变化等差分信号。

所有这些信号均源自观测，可直接用于塑造奖励，无需依赖 info。
```
