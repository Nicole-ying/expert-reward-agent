# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs\env_001\ablation_unconstrained_v4\seed_3\iter_03\generation\validations\reward_v3.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
<search_reward_design_knowledge>
# 问题诊断与修改建议

## 当前失败模式分析

根据训练反馈数据，agent 表现出以下行为特征：

| 行为表现 | 数据支持 |
|---|---|
| 从未双脚同时触地 | contact_bonus = 0%, active_rate = 0% |
| 持续使用引擎 | engine_penalty active_rate = 98.6% |
| 未发生碰撞或出界（20个episode均达1000步截断） | terminated=0/20, truncated=20/20 |
| 高度和距离奖励持续获得但在正区域徘徊 | height_reward=283.2, gated_goal=185.9，但最终得分仍为负（-63.8），说明奖励流实际上被截断或被环境惩罚抵消 |

## 根因推演

1. **着陆激励缺失**：当前 `contact_bonus` 仅在左右腿**同时**触地的瞬间给出 2 点奖励。但在真实轨迹中，机体可能先单腿着地、角度倾斜，导致触发器 `left_contact * right_contact` 恒为零，agent 永远学不到“先站稳再放平”的过渡行为。
2. **奖励曲线平坦化**：`height_reward` 使用 `1/(1+3*|y|)`，当机体在接近着陆区（y≈0.2）时，奖励仅为 0.625，与高空时差距不大，无法形成强梯度，导致 agent 悬停在没有明显惩罚的区域。
3. **episode 始终截断**：所有 episode 在 1000 步被截断，agent 从未达到终止条件（无论是成功着陆 or 失败），说明**奖励函数未鼓励终止**。环境本身没有成功信号，agent 学不到“早着陆=更好”的关联，反而可能将截断当作正常结束。
4. **引擎惩罚太小**：每个引擎动作惩罚仅有 0.05，相对于持续获得的正向奖励（height+gated ≈ 0.4/步），agent 宁愿多开引擎以维持位置，不愿冒险尝试着陆。

## 干预点优先级

| 干预点 | 优先级 | 理由 |
|---|---|---|
| 连续化接触奖励，引导渐进着陆 | 高 | 当前二进制式奖励 0/2 永远不会激活，必须提供从接近地面到双腿触地的稠密信号 |
| 增加着陆后的大额奖励 | 高 | 截断式 episode 导致 agent 不知道“成功”会带来大回报，需要在着陆后给予巨大的正向脉冲 |
| 调整高度奖励，强化近地层梯度 | 中 | 当前平坦曲线不利于区分“悬停区”和“着陆区”，可用 hinge 或线性化 |
| 适当放宽角度约束的杀伤力 | 低 | `angle_gate` 目前为 `1/(1+0.5|angle|)`，可能过于严格，但尚未明显致病 |

## 建议的奖励函数结构（修改版）

建议保留 `gated_goal` 作为核心前进信号，但做出以下修改：

1. **新增连续着陆激励（替代原来的 contact_bonus）**  
   - 使用 `min(left_contact, right_contact) * landing_quality`  
   - `landing_quality` 可基于低垂直速度、低角速度、小倾斜角的乘积（几何平均），鼓励稳定、轻柔着陆。
2. **加入“着陆成功”大额奖励**  
   - 条件：`left_contact AND right_contact AND vertical_velocity near zero AND angle near zero AND body_sleep`（body_sleep 可通过 `obs` 中某个标志判断，但这里 obs 空间[8] 中貌似没有 awaker 标志。可使用 `body_not_awake_or_settled` 终止条件，但我们的函数不能直接访问终止标志，不过可以猜测着陆后持续接触且速度为零持续数步，用以给予奖励）。  
   - 由于无法直接获得 awake 标志，可改用：连续多步双脚接触且速度、角度极小，给予递增奖励，最终在 episode 终止时由环境自然结束。
3. **增强引擎惩罚至 0.1/步**  
   - 进一步降低无意义悬停的边际收益。
4. **近地距离奖励使用 hinge**  
   - `max(0, 1 - 5 * |y|)` 代替 `1/(1+3|y|)`，使得 y=0 附近奖励上升迅速，区分度高。

## 可调参数与验证目标

- 接触质量权重：建议 1.0，观察 active_rate 应 > 5%。
- 近地层 hinge 阈值：5，使得 y=0.2 时奖励 0，y=0.0 奖励 1。
- 引擎惩罚：0.1，预期 engine_penalty sum 在 1000 步中可能仍为负，但 agent 将学习减少引擎使用。
- 验证目标：至少出现 1 个 episode 终止（不是截断），且 contact_bonus 开始激活。

</search_reward_design_knowledge>好的，search给出了详细的根因和修改建议，与我分析一致。它指出需要连续化接触奖励，加大着陆后奖励，强化近地层梯度，增强引擎惩罚。但是，我们不能使用环境内部的标志比如body_sleep，因为我们只能从obs获取，而obs没有awake标志。环境事实中终止条件有一个“body_not_awake_or_settled”，可能意味着着陆后awake变false导致terminated，但我们无法直接读取。但我们可以在agent双脚着地且速度接近零时给予持续奖励，促使它稳定，然后环境自然会终止。

我们要确保实现一个有效的奖励结构。应当参照knowledge的建议，结合历史best（contact_reward + gated_goal_reward），构造新函数。我们可以使用新骨架如continuous_contact + soft_landing_gate等。

同时，当前height_reward可能是问题，可以用hinge替代，使近地梯度更强。但是，也要小心不要过度惩罚高度，因为agent需要从较高处下降，hinge会奖零奖，可能导致下降无引导。可以使用分段奖励：距离奖励用hinge，但门控目标奖励已经能引导向目标。对于高度，可以用hinge但阈值设在相对你期望的范围，如max(0, 1 - 3*|y|)？但要确保在y较大时奖励为0，但总比有负奖励好。但可能会失去下降梯度。可以用smooth但更强的梯度，比如 `height_reward = 1/(0.1 + abs(y))`，但绝对值靠近0时急剧增大，但避免除零。但是abs(y)可能为0，所以0.1+abs(y)没问题。这会使得近地时奖励很高，快速下降。但这仍是一个无限奖励，可能导致奖励爆炸。最好用bounded形式如 `exp(-α*abs(y))` 或 `1/(1+α*abs(y))` 但α更大如10。以前的3太弱。可以改成α=10。

不过，按照knowledge，hinge形式 `max(0, 1 - α*|y|)` 有边界，且在近地处梯度固定。这个好，因为奖励在y=0时为1，在y>1/α时为0，提供了清晰的着陆带。选择α=5，那么着陆带宽度0.2单位，但y通常可能较大？我们不知道单位。但之前的平均height_reward=0.283对应y≈0.84（根据1/(1+3y)=0.283 => y=(1/0.283 -1)/3 ≈0.84）。所以agent常处在y≈0.84高度。如果使用hinge max(0, 1-5|y|)，当y=0.2时奖励0，所以高度大于0.2都得不到奖励。这会使梯度消失，agent更难下降。所以hinge可能导致靠近但不进入着陆带。应该用更宽的阈值，如α=1，即max(0,1-|y|)，着陆带半径1，y=0.84时奖励0.16，这样梯度还存在。或者用α=2，着陆带0.5，y=0.84奖励0。这不好，会导致奖励突然消失，agent不敢下降。也许保留平滑函数但系数增加：`height_reward = 1/(1 + 8*abs(y))` 或其他。我觉得可以保留单调递减的bounded函数，但要确保近地奖励远超悬停高度奖励，引导下降。例如：`height_reward = 1/(0.05 + abs(y))`，但无界。也可以用 `1/(0.1 + y^2)`，但更平滑。最好用`1/(1+10*abs(y))`这样在0为1，在0.84为1/(1+8.4)=0.106，而之前是0.283，反而降低了奖励。其实agent可能因为height_reward太低而放弃高度。我们要增加着陆激励，可以改用更高的系数，让近地层奖励更高，同时配上接触奖励。或许不用height_reward，直接用contact的连续版本，以及与速度相关。

另一个关键：引擎惩罚太小，需要加大到-0.1或-0.2，并可能根据速度增加惩罚，但先加大常数。还要增加着陆成功的大bonus，但着陆成功需要定义。可以在双脚接触时给予即时bonus，但为了避免稀疏，结合连续接触质量。knowledge建议使用`min(left_contact, right_contact) * landing_quality`，这样当单腿接触时有部分奖励，引导另一腿也接触。landing_quality可基于低垂直速度、低角速度、小角度等。这样形成稠密奖励。

同时，门控目标奖励可能被gate限制。gate中speed_gate和angle_gate可能过度压制，因为agent若高速下降，gate很小，导致奖励变小。但为了着陆，需要控制速度，所以gate是有益的。不过可以允许更大的速度范围，比如将speed_gate改为`1/(1+0.5*speed_sq)`，更宽容。

结合这些，我们设计新奖励函数：

结构：
- 目标接近奖励 (goal_approach): 基于距离的reward，不做gate，或者轻gate。可以直接给`1/(1+5*dist)`作为基础前进信号，不必gate。因为gate可能导致agent不敢大动作。或者保留gate但放宽。
- 着陆质量奖励 (landing_incentive): 连续化的着陆激励，替代height_reward和contact_bonus。可以使用`min(left_contact, right_contact) * quality`，其中quality由垂直速度、水平速度、角度等因子乘积或几何平均。因为agent下降时可能速度快，需要鼓励低速触地。此外，当没有接触时，也可以给一个接近地面的奖励（height_reward）来引导下降，否则没有接触奖励时，agent会迷失。所以保留独立的height_reward但调整系数。
- 引擎惩罚加大。
- 着陆成功大bonus：当双脚接触且速度、角度满足阈值时，给予大奖励（如50点），但要让agent能够重复获得？但着陆后episode可能很快结束，所以bonus只需在着陆瞬间给一次。可以在第一次满足着陆条件时给，但每步判断可能重复。简单点：当left_contact和right_contact都为1，且vertical_velocity的绝对值小，且angle小，给予额外奖励。但是，如果episode继续（环境可能不立刻终止），会持续给bonus？可以设计成只要满足条件就给一个固定bonus，但最好只用一次。可以用条件`if left_contact and right_contact and abs(vy) < 0.1 and abs(angle) < 0.1: landing_bonus = 50.0` 在每一步？如果持续满足，总奖励会很大，但着陆后应该很快terminated，所以问题不大。但如果环境不终止，会无限加。我们需要环境信息，无法判断是否终止，但可以猜测环境会在稳定后终止。可以加个flag，但函数不能有状态。我们可以认为，双脚稳定接触后，episode很快会结束，所以短期内多次bonus也可接受，或者只给一次。但训练中可能导致agent反复弹跳以获取多次bonus。更好的方式是只在第一次接触时给，但函数无状态。因此，我们给一个适度的持续奖励，例如每步满足条件给2，类似contact_bonus但条件放宽。这也能引导着陆。

考虑最佳实践：使用连续接触奖励和高度奖励结合，加大引擎惩罚，并可能添加速度惩罚（在靠近地面时惩罚高速）。这样可避免硬着陆。

决定：

组件：
1. **goal_reward**: `1/(1+5*dist)` 不gate，或极轻gate。
2. **height_reward**: `1/(1+8*abs(y))` 加强梯度。
3. **contact_incentive**: `min(left_contact, right_contact) * (1 - abs(vy)/5) * (1 - abs(angle)/2)` 等，确保质量。
4. **engine_penalty**: -0.1 当action != 0。
5. **velocity_penalty (when close to ground)**: 如果高度小且脚未接触，对高速给予惩罚，以鼓励减速。可以用 `-abs(vy) * (1 - left_contact) * (1 - right_contact)` 但不宜太大。
6. 着陆成功大bonus：`if left_contact and right_contact and abs(vy) < 0.2 and abs(angle) < 0.2: landing_bonus = 5.0`，每步给，直到终止。

但需要小心不超出score。

从历史best的skeleton: contact_reward + gated_goal_reward，看来iter1只有这两项。可能是contact_reward是连续化的？名字是contact_reward，不是contact_bonus。所以之前iter1的contact_reward可能是一个逐步奖励。反正是best得分-10.16，我们应试图改进。

我们可以重用best的结构：gated_goal + contact_reward（连续的）。关于contact_reward，可以定义为 `min(left_contact, right_contact)` 加上速度质量。这样agent会学到降速着陆。

另外，环境事实中提到“本任务是一个二维飞行器的轨迹优化问题。主体从视口上方中央附近以随机初始受力出发，需要在最短时间内到达并稳定在中央目标着陆垫上”。因此有时间优化，但episode 1000步很长，我们希望agent快速着陆。所以我们可以加入时间惩罚，但之前没加。可以通过间接的引擎惩罚或距离惩罚来促进快速。也可以加入一个较小的每步惩罚（比如-0.01）促使agent尽快结束，但这可能会被agent误解导致坠落。但可行，因为如果着陆有奖励，agent会尝试快速着陆。可以考虑加一个常量-0.02 per step作为生存代价。但需要观察结果。这可能促使agent更快着陆，也能解决徘徊问题。但document没有明确要求时间，但目标提到“在最短时间内”，所以可以加入时间压力。不过我们要谨慎，先不加，先用前一轮策略改进。

我的改进计划：
- 移除或大幅调整height_reward，用连续接触奖励替代。
- 放宽gate，甚至去掉gate，仅保留goal_reward。
- 增加contact_incentive: 使用 `min(left_contact, right_contact) * landing_quality`。
- 引擎惩罚加到-0.1。
- 增加着陆成功bonus（当双脚接触且低速低角度）。
- 调整系数使奖励规模合理。

验证可能

# Search objective
- target_score: 200.000000
- current_score: -63.805763
- gap_to_target: 263.805763
- target_achievement_ratio: -31.903%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -63.805763）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
    x = obs[0]          # horizontal position relative to pad
    y = obs[1]          # vertical position relative to pad
    vx = obs[2]         # horizontal velocity
    vy = obs[3]         # vertical velocity
    angle = obs[4]      # body angle (rad)
    left_contact = obs[6]   # left leg contact flag
    right_contact = obs[7]  # right leg contact flag

    # --- Component A: goal-approach (scaled & gated) ---
    distance = (x**2 + y**2)**0.5
    goal_reward = 1.0 / (1.0 + 5.0 * distance)

    # Soft gate on speed and angle (slightly relaxed)
    speed_sq = vx**2 + vy**2
    speed_gate = 1.0 / (1.0 + 1.0 * speed_sq)   # relaxed from 2.0
    angle_gate = 1.0 / (1.0 + 0.5 * abs(angle))  # relaxed from 1.0
    gate = speed_gate * angle_gate
    gated_goal = goal_reward * gate

    # --- Component B: height incentive (encourage descent) ---
    # Give reward when y is small (close to pad). Bounded [0, 1].
    height_reward = 1.0 / (1.0 + 3.0 * abs(y))

    # --- Component C: contact bonus (stronger landing incentive) ---
    # Boost for both legs touching.
    contact_bonus = 2.0 * (left_contact * right_contact)  # 2.0 if both touch, else 0

    # --- Component D: engine usage penalty (fuel saving) ---
    # Penalize any action that fires an engine (action != 0)
    engine_penalty = -0.05 if action != 0 else 0.0

    # Total reward
    total_reward = gated_goal + height_reward + contact_bonus + engine_penalty

    components = {
        'gated_goal': gated_goal,
        'height_reward': height_reward,
        'contact_bonus': contact_bonus,
        'engine_penalty': engine_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-63.805763, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-97.520066, -25.163108]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| height_reward | 283.190755 | 54.6% | 54.6% | 100.0% |
| gated_goal | 185.916275 | 35.9% | 35.9% | 100.0% |
| engine_penalty | -49.310000 | -9.5% | 9.5% | 98.6% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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