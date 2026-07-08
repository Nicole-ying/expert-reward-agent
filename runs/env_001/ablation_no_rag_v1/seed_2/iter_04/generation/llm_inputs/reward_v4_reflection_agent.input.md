# Search objective
- target_score: 200.000000
- current_score: -82.728679
- gap_to_target: 282.728679
- target_achievement_ratio: -41.364%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -82.728679）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：向目标中心的 progress（potential-based）
    dist_old = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_new = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_old - dist_new

    # 稳定着陆引导：连续乘积状态奖励，消除 transition exploit
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 接触因子：平均双脚接触（0~1），鼓励双脚均着地
    contact_factor = (left_contact + right_contact) / 2.0

    # curriculum：早期使用宽松的高斯尺度，后期收紧以要求精确
    sigma_p = 2.0 - 1.5 * training_progress  # 位置尺度：2.0 → 0.5
    sigma_v = 2.0 - 1.5 * training_progress  # 速度尺度：2.0 → 0.5
    sigma_a = 0.6 - 0.45 * training_progress  # 角度尺度：0.6 → 0.15

    # 组合指数项，同时要求靠近、低速、竖直
    exponent = -(
        (x ** 2 + y ** 2) / (2.0 * sigma_p ** 2) +
        (vx ** 2 + vy ** 2) / (2.0 * sigma_v ** 2) +
        (angle ** 2) / (2.0 * sigma_a ** 2)
    )

    # 使用 2.718281828**exponent 替代 exp；系数 10.0 提供足够推动力
    stable_landing_reward = 10.0 * contact_factor * (2.718281828 ** exponent)

    total_reward = progress_reward + stable_landing_reward

    components = {
        "progress_reward": progress_reward,
        "stable_landing_reward": stable_landing_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：每一步向目标靠近的 progress（potential-based shaping）
    dist_old = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_new = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = dist_old - dist_new

    # 稳定着陆 proxy：多条件连续乘积，引导低速、竖直、双脚接触且对准着陆垫中心
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 温度参数（根据环境大致范围设定，可在后续迭代中调整）
    sigma_p = 1.0    # 位置衰减尺度
    sigma_v = 1.0    # 速度衰减尺度
    sigma_a = 0.3    # 角度衰减尺度（弧度）

    # 距离、速度、角度项的指数和
    exponent = -(
        (x ** 2 + y ** 2) / (2.0 * sigma_p ** 2) +
        (vx ** 2 + vy ** 2) / (2.0 * sigma_v ** 2) +
        (angle ** 2) / (2.0 * sigma_a ** 2)
    )

    contact_factor = (left_contact + right_contact) / 2.0
    # 使用 2.718281828**exponent 替代 exp，避免 numpy 依赖
    stable_landing_reward = 10.0 * contact_factor * (2.718281828 ** exponent)

    total_reward = progress_reward + stable_landing_reward

    components = {
        "progress_reward": progress_reward,
        "stable_landing_reward": stable_landing_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-82.728679, len=858.200000, terminated=11/20, truncated=9/20, reward_errors=0
score_range=[-214.630059, 153.094025]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_landing_reward | 109.465266 | 98.7% | 98.7% | 6.9% |
| progress_reward | 0.708042 | 0.6% | 1.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从视口顶部中央附近出发，尽快飞到画面中心的着陆垫上并稳定停靠。过程中需要尽量节省引擎推力，同时避免发生碰撞、水平出界或提前休眠。最终期望的位置是着陆垫上方、接近静止、姿态稳定且左右支撑脚均与垫子接触。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据常规推断，其中 contact 标志用 0.0/1.0 表示）
- obs[0] (x_position): 飞行器相对着陆垫中心的水平坐标（可能是负、零或正）
- obs[1] (y_position): 飞行器相对着陆垫高度的垂直坐标（正值在上方，负值在下方）
- obs[2] (x_velocity): 水平线速度
- obs[3] (y_velocity): 垂直线速度
- obs[4] (body_angle): 机体朝向角度（弧度或度？具体范围由物理实现决定，但在奖励中应处理为接近 0 表示竖直姿态）
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左支撑脚是否接触着陆垫（0.0 无接触，1.0 接触）
- obs[7] (right_support_contact): 右支撑脚是否接触着陆垫（0.0 无接触，1.0 接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine —— 什么都不做（不点火，惯性和风力影响下自由飘动）
- action 1: left_orientation_engine —— 点火某个方向的姿态控制引擎（用于向左旋转机体）
- action 2: main_engine —— 点火主引擎（提供向上的推力，用于减速或上升）
- action 3: right_orientation_engine —— 点火相反方向的姿态控制引擎（用于向右旋转机体）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 飞行器在着陆垫上方稳定着陆，速度很小，角度接近 0，且左右支撑脚均接触（即 body_not_awake_or_settled 触发，同时位置、速度、角度、接触均满足期望状态）。该情形隐含“成功到达目标并安全停靠”。
- failure-like termination: 
  - crash_or_body_contact（飞行器主体与地面或边界发生破坏性碰撞）
  - horizontal_position_outside_viewport（飞行器水平飞出画面）
- ambiguous termination: body_not_awake_or_settled 可能既包含成功着陆（满足条件），也可能包含飞行器在非目标位置提前休眠（例如坠落在远处后不动、或能量耗尽停止）。因此该终止条件本身不能直接等同于成功，必须结合位置、接触等信息判断。
- truncation: 源码中为 `False`，未显示任何截断机制，可假设无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典，无任何显式标记）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空，无有效字段可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段（因为 info 为空，不能假设存在 success、failure 或其他标记）

## 7. 可用于奖励函数的信号
以下信号均可从 next_obs（或结合 obs 计算增量）中提取，且属于任务相关的可靠特征：

- position: x_position, y_position（obs[0], obs[1]）—— 可衡量距着陆目标的远近、是否处于垫子正上方
- velocity: x_velocity, y_velocity（obs[2], obs[3]）—— 速度大小影响安全着陆和平顺性
- orientation: body_angle（obs[4]）—— 姿态是否竖直（接近 0）
- angular_velocity: obs[5] —— 旋转是否稳定
- contact: left_support_contact, right_support_contact（obs[6], obs[7]）—— 是否双脚接触，判断着陆状态
- action/engine: 可依据动作是否使用主引擎或姿态引擎来鼓励节能（如惩罚主引擎使用）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + stable_landing_reward | -37.29 | -37.29 | 0.00 | 1000.00 | progress_reward=0.003 stable_landing_reward=4.135 | new_best |
| 2 | progress_reward + stable_landing_reward | -43.99 | -37.29 | -6.70 | 101.20 | progress_reward=0.014 stable_landing_reward=0.138 | no_meaningful_improvement |
| 3 | progress_reward + stable_landing_reward | -82.73 | -37.29 | -45.44 | 858.20 | progress_reward=0.003 stable_landing_reward=4.262 | no_meaningful_improvement |
