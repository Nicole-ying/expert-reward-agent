# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: runs\env_001\paper_lander_deres_main_frozen_20260704\seed_0\iter_03\generation\validations\reward_v3.validation.json
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
```python
import numpy as np


def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # 目标位置（平台中心）定义为 (0,0)，因为 obs[0], obs[1] 是相对偏移
    # ------------------------------------------------------------

    # 1. 主学习信号：朝向目标的进度
    dist_curr = np.sqrt(obs[0] ** 2 + obs[1] ** 2)
    dist_next = np.sqrt(next_obs[0] ** 2 + next_obs[1] ** 2)
    progress = 1.0 * (dist_curr - dist_next)  # 正值表示接近目标

    # 2. 稳定性约束：抑制高速、大角度和高角速度，确保着陆平稳
    linear_speed_cost = abs(next_obs[2]) + abs(next_obs[3])
    angle_cost = abs(next_obs[4])
    angular_cost = abs(next_obs[5])

    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = (
        -w_vel * linear_speed_cost - w_angle * angle_cost - w_angvel * angular_cost
    )

    # 3. 着陆质量：连续有界信号，替代稀疏 soft_landing_bonus
    #    在目标附近提供渐进梯度，引导减速、调姿、双足接触
    speed_next = np.sqrt(next_obs[2] ** 2 + next_obs[3] ** 2)

    # 接近度：高斯衰减，dist=0 时值为 1，dist≈1 时降至 ~0.37
    proximity = np.exp(-(dist_next ** 2) / 1.0)

    # 速度质量：低速得高分，speed≈0.7 时降至 ~0.37
    speed_quality = np.exp(-(speed_next ** 2) / 0.5)

    # 角度质量：水平姿态得高分，|angle|≈0.3 时降至 ~0.37
    angle_quality = np.exp(-(next_obs[4] ** 2) / 0.1)

    # 接触质量：乘积形式要求双足同时接触才能显著得分
    contact_quality = next_obs[6] * next_obs[7]

    # 综合着陆质量：proximity 作为全局门控，内部三因子取均值
    landing_quality = (
        2.0 * proximity * (speed_quality + angle_quality + contact_quality) / 3.0
    )

    # 总奖励
    total_reward = progress + stability_penalty + landing_quality

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "landing_quality": landing_quality,
    }

    return float(total_reward), components
```

# 上一轮奖励函数代码（该轮得分: -115.609624）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # 目标位置（平台中心）定义为 (0,0)，因为 obs[0], obs[1] 是相对偏移
    # ------------------------------------------------------------
    # 1. 主学习信号：朝向目标的进度
    dist_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 1.0 * (dist_curr - dist_next)  # 正值表示接近目标

    # 2. 稳定性约束：大幅降低系数，使 |penalty/progress| ≈ 0.1
    linear_speed_cost = abs(next_obs[2]) + abs(next_obs[3])
    angle_cost = abs(next_obs[4])
    angular_cost = abs(next_obs[5])

    w_vel = 0.001      # 原 0.01 → 降10倍
    w_angle = 0.001    # 原 0.01 → 降10倍
    w_angvel = 0.0005  # 原 0.005 → 降10倍
    stability_penalty = - w_vel * linear_speed_cost - w_angle * angle_cost - w_angvel * angular_cost

    # 3. 软着陆近似奖励：多条件组合，引导飞行器低速、低角度、双足接触着陆
    near_target = dist_next < 0.5
    low_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5 < 0.5
    stable_angle = abs(next_obs[4]) < 0.1
    both_contacts = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    soft_landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contacts) else 0.0

    # 总奖励
    total_reward = progress + stability_penalty + soft_landing_bonus

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # 目标位置（平台中心）定义为 (0,0)，因为 obs[0], obs[1] 是相对偏移
    # ------------------------------------------------------------
    # 1. 主学习信号：朝向目标的进度
    dist_curr = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = 1.0 * (dist_curr - dist_next)  # 正值表示接近目标

    # 2. 稳定性约束：抑制高速、大角度和高角速度，确保着陆平稳
    linear_speed_cost = abs(next_obs[2]) + abs(next_obs[3])
    angle_cost = abs(next_obs[4])
    angular_cost = abs(next_obs[5])

    w_vel = 0.01
    w_angle = 0.01
    w_angvel = 0.005
    stability_penalty = - w_vel * linear_speed_cost - w_angle * angle_cost - w_angvel * angular_cost

    # 3. 软着陆近似奖励：多条件组合，引导飞行器低速、低角度、双足接触着陆
    near_target = dist_next < 0.5
    low_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5 < 0.5
    stable_angle = abs(next_obs[4]) < 0.1
    both_contacts = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    soft_landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contacts) else 0.0

    # 总奖励
    total_reward = progress + stability_penalty + soft_landing_bonus

    components = {
        "progress": progress,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-115.609624, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.097838, -99.292225]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress | 1.119159 | 63.6% | 65.9% | 100.0% |
| soft_landing_bonus | 0.500000 | 28.4% | 28.4% | 0.4% |
| stability_penalty | -0.099744 | -5.7% | 5.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
环境模拟一个 2D 飞行器轨迹优化问题。一个刚体从视口顶部中心附近出发，带有随机初始扰动力。目标是**尽可能快地飞到中央目标平台并平稳降落**，同时尽量少用引擎推力。智能体需要学会靠近目标、减小速度、保持稳定姿态，并安全接触平台。

简化为一句话：以最小的燃料消耗和最短时间，精准降落到固定的目标平台上并稳定下来。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float64 或 float32 (具体取决于底层实现，均为浮点)
- obs[0]: x_position —— 飞行器水平位置相对于目标平台的水平偏移
- obs[1]: y_position —— 飞行器垂直位置相对于平台高度
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体朝向角
- obs[5]: angular_velocity —— 机体旋转角速度
- obs[6]: left_support_contact —— 左侧支撑点是否接触（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact —— 右侧支撑点是否接触（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine —— 不点火，无推力
- action 1: left_orientation_engine —— 点燃左侧姿态调节发动机（产生旋转力矩）
- action 2: main_engine —— 点燃主发动机（产生与机体角度相关的推力，通常用于减速/悬浮）
- action 3: right_orientation_engine —— 点燃右侧姿态调节发动机（产生相反旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` —— 机体速度与角速度极低且可能已接触，视为稳定着陆。
- failure-like termination:  
  `horizontal_position_outside_viewport` —— 水平位置超出边界，飞行器丢失或失控。
- ambiguous termination:  
  `crash_or_body_contact` —— 字面上包含“撞击”和“身体接触”，可能包含不安全的碰撞（失败）或刚好接触平台但未稳定（失败）甚至安全接触（成功）。由于无法从返回值直接区分，除非结合其他状态判断，否则不可作为干净的成败信号。
- truncation:  
  无。step 返回元组中 truncated 恒为 False，说明没有时间步上限截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 恒为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在，不可用

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]) 和 y_position (obs[1])，可计算到目标的距离
- velocity: x_velocity (obs[2]) 和 y_velocity (obs[3])，可评估降落平稳性
- orientation: body_angle (obs[4]) 和 angular_velocity (obs[5])，可衡量姿态稳定性
- contact: left_support_contact (obs[6]) 和 right_support_contact (obs[7])，可判断是否已经接触、是否双足着地
- action/engine: 当前动作是否使用引擎（0为非零推力，2为主推，1、3为姿态推），可评估燃料消耗