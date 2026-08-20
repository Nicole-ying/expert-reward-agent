# 1. Search objective
- target_score: 200.000000
- current_score: 227.061788
- gap_to_target: -27.061788
- target_achievement_ratio: 113.531%

# 2. 上一轮奖励函数代码（该轮得分: 227.061788）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------
    # Component A: progress_reward (improvement_delta)
    #   Potential = -(x^2 + y^2), reward = potential_curr - potential_next
    #   Positive when distance to pad decreases, zero when stationary.
    # ------------------------------------------------------------
    dist_curr = obs[0] ** 2 + obs[1] ** 2
    dist_next = next_obs[0] ** 2 + next_obs[1] ** 2
    progress_reward = 1.0 * (dist_curr - dist_next)

    # ------------------------------------------------------------
    # Component B: velocity_damping (proximity_gated quadratic penalty)
    #   Penalise speed more strongly when near the pad.
    # ------------------------------------------------------------
    dist_next_sqrt = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    gate = 1.0 / (1.0 + 2.0 * dist_next_sqrt)          # alpha = 2.0
    vel_penalty = 0.1 * (next_obs[2] ** 2 + next_obs[3] ** 2) * gate
    velocity_damping_reward = -vel_penalty

    # ------------------------------------------------------------
    # Component C: orientation_stability (quadratic_penalty)
    #   Keep body upright and angular velocity small.
    # ------------------------------------------------------------
    angle_penalty = 1.0 * (next_obs[4] ** 2)
    angvel_penalty = 0.2 * (next_obs[5] ** 2)
    orientation_stability_reward = -(angle_penalty + angvel_penalty)

    # ------------------------------------------------------------
    # Component D: safe_contact_reward (joint_condition_proxy)
    #   Give positive reward when both legs touch the pad,
    #   vehicle is at pad height, and vertical velocity is low.
    #   (Coefficient increased from 0.5 to 1.0)
    # ------------------------------------------------------------
    both_contacts = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    # y position close to zero (pad height)
    proximity_to_pad = max(0.0, 1.0 - abs(next_obs[1]) / 0.2)   # threshold_y = 0.2
    # vertical velocity small
    soft_vel = max(0.0, 1.0 - abs(next_obs[3]) / 0.2)           # threshold_vy = 0.2
    safe_contact_reward = 1.0 * both_contacts * proximity_to_pad * soft_vel

    total_reward = (progress_reward +
                    velocity_damping_reward +
                    orientation_stability_reward +
                    safe_contact_reward)

    components = {
        "progress_reward": progress_reward,
        "velocity_damping_reward": velocity_damping_reward,
        "orientation_stability_reward": orientation_stability_reward,
        "safe_contact_reward": safe_contact_reward,
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 965.45 | 144.17 | ✅ |
| 2 | 骨架变化: orientation_stability_reward + progress_reward + s | — | 604.20 | 124.27 | ❌ |
| 3 | 骨架变化: action_penalty + landing_proxy_reward + orientatio | — | 528.55 | 110.44 | ❌ |
| 4 | 骨架变化: orientation_stability_reward + progress_reward + s | — | 490.10 | 227.06 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

# 行为事实
- 得分：score=227.06，range=[190.92, 273.11]
- 终止方式：terminated 20/20（全部成功），truncated 0/20
- 回合长度：len=490.10
- Checkpoint 轨迹：无数据

# 骨架演进表
| iter | score | len | term | prog_ep_sum | prog_act | vel_ep_sum | vel_act | ori_ep_sum | ori_act | safe_ep_sum | safe_act | land_ep_sum | land_act | pen_ep_sum | pen_act |
|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 144.17 | 965.45 | T:1, Tr:19 | 1.99 | 100.0% | -1.29 | 100.0% | -3.95 | 100.0% | 312.46 | 65.2% | — | — | — | — |
| 2 | 124.27 | 604.20 | T:13, Tr:7 | 1.88 | 99.2% | -1.44 | 99.2% | -9.48 | 100.0% | 852.00 | 7.1% | — | — | — | — |
| 3 | 110.44 | 528.55 | T:18, Tr:2 | 3.64 | 99.8% | — | — | -15.55 | 100.0% | — | — | 359.97 | 24.5% | -15.68 | 98.9% |
| 4 | 227.06 | 490.10 | T:20, Tr:0 | 1.99 | 97.1% | -1.28 | 97.2% | -3.04 | 100.0% | 33.09 | 7.0% | — | — | — | — |

说明：`—` 表示该版本无此组件。`T` = terminated, `Tr` = truncated。

# 数字标记
- **主导**：safe_contact_reward magnitude_share 83.9% (>80%)
- **死组件**：无
- **淹没**：无
- **全覆盖惩罚**：
  - orientation_stability_reward：active_rate 100% (>95%)，signed_share -7.7%（负）
  - velocity_damping_reward：active_rate 97.2% (>95%)，signed_share -3.2%（负）
- **量级对比**（每步均值 = episode_sum_mean / len = 490.1）：
  - 最大两个：safe_contact_reward (+0.0675)，progress_reward (+0.0041)
  - 最小两个：orientation_stability_reward (-0.0062)，velocity_damping_reward (-0.0026)
- **修改变化**（iter 4 相对于 iter 3 有共同组件的 Δ）：
  - orientation_stability_reward：ep_sum Δ = +12.52（-3.04 vs -15.55），active_rate Δ = 0
  - progress_reward：ep_sum Δ = -1.66（1.99 vs 3.64），active_rate Δ = -2.7%
  - 其余组件在两轮间无可比数据（iter 3 无 safe_contact/velocity_damping，iter 4 无 landing_proxy/action_penalty）。

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| action_penalty | -0.020 | — (新增) | 已删除 | 67.1% | — | — |
| landing_proxy_reward | 1.192 | — (新增) | 已删除 | 37.7% | — | — |
| orientation_stability_reward | -0.078 | -0.020 | +0.058 | 100.0% | 100.0% | -0.0pp |
| progress_reward | 0.021 | 0.005 | -0.016 | 99.9% | 99.8% | -0.1pp |
| safe_contact_reward | — (新增) | 0.476 | 新增 | — | 50.1% | — |
| velocity_damping_reward | — (新增) | -0.008 | 新增 | — | 99.8% | — |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | 110.44 | 227.06 | +116.62 |
| 平均回合长度 | 528.5 | 490.1 | -38.4 |
| 终止回合数 | 18/20 | 20/20 | +2 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=227.061788, len=490.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[190.921549, 273.107479]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 33.088806 | 83.9% | 83.9% | 7.0% |
| orientation_stability_reward | -3.035867 | -7.7% | 7.7% | 100.0% |
| progress_reward | 1.987577 | 5.0% | 5.2% | 97.1% |
| velocity_damping_reward | -1.277038 | -3.2% | 3.2% | 97.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个 2D 飞行器从画面上方中心附近出发，受初始随机力作用。核心目标是**到达并稳定在中心目标平台上**（即精准着陆并停稳）。任务同时要求**尽可能快**地完成着陆，并且**尽可能少使用发动机推力**。在过程中，智能体需要学会接近目标、降低速度、保持姿态稳定并实现安全接触（避免硬撞击）。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: float32（推测）  
- **obs[0]**: `x_position`，水平坐标相对于目标 pad，可用于衡量横向距离，reward_usable: true  
- **obs[1]**: `y_position`，垂直坐标相对于 pad 高度，可用于衡量高度，reward_usable: true  
- **obs[2]**: `x_velocity`，水平线速度，用于速度控制，reward_usable: true  
- **obs[3]**: `y_velocity`，垂直线速度，用于软着陆控制，reward_usable: true  
- **obs[4]**: `body_angle`，机体角度，用于姿态稳定性，reward_usable: true  
- **obs[5]**: `angular_velocity`，角速度，用于姿态稳定，reward_usable: true  
- **obs[6]**: `left_support_contact`，左侧支撑接触标志（0/1），用于判断着陆状态，reward_usable: true  
- **obs[7]**: `right_support_contact`，右侧支撑接触标志（0/1），用于判断着陆状态，reward_usable: true

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  
- **action 0**: `no_engine`，无发动机动作（什么都不做）  
- **action 1**: `left_orientation_engine`，左姿态发动机，提供侧向推力或扭矩  
- **action 2**: `main_engine`，主发动机，提供向上推力  
- **action 3**: `right_orientation_engine`，右姿态发动机，提供另一侧推力或扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` – 可能表示机体不再活跃或已稳定（例如成功着陆后静止），但也可能包含坠落静止等失败情况。**无显式成功标志**。  
- **failure-like termination**:  
  - `crash_or_body_contact` – 机体与地面或障碍物碰撞导致的失败。  
  - `horizontal_position_outside_viewport` – 水平位置超出画面边界，视为失败。  
- **ambiguous termination**: 上述 settled 条件含义模糊，其设计意图是探测着陆稳定，但不能直接作为可靠的成功指标。  
- **truncation**: 未提供任何时间步或步数限制的截断信号。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
- `explicit_failure_flag_available`: **false**  
- `allowed_info_fields`: 环境返回空字典，无可用的额外信号。  
- `forbidden_or_uncertain_info_fields`: 所有未声明的 info 字段均不可用，且不应假设存在 `success`、`failure` 或 `termination_reason`。

## 7. 可用于奖励函数的信号
- **position**: `obs[0]`（x_position）、`obs[1]`（y_position），可计算到目标 pad 的距离；当两者趋近于 0 时表示到达。  
- **velocity**: `obs[2]`（x_velocity）、`obs[3]`（y_velocity），可用于惩罚或塑造速度，尤其在接近目标时。  
- **orientation**: `obs[4]`（body_angle）、`obs[5]`（angular_velocity），用于保持竖直与姿态稳定。  
- **contact**: `obs[6]`（left_support_contact）、`obs[7]`（right_support_contact），指示是否已接触环境，用于判断着陆状态与安全性。  
- **action/engine**: `action` 取值 1/2/3 代表发动机点火，可计算燃料消耗代价。  
- **other**: 无其他明确可用的信号。

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 144.17 | 144.17 | 0.00 | 965.45 | orientation_stability_reward=-0.019 progress_reward=0.005 safe_contact_reward=0.278 velocity_damping_reward=-0.007 | new_best |
| 2 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 124.27 | 144.17 | -19.90 | 604.20 | orientation_stability_reward=-0.030 progress_reward=0.008 safe_contact_reward=1.037 velocity_damping_reward=-0.013 | no_meaningful_improvement |
| 3 | action_penalty + landing_proxy_reward + orientation_stability_reward + progress_reward | 110.44 | 144.17 | -33.74 | 528.55 | action_penalty=-0.020 landing_proxy_reward=1.192 orientation_stability_reward=-0.078 progress_reward=0.021 | no_meaningful_improvement |
| 4 | orientation_stability_reward + progress_reward + safe_contact_reward + velocity_damping_reward | 227.06 | 227.06 | 0.00 | 490.10 | orientation_stability_reward=-0.020 progress_reward=0.005 safe_contact_reward=0.476 velocity_damping_reward=-0.008 | target_solved_new_best |
