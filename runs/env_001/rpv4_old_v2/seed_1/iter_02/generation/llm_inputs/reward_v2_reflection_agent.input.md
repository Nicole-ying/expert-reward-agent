# 1. Search objective
- target_score: 200.000000
- current_score: -85.932872
- gap_to_target: 285.932872
- target_achievement_ratio: -42.966%

# 2. 上一轮奖励函数代码（该轮得分: -85.932872）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D着陆器奖励函数 v1
    使用 observation [8] 和 action (0-3) 计算奖励
    """
    # ---------- 从观测中提取变量 ----------
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 核心进展：到目标(0,0)的距离变化 ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next   # 正 = 接近目标

    # ---------- soft_health_gate：根据下一步速度与姿态衰减进展奖励 ----------
    speed_sq = vx_next**2 + vy_next**2      # 线速度平方
    speed = speed_sq**0.5
    # 门控系数：速度越高、倾角越大，gate 越小
    gate_speed = 1.0 / (1.0 + 0.5 * speed_sq)
    gate_angle = 1.0 / (1.0 + 2.0 * abs(angle_next))
    gate_health = gate_speed * gate_angle

    # progress_reward 已经带健康门控
    progress_reward = progress_delta * gate_health

    # ---------- soft_landing_contact：双腿同时接触并满足稳定条件 ----------
    contact_bonus = 0.0
    if left_contact_next > 0.5 and right_contact_next > 0.5:
        # 速度与姿态的指数衰减因子
        speed_temp = 0.2      # 速度尺度
        angle_temp = 0.1      # 角度尺度
        stable_factor = (2.718281828 ** (-speed_sq / speed_temp)) * \
                        (2.718281828 ** (-(angle_next**2) / angle_temp))
        contact_bonus = 10.0 * stable_factor   # 基础分10，由稳定性缩放

    # ---------- fuel_penalty：惩罚引擎使用 ----------
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # ---------- orientation_stability：接近地面时额外惩罚姿态偏差 ----------
    orientation_penalty = 0.0
    if y_next < 0.5:   # 高度阈值，接近着陆垫
        orientation_penalty = -0.5 * (angle_next**2) - 0.2 * (abs(angvel_next)**2)

    # ---------- 组合总奖励 ----------
    total_reward = progress_reward + contact_bonus + fuel_penalty + orientation_penalty

    # 记录各组件，便于调试
    components = {
        "progress": progress_reward,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "orientation_penalty": orientation_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 4. Subagent 行为诊断报告

# 数据报告

## 1. 行为事实
- 得分：score=-85.93，range=[-136.39, -28.36]
- 终止方式：terminated 20/20（全部失败），truncated 0/20
- 回合长度：len=71.10
- Checkpoint 轨迹：无数据（pct 表为空）

## 2. 骨架演进表
| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | orientation_penalty_ep_sum | orientation_penalty_active | fuel_penalty_ep_sum | fuel_penalty_active | progress_ep_sum | progress_active |
|------|-------|-----|------|---------------------|---------------------|---------------------------|---------------------------|---------------------|---------------------|-----------------|-----------------|
| 1    | -85.93 | 71.10 | terminated=20/20, truncated=0/20 | 4.2538 | 2.2% | -2.7482 | 23.9% | -0.7850 | 22.1% | 0.5393 | 100.0% |

## 3. 数字标记
- **主导**：无（无组件 magnitude_share > 80%）
- **死组件**：无（active_rate 最低为 contact_bonus 2.2%，>2% 阈值）
- **淹没**：无（progress active_rate 100% 但 magnitude_share 6.8% > 3%）
- **全覆盖惩罚**：无（无组件满足 active_rate > 95% 且 signed_share < 0）
- **量级对比**（每步均值 = episode_sum_mean / len）：
  - 最大：contact_bonus 0.0598，progress 0.0076
  - 最小：orientation_penalty -0.0387，fuel_penalty -0.0110
- **修改变化**：无上一轮数据，Δ 为空

# 5. 组件逐项对比
（第一轮反思，无历史对比）

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-85.932872, len=71.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-136.391192, -28.364158]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 4.253808 | 50.9% | 50.9% | 2.2% |
| orientation_penalty | -2.748167 | -32.9% | 32.9% | 23.9% |
| fuel_penalty | -0.785000 | -9.4% | 9.4% | 22.1% |
| progress | 0.539318 | 6.5% | 6.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 16/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：控制一个 2D 平面内的刚体飞行器（带有主引擎和两侧姿态引擎），从初始偏高位置出发，尽可能快地下降并稳定、安全地停在中央目标垫上。停靠时要求速度极低、姿态竖直、双腿同时接触垫面（soft settling）。  
次要目标：在达成主目标的前提下，最小化引擎推力使用（燃料效率），并避免坠毁、飞出视口或粗鲁撞击。  
不应混淆的目标：单纯省燃料而拒绝降落、长时间悬停但不靠近垫子、快速但剧烈碰撞都属于失败行为。

## 3. 观察空间 observation_space
- **type:** Box  
- **shape:** [8]  
- **dtype:** 假定为 float32，连续值；最后两维为 float 但实际是 0/1  
- **各维含义（下述均为可用于奖励计算的信号）：**  

| index | name                         | meaning                                                                   | reward_usable |
|-------|------------------------------|---------------------------------------------------------------------------|---------------|
| 0     | x_position                   | 刚体中心相对于目标垫中心的水平距离（正方向未知，可正可负）               | true          |
| 1     | y_position                   | 刚体中心相对于目标垫高度的垂直距离（正值表示在垫上方）                   | true          |
| 2     | x_velocity                   | 水平线速度                                                                 | true          |
| 3     | y_velocity                   | 垂直线速度（正值向上）                                                     | true          |
| 4     | body_angle                   | 刚体姿态角（弧度），0 可能表示竖直向上                                     | true          |
| 5     | angular_velocity             | 角速度                                                                     | true          |
| 6     | left_support_contact         | 左支撑腿是否接触目标垫/地面（1：接触，0：未接触）                         | true          |
| 7     | right_support_contact        | 右支撑腿是否接触目标垫/地面（1：接触，0：未接触）                         | true          |

## 4. 动作空间 action_space
- **type:** Discrete  
- **n:** 4  
- **动作说明：**  

| action_id | name                        | meaning                           |
|-----------|-----------------------------|-----------------------------------|
| 0         | no_engine                   | 不点火任何引擎（滑行）            |
| 1         | left_orientation_engine     | 点燃左侧姿态引擎（产生一个力矩和可能的侧向推力） |
| 2         | main_engine                 | 点燃主引擎（产生向上的推力，可能带少量横向分量） |
| 3         | right_orientation_engine    | 点燃右侧姿态引擎（与左引擎对称）  |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination:** `body_not_awake_or_settled` —— 当刚体速度/角速度低到被视为“睡着”或稳定停靠时触发，这极有可能是成功着陆的信号（双腿接触、速度近零）。  
- **failure‑like termination:**  
   - `crash_or_body_contact` —— 与目标垫以外的其他物体（如地面、边界）发生危险碰撞或翻滚。  
   - `horizontal_position_outside_viewport` —— 飞出水平视口，代表完全失控。  
- **ambiguous termination:** `body_not_awake_or_settled` 也可能因为某些物理休眠设定错误触发，但通常视为成功。  
- **truncation:** 当前环境未提供明确的截断条件（可能具有最大步数限制，未在 spec 中列出，属于环境内部实现）。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** false（`info` 字典为空 {}，无 success 标记）  
- **explicit_failure_flag_available:** false（同上）  
- **allowed_info_fields:** 无（`info` 为空，不可用任何字段）  
- **forbidden_or_uncertain_info_fields:** 所有 `info` 字段均禁止使用；`original_reward` 必须被屏蔽，禁止读取。

## 7. 可用于奖励函数的信号
- **位置相关：**
  - `obs[0]`（x_position） / `next_obs[0]`
  - `obs[1]`（y_position） / `next_obs[1]`
- **速度相关：**
  - `obs[2]`（x_velocity） / `next_obs[2]`
  - `obs[3]`（y_velocity） / `next_obs[3]`
- **姿态相关：**
  - `obs[4]`（body_angle） / `next_obs[4]`
  - `obs[5]`（angular_velocity） / `next_obs[5]`
- **接触/着陆信号：**
  - `obs[6]`（left_support_contact） / `next_obs[6]`
  - `obs[7]`（right_support_contact） / `next_obs[7]`
- **动作/推力：**
  - `action` （离散值）可用来推断哪个引擎被点燃，从而施加推力惩罚
- **其他派生信号：**
  - 从原始信号可以衍生出 `dist_to_target = sqrt(x^2 + y^2)`、`speed = sqrt(vx^2 + vy^2)`、`both_legs_contact = left_contact & right_contact` 等，均为合法使用。

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + fuel_penalty + orientation_penalty + progress | -85.93 | -85.93 | 0.00 | 71.10 | contact_bonus=0.048 fuel_penalty=-0.017 orientation_penalty=-0.043 progress=0.008 | new_best |
