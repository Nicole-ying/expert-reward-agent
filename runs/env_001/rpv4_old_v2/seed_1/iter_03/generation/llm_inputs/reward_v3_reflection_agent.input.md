# 1. Search objective
- target_score: 200.000000
- current_score: -104.327660
- gap_to_target: 304.327660
- target_achievement_ratio: -52.164%

# 2. 上一轮奖励函数代码（该轮得分: -104.327660）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next

    # 放宽的门控，使进展奖励更具驱动力
    speed_sq = vx_next**2 + vy_next**2
    gate_speed = 1.0 / (1.0 + 0.1 * speed_sq)
    gate_angle = 1.0 / (1.0 + 0.5 * abs(angle_next))
    gate_health = gate_speed * gate_angle

    progress_reward = 5.0 * progress_delta * gate_health

    # 着陆接触奖励（保持原有）
    contact_bonus = 0.0
    if left_contact_next > 0.5 and right_contact_next > 0.5:
        speed_temp = 0.2
        angle_temp = 0.1
        stable_factor = (2.718281828 ** (-speed_sq / speed_temp)) * \
                        (2.718281828 ** (-(angle_next**2) / angle_temp))
        contact_bonus = 10.0 * stable_factor

    # 燃料惩罚（保持原有）
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # 近地姿态惩罚（保持原有）
    orientation_penalty = 0.0
    if y_next < 0.5:
        orientation_penalty = -0.5 * (angle_next**2) - 0.2 * (abs(angvel_next)**2)

    total_reward = progress_reward + contact_bonus + fuel_penalty + orientation_penalty

    components = {
        "progress": progress_reward,
        "contact_bonus": contact_bonus,
        "fuel_penalty": fuel_penalty,
        "orientation_penalty": orientation_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 71.10 | -85.93 | ✅ |
| 2 | 骨架变化: contact_bonus + fuel_penalty + orientation_penalty | — | 301.40 | -104.33 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

# 训练数据报告

## 1. 行为事实
- 得分：score=-104.33，range=[-180.87, 31.48]
- 终止方式：terminated 19/20（成功/失败），truncated 1/20（时间上限）
- 回合长度：len=301.40
- Checkpoint 轨迹：无 checkpoint 数据（本次评估未提供官方得分曲线）

## 2. 骨架演进表

| iter | score | len | term | progress_ep_sum | progress_active | contact_bonus_ep_sum | contact_bonus_active | fuel_penalty_ep_sum | fuel_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active |
|------|--------|------|------|-----------------|-----------------|----------------------|---------------------|---------------------|---------------------|----------------------------|----------------------------|
| 1 | -85.93 | 71.10 | 20/20 (0 truncated) | 0.539 | 100.0% | 4.254 | 2.2% | -0.785 | 22.1% | -2.748 | 23.9% |
| 2 | -104.33 | 301.40 | 19/20 (1 truncated) | 2.207 | 100.0% | 3.128 | 0.3% | -11.463 | 76.1% | -1.587 | 5.1% |

说明：
- 所有数值直接取自 `episode_sum_mean`，`active_rate` 为百分比。
- iter 1 为初始版本（历史最佳，score=-85.93），iter 2 为本轮。

## 3. 数字标记

### 主导
- 无组件 magnitude_share > 80%（本轮最高 fuel_penalty 50.9%）。

### 死组件
- **contact_bonus**：active_rate = 0.3% (< 2%)

### 淹没
- 无组件同时满足 active_rate > 90% 且 magnitude_share < 3%。

### 全覆盖惩罚
- 无组件同时满足 active_rate > 95% 且 signed_share 为负。

### 量级对比（每步均值 = ep_sum / len，取两位小数）
- 总 len=301.40
  - fuel_penalty: -11.463 / 301.40 = -0.0380
  - orientation_penalty: -1.587 / 301.40 = -0.0053
  - progress: 2.207 / 301.40 = 0.0073
  - contact_bonus: 3.128 / 301.40 = 0.0104

**最大的两个**（按数值大小）：
1. contact_bonus: 0.0104
2. progress: 0.0073

**最小的两个**（按数值大小）：
1. fuel_penalty: -0.0380
2. orientation_penalty: -0.0053

### 修改变化（iter 1 → iter 2）
| 组件 | ep_sum Δ | active_rate Δ |
|------|----------|----------------|
| progress | +1.668 | 0% |
| contact_bonus | -1.125 | -1.9% |
| fuel_penalty | -10.678 | +54.0% |
| orientation_penalty | +1.161 | -18.8% |

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
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

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| contact_bonus | 0.048 | 0.167 | +0.119 | 1.6% | 3.3% | +1.7pp |
| fuel_penalty | -0.017 | -0.015 | +0.002 | 34.9% | 30.5% | -4.4pp |
| orientation_penalty | -0.043 | -0.039 | +0.004 | 22.4% | 27.3% | +4.9pp |
| progress | 0.008 | 0.058 | +0.050 | 100.0% | 100.0% | -0.0pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -85.93 | -104.33 | -18.39 |
| 平均回合长度 | 71.1 | 301.4 | +230.3 |
| 终止回合数 | 20/20 | 19/20 | -1 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-104.327660, len=301.400000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-180.870995, 31.475726]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| fuel_penalty | -11.462500 | -50.9% | 50.9% | 76.1% |
| progress | 2.206785 | 9.8% | 28.1% | 100.0% |
| contact_bonus | 3.128335 | 13.9% | 13.9% | 0.3% |
| orientation_penalty | -1.587194 | -7.1% | 7.1% | 5.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
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
| 2 | contact_bonus + fuel_penalty + orientation_penalty + progress | -104.33 | -85.93 | -18.39 | 301.40 | contact_bonus=0.167 fuel_penalty=-0.015 orientation_penalty=-0.039 progress=0.058 | no_meaningful_improvement |
