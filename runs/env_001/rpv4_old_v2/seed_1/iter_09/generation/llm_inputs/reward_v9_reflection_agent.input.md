# 1. Search objective
- target_score: 200.000000
- current_score: -54.722843
- gap_to_target: 254.722843
- target_achievement_ratio: -27.361%

# 2. 上一轮奖励函数代码（该轮得分: -54.722843）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- progress reward ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next

    speed_sq = vx_next**2 + vy_next**2
    gate_speed = 1.0 / (1.0 + 0.1 * speed_sq)
    gate_angle = 1.0 / (1.0 + 0.5 * abs(angle_next))
    gate_health = gate_speed * gate_angle
    progress_reward = 5.0 * progress_delta * gate_health

    # ---------- contact_bonus: near-ground continuous guidance + contact lump sum ----------
    descent_target_vy = -1.0   # 加快下降的期望速度
    vy_error = vy_next - descent_target_vy
    soft_factor = (2.718281828 ** (-(vy_error**2) / 0.1)) * \
                  (2.718281828 ** (-(vx_next**2) / 0.1)) * \
                  (2.718281828 ** (-(angle_next**2) / 0.05)) * \
                  (2.718281828 ** (-(angvel_next**2) / 0.05))

    contact_bonus = 0.0
    if y_next < 1.0:
        contact_bonus = 5.0 * soft_factor   # 近地连续引导

    if left_contact_next > 0.5 and right_contact_next > 0.5:
        contact_bonus += 30.0 * soft_factor # 接触时额外稳定性奖励

    # ---------- fuel penalty ----------
    fuel_penalty = -0.05 if action != 0 else 0.0

    # ---------- near-ground orientation penalty ----------
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
| 3 | 骨架变化: contact_bonus + fuel_penalty + orientation_penalty | — | 900.70 | 91.82 | ✅ |
| 4 | 骨架变化: contact_bonus + fuel_penalty + orientation_penalty | — | 989.50 | 13.47 | ❌ |
| 5 | 骨架变化: contact_bonus + fuel_penalty + orientation_penalty | — | 863.40 | -39.29 | ❌ |
| 6 | 骨架变化: contact_bonus + fuel_penalty + orientation_penalty | — | 611.10 | -20.44 | ➖ |
| 7 | 骨架变化: contact_bonus + fuel_penalty + orientation_penalty | — | 656.00 | -133.27 | ❌ |
| 8 | 骨架变化: contact_bonus + fuel_penalty + orientation_penalty | — | 162.35 | -54.72 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 4. Subagent 行为诊断报告

# 训练数据报告

## 1. 行为事实

- **得分**：score = –54.72，range = [–780.97, 8.90]
- **终止方式**：terminated = 20/20（全部坠毁/终止），truncated = 0/20（无时间截断）
- **回合长度**：len = 162.35 步
- **Checkpoint 轨迹**：本轮未提供 checkpoint 评估数据（无数据）

**各组件在固定评估中的行为数值**  
episode_sum_mean（ep_sum）与 active_rate：

| component | ep_sum | active_rate |
|---|---|---|
| contact_bonus | 174.654 | 33.7% |
| progress | 4.250 | 100.0% |
| fuel_penalty | –4.930 | 60.7% |
| orientation_penalty | –0.542 | 17.3% |

## 2. 骨架演进表

组件顺序统一为：A = contact_bonus, B = progress, C = fuel_penalty, D = orientation_penalty。  
“term”列格式为 `terminated/truncated`。

| iter | score | len | term | compA_ep_sum | compA_active | compB_ep_sum | compB_active | compC_ep_sum | compC_active | compD_ep_sum | compD_active |
|------|-------|------|------|---------------|--------------|---------------|--------------|---------------|--------------|---------------|--------------|
| 1 | -85.93 | 71.10 | 20/0 | 4.254 | 2.2% | 0.539 | 100.0% | -0.785 | 22.1% | -2.748 | 23.9% |
| 2 | -104.33 | 301.40 | 19/1 | 3.128 | 0.3% | 2.207 | 100.0% | -11.463 | 76.1% | -1.587 | 5.1% |
| 3 | 91.82 | 900.70 | 5/15 | 1029.163 | 94.4% | 6.361 | 100.0% | -45.035 | 100.0% | -0.634 | 85.2% |
| 4 | 13.47 | 989.50 | 3/17 | 614.993 | 93.9% | 24.984 | 99.8% | -48.768 | 98.6% | -0.433 | 83.3% |
| 5 | -39.29 | 863.40 | 11/9 | 2866.351 | 91.8% | 4.049 | 99.8% | -39.860 | 92.3% | -2.026 | 76.7% |
| 6 | -20.44 | 611.10 | 14/6 | 3244.916 | 13.7% | 4.204 | 100.0% | -23.950 | 78.4% | -2.135 | 67.1% |
| 7 | -133.27 | 656.00 | 18/2 | 561.361 | 1.1% | 2.059 | 100.0% | -22.590 | 68.9% | -1.352 | 83.2% |
| 8 (当前) | -54.72 | 162.35 | 20/0 | 174.654 | 33.7% | 4.250 | 100.0% | -4.930 | 60.7% | -0.542 | 17.3% |

## 3. 数字标记

（以下仅基于 **本轮 iter_08** 的评估数据，与 iter_07 对比的变化部分使用上一轮数据）

- **主导**：contact_bonus 的 magnitude_share = 92.7%（>80%），属于主导组件。
- **死组件**：无（所有组件 active_rate 均 ≥ 2%）。
- **淹没**：无（active_rate > 90% 的组件 progress、fuel_penalty、orientation_penalty 的 magnitude_share 分别为 4.4%、2.6%、0.3%，其中 progress 的 magnitude_share 未低于 3%，另两者 active_rate 未超过 90%）。
- **全覆盖惩罚**：无（active_rate > 95% 的组件 progress 的 signed_share 为 +2.3%，不为负）。
- **量级对比**（每步均值 = ep_sum ÷ len）：  
  最大两个：contact_bonus (≈1.0756) > progress (≈0.0262)  
  最小两个：orientation_penalty (≈–0.00334) > fuel_penalty (≈–0.03036)  
  （按 signed 值排序）
- **修改变化**（iter_08 相对于 iter_07 的差分）：
  - contact_bonus: Δep_sum = –386.707, Δactive_rate = +32.6 百分点
  - progress: Δep_sum = +2.191, Δactive_rate = 0 百分点
  - fuel_penalty: Δep_sum = +17.660, Δactive_rate = –8.2 百分点
  - orientation_penalty: Δep_sum = +0.810, Δactive_rate = –65.9 百分点

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # ---------- 进展奖励（保持原样） ----------
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5
    progress_delta = dist_obs - dist_next

    speed_sq = vx_next**2 + vy_next**2
    gate_speed = 1.0 / (1.0 + 0.1 * speed_sq)
    gate_angle = 1.0 / (1.0 + 0.5 * abs(angle_next))
    gate_health = gate_speed * gate_angle
    progress_reward = 5.0 * progress_delta * gate_health

    # ---------- 重新设计的 contact_bonus：近地连续引导 + 接触高额奖励 ----------
    contact_bonus = 0.0
    descent_target_vy = -0.5   # 期望的下降速度
    vy_error = vy_next - descent_target_vy
    soft_factor = (2.718281828 ** (-(vy_error**2) / 0.1)) * \
                  (2.718281828 ** (-(vx_next**2) / 0.1)) * \
                  (2.718281828 ** (-(angle_next**2) / 0.05)) * \
                  (2.718281828 ** (-(angvel_next**2) / 0.05))

    if y_next < 1.0:
        contact_bonus = 5.0 * soft_factor   # 近地鼓励良好的下降与姿态

    if left_contact_next > 0.5 and right_contact_next > 0.5:
        contact_bonus += 30.0 * soft_factor # 接触时大额稳定性奖励

    # ---------- 燃料惩罚（保持原样） ----------
    fuel_cost = 0.05
    fuel_penalty = -fuel_cost if action != 0 else 0.0

    # ---------- 近地姿态惩罚（保持原样） ----------
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

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| contact_bonus | 33.531 | 1.047 | -32.484 | 36.0% | 46.4% | +10.4pp |
| fuel_penalty | -0.033 | -0.038 | -0.005 | 65.9% | 76.0% | +10.1pp |
| orientation_penalty | -0.021 | -0.017 | +0.004 | 62.3% | 22.1% | -40.1pp |
| progress | 0.025 | 0.044 | +0.020 | 99.7% | 100.0% | +0.3pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -133.27 | -54.72 | +78.54 |
| 平均回合长度 | 656.0 | 162.3 | -493.6 |
| 终止回合数 | 18/20 | 20/20 | +2 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-54.722843, len=162.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-780.973055, 8.898390]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 174.654002 | 92.7% | 92.7% | 33.7% |
| progress | 4.250309 | 2.3% | 4.4% | 100.0% |
| fuel_penalty | -4.930000 | -2.6% | 2.6% | 60.7% |
| orientation_penalty | -0.542035 | -0.3% | 0.3% | 17.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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
| 3 | contact_bonus + fuel_penalty + orientation_penalty + progress | 91.82 | 91.82 | 0.00 | 900.70 | contact_bonus=1.736 fuel_penalty=-0.034 orientation_penalty=-0.012 progress=0.011 | new_best |
| 4 | contact_bonus + fuel_penalty + orientation_penalty + progress | 13.47 | 91.82 | -78.35 | 989.50 | contact_bonus=1.715 fuel_penalty=-0.031 orientation_penalty=-0.013 progress=0.047 | no_meaningful_improvement |
| 5 | contact_bonus + fuel_penalty + orientation_penalty + progress | -39.29 | 91.82 | -131.10 | 863.40 | contact_bonus=19.989 fuel_penalty=-0.037 orientation_penalty=-0.015 progress=0.016 | no_meaningful_improvement |
| 6 | contact_bonus + fuel_penalty + orientation_penalty + progress | -20.44 | 91.82 | -112.25 | 611.10 | contact_bonus=0.987 fuel_penalty=-0.027 orientation_penalty=-0.034 progress=0.051 | unsolved_high_achievement_continue_from_best |
| 7 | contact_bonus + fuel_penalty + orientation_penalty + progress | -133.27 | 91.82 | -225.08 | 656.00 | contact_bonus=33.531 fuel_penalty=-0.033 orientation_penalty=-0.021 progress=0.025 | no_meaningful_improvement |
| 8 | contact_bonus + fuel_penalty + orientation_penalty + progress | -54.72 | 91.82 | -146.54 | 162.35 | contact_bonus=1.047 fuel_penalty=-0.038 orientation_penalty=-0.017 progress=0.044 | no_meaningful_improvement |
