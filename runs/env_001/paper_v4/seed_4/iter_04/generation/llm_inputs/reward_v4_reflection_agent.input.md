# ⚠️ 上一版代码验证失败
错误信息：Reward v4 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs\env_001\paper_v4\seed_4\iter_04\generation\validations\reward_v4.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="get_skeleton_detail">
<｜｜DSML｜｜parameter name="skeleton_name" string="true">state_to_improvement</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>

# Search objective
- target_score: 200.000000
- current_score: 206.135560
- gap_to_target: -6.135560
- target_achievement_ratio: 103.068%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 206.135560）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # ------------------ 1. Progress reward ------------------
    w_progress = 2.0
    progress = dist - next_dist
    progress_reward = w_progress * progress

    # ------------------ 2. Stability penalty (gated) ------------------
    gate = 1.0 / (1.0 + dist)
    w_vel = 0.2
    w_ang = 0.2
    stability_error = w_vel * (vx ** 2 + vy ** 2) + w_ang * (angle ** 2 + angvel ** 2)
    stability_penalty = -gate * stability_error

    # ------------------ 3. Fuel penalty ------------------
    w_fuel = 0.02
    fuel_penalty = -w_fuel if action != 0 else 0.0

    # ------------------ 4. Safe contact reward (sustained quality-gated) ------------------
    w_contact = 1.0
    contact_now = next_left * next_right                              # 1.0 only if both legs touch
    dist_factor = 1.0 / (1.0 + 10.0 * next_dist)                      # near center
    vel_factor = 1.0 / (1.0 + 5.0 * (next_vx ** 2 + next_vy ** 2))   # low velocity gating
    angle_factor = 1.0 / (1.0 + 5.0 * (abs(next_angle) + abs(next_angvel)))  # upright + still
    safe_contact_reward = w_contact * contact_now * dist_factor * vel_factor * angle_factor

    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_reward

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_reward": safe_contact_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=206.135560, len=681.450000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[127.443360, 288.929483]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 294.208533 | 94.6% | 94.6% | 51.4% |
| fuel_penalty | -9.771000 | -3.1% | 3.1% | 71.7% |
| stability_penalty | -4.337368 | -1.4% | 1.4% | 100.0% |
| progress_reward | 2.768706 | 0.9% | 0.9% | 99.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 载具式着陆任务。主体从视口上方中央附近开始，带有随机初速扰动。**核心目标**是在尽可能短的时间内安全抵达画面中央的目标着陆垫并稳定停驻，同时尽可能减少引擎推力消耗。智能体必须学会精确控制位置与速度，在接近着陆垫时减速、保持竖直姿态，并实现两条支撑腿的平稳接触。

**次要目标**：节约引擎燃料；快速完成任务。  
**不应混淆的目标**：不存在与到达目标同等权重的冲突目标（如“保持高速”或“探索未知区域”），燃料节省仅为附属要求。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（来自 Box 观察）
- 各维度含义：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | x_position | 相对于目标垫的水平坐标 | true |
| 1 | y_position | 相对于目标垫高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 主体姿态角（绝对值或从竖直偏移） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑腿是否触地（1.0 或 0.0） | true（谨慎） |
| 7 | right_support_contact | 右支撑腿是否触地（1.0 或 0.0） | true（谨慎） |

**注意**：左/右支撑接触标志可用于奖励，但需考虑它们可能与导致终止的“crash_or_body_contact”混淆风险（见下文分析）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 各动作含义：

| 动作编号 | 名称 | 含义 |
|----------|------|------|
| 0 | no_engine | 不点火，仅靠惯性漂移 |
| 1 | left_orientation_engine | 点燃左侧姿态调节引擎，产生一个方向扭矩/推力 |
| 2 | main_engine | 点燃主引擎（推测向上推力，对抗重力） |
| 3 | right_orientation_engine | 点燃右侧姿态调节引擎，产生相反方向扭矩/推力 |

动作空间为离散选择，每次步只能执行四个动作之一。无连续控制。

## 5. step 与终止条件分析
### 5.1 终止模式
源码中的终止判断为：
```python
terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled
```
其中 **crash_or_body_contact**、**horizontal_position_outside_viewport** 和 **body_not_awake_or_settled** 均为复合条件（具体实现未暴露）。

- **success-like termination**：`body_not_awake_or_settled` 可能表示主体已安定在着陆垫上（无运动或规定时间内稳定），此类终止可视为成功。
- **failure-like termination**：`crash_or_body_contact`（如翻滚、身体直接碰撞）与 `horizontal_position_outside_viewport`（飞出边界）明显是失败。
- **ambiguous termination**：`body_not_awake_or_settled` 若发生在着陆垫之外，可能不算成功，但由于无额外信息，无法在 step 返回值中分辨。环境没有提供显式的“成功 / 失败”标志。
- **truncation**：源码中 `step` 未返回 `truncated`（第四个返回值为 `False`），因此无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: `info` 固定为空字典 `{}`，无任何可用字段。
- forbidden_or_uncertain_info_fields: 不可假设存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等。`info` 完全不可用。

## 7. 可用于奖励函数的信号
基于观察和动作的可用信号分类如下：

- **位置**：`x_position`（0）、`y_position`（1）—— 相对目标垫坐标，目标为 (0, 0)。
- **速度**：`x_velocity`（2）、`y_velocity`（3）—— 目标为零。
- **方向/姿态**：`body_angle`（4）、`angular_velocity`（5）—— 理想为零（竖直稳定）。
- **接触/着陆**：`left_support_contact`（6）、`right_support_contact`（7）—— 标志双腿是否触地，可用于安全着陆奖励，但需注意避免与 crash 混淆（见下节）。
- **动作/引擎**：采取的动作 `action`（0~3）用于推力惩罚（如约束不点火为主，或限制除主引擎外的浪费）。可设计动作惩罚。