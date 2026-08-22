# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取关键观测
    hull_angle = next_obs[0]          # body pitch angle
    horizontal_speed = next_obs[2]   # forward (assumed positive) velocity
    vertical_speed = next_obs[3]     # vertical velocity, bouncing indicator

    # 主进展信号：水平速度，鼓励持续前进（每步有梯度）
    progress_raw = horizontal_speed

    # 身体稳定性门控：倾角越接近安全上限（0.6 rad），门控值越接近0
    angle_threshold = 0.6
    angle_gate = max(0.0, 1.0 - abs(hull_angle) / angle_threshold)

    # 垂直弹跳门控：垂直速度越小越好，门控线性衰减
    vert_threshold = 2.0
    vertical_gate = max(0.0, 1.0 - abs(vertical_speed) / vert_threshold)

    # 综合健康门控，乘积作用于主奖励，避免硬惩罚
    health_gate = angle_gate * vertical_gate

    # 总奖励：在身体状态良好时充分奖励前进，恶化时自动衰减激励
    total_reward = progress_raw * health_gate

    # 组件记录，便于调试
    components = {
        'progress_raw': progress_raw,
        'angle_gate': angle_gate,
        'vertical_gate': vertical_gate,
        'health_gate': health_gate,
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像与动力学子类型
- **task_family**: `locomotion_continuous_control`
- **dynamics_subtype**: `planar_bipedal_gait`（双足步行，髋+膝关节力矩控制）
- **主目标**: 持续稳定地向前推进，最大化水平速度积分（前进距离），同时抑制摔倒。

## 选中的奖励职责（reward roles）与信号映射
| role | signal | 公式算子 |
|---|---|---|
| 主前进信号 | `horizontal_speed` (next_obs[2]) | `dense_state_signal` 线性正奖励 |
| 身体稳定性约束 | `hull_angle` (next_obs[0]) | `soft_health_gate` 线性衰减门控 |
| 弹跳/垂直稳定约束 | `vertical_speed` (next_obs[3]) | `soft_health_gate` 线性衰减门控 |

## 排除的职责与原因
- **terminal_success_reward**: 环境无显式 success flag，info 为空，且无法在步级别获知 episode 终止，因此不适用。
- **terminal_failure_penalty**: 同样因为无法可靠识别终止步，不能施加硬覆盖惩罚。
- **action_efficiency_penalty**: v1 阶段优先学习前进方向，动作代价容易压制探索；后续迭代再加入。
- **关节限位/平滑性约束**: 动作维度较少（4），且关节角度/速度未表现出明显抖动风险。
- **交替接触奖励**: 属于步态模式优化，属于二级目标，v1 暂不包括。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
`explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 false，且 `info` 始终为空。奖励函数无法感知 episode 是否终止，不能可靠施加终端事件奖励/惩罚。因此完全采用稠密信号 + 衰减门控的策略。

## 与之前失败尝试的结构性区别
此前尝试使用了正向进度奖励 + 多个二次惩罚项（稳定性、角速度、动作效率），导致累积奖励为较大负值（-18 至 -50），agent 难以获得正向梯度。本方案完全移除独立惩罚项，改为 **progress_raw × soft_health_gate** 结构：
- 没有任何可能制造负奖励的惩罚项（除非水平速度本身为负，此时鼓励避免倒退）。
- 健康门控在危险时只是衰减主奖励，不引入额外负数，确保总奖励始终更接近零或正。
- 门控采用线性衰减（`1 - value/threshold`），在安全区间内变化平滑、梯度连续，且阈值设置在安全边界附近（倾角 0.6 rad、垂直速度 2.0 m/s），避免过早严厉压制。

## 哪些职责留到后续迭代
- 能量效率（关节力矩二次惩罚）
- 步态交替接触/落地模式奖励
- 如果观测出现翻滚倾向，可增加 `hull_angular_velocity` 门控
- 复杂地形预判（li‑dar 信息的使用），可结合 preview_conditioned_reward 操作算子

## 训练后应该观察的 failure modes 与预期行为
- **gate 过于激进**：如果 angle_threshold 或 vert_threshold 设置过小，可能导致 agent 在大多数步中 gate≈0，总奖励极低，停滞不动。
- **stand‑still 逃避**：若 agent 发现站立不动（reward=0）比冒险前进更安全（偶尔摔倒回报低于 0），可能会学习静止策略。但本任务地形不规则且无生存奖励，水平速度为 0 的步累计得分很低，通常仍需前进以获取总分。
- **速度刷分但摔倒风险高**：若 gate 阈值过高，agent 可能学会快速前进但不时摔倒，但仍能得到正分数。可通过降低阈值或增加额外倾角惩罚在后续迭代中处理。
- **倒退行为**：`horizontal_speed` 为负时奖励为负，自然惩罚倒退；但若 agent 学会先倒退再前进以获取更多总步长？不可能，因为 episode 在摔倒或终点后终止，倒退只会浪费步数，降低总分。正常学习应抑制倒退。
