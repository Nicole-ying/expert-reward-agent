# 设计理由
当前 `grounded_quality` 是每步持续奖励（接触且低速时发放），均值高达 28.08，且 episode 平均长度 290 步，terminated 率 19/20。推断 agent 在接触着陆垫后仍可能低速徘徊以累积该奖励，拖延了最终 settle。为消除这一拖延动机并鼓励快速且低速的着陆，将 `grounded_quality` 替换为一次性的 `touchdown_success` 事件奖励：仅在脚从未接触到接触的转移时刻发放，并乘以速度因子（`1 - speed/safe_speed`，限制在 [0,1]），从而同时要求低速触地。系数 10.0 与原来 grounded_quality 的总贡献（~28）量级可比，事件奖励不会鼓励持续停留。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    dx_curr, dy_curr = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_curr = (dx_curr**2 + dy_curr**2) ** 0.5
    dist_next = (dx_next**2 + dy_next**2) ** 0.5

    approach_delta = dist_curr - dist_next

    vx_next, vy_next = next_obs[2], next_obs[3]
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    safe_speed = 0.2

    angle_next = next_obs[4]
    angvel_next = next_obs[5]

    left_contact_curr = obs[6] > 0.5
    right_contact_curr = obs[7] > 0.5
    left_contact_next = next_obs[6] > 0.5
    right_contact_next = next_obs[7] > 0.5

    w_approach = 5.0
    w_vel_penalty = 0.5
    w_angle = 0.5
    w_angvel = 0.1
    w_landing = 0.1
    w_touchdown = 10.0

    approach_reward = w_approach * approach_delta

    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    landing_reward = w_landing * dist_factor * speed_factor

    prev_contact = left_contact_curr or right_contact_curr
    cur_contact = left_contact_next or right_contact_next
    touchdown_reward = 0.0
    if not prev_contact and cur_contact:
        speed_ratio = max(0.0, 1.0 - speed_next / safe_speed)
        touchdown_reward = w_touchdown * speed_ratio

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward + touchdown_reward

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
        "touchdown_success": touchdown_reward,
    }
    return float(total_reward), components
```

# 诊断摘要
- **evidence**: terminated 19/20，episode 平均长度 290，grounded_quality 均值 28.08（持续累积），velocity_penalty -18.88。
- **behavior**: 安全着陆但可能在脚接触垫子后低速徘徊以刷取 grounded_quality，导致 episode 较长。
- **signal**: 缺少鼓励“快速最终着陆并立刻 settle”的信号，持续奖励反而激励拖延。
- **level**: Level 2（持续事件被重复领取 → 转移事件）。
- **hypothesis**: 改为一次性 touchdown 事件奖励（附带低速要求）可消除徘徊动机，促使 agent 更快触地并立即进入 settle 状态。
- **risk**: 若 touchdown 奖励过高且速度因子掉落太快，可能鼓励稍高速度撞击；但 velocity_penalty 和速度因子联合约束应能避免。