# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x, y, vx, vy, angle, ang_vel, left, right = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft, nright = next_obs

    # 常用量
    dist = (x**2 + y**2)**0.5
    speed = (vx**2 + vy**2)**0.5
    next_dist = (nx**2 + ny**2)**0.5
    next_speed = (nvx**2 + nvy**2)**0.5

    # ---- 势能塑造：鼓励靠近原点、减小速度 ----
    w_dist = 5.0
    w_speed = 20.0          # 大幅提高速度惩罚权重，防止高速撞击
    potential_cur = -(w_dist * dist + w_speed * speed)
    potential_next = -(w_dist * next_dist + w_speed * next_speed)
    shaping = potential_next - potential_cur

    # ---- 姿态与角速度惩罚 ----
    angle_penalty = -0.5 * abs(nangle)        # 鼓励竖直
    angvel_penalty = -0.05 * abs(nang_vel)    # 抑制旋转

    # ---- 燃料效率惩罚 ----
    if action == 2:           # 主引擎
        fuel_penalty = -0.15
    elif action in (1, 3):    # 姿态引擎
        fuel_penalty = -0.02
    else:                     # 无推力
        fuel_penalty = 0.0

    # ---- 下降过快惩罚（负的 vy 即为向下） ----
    descending_penalty = -0.5 * max(0.0, -nvy)

    # ---- 存活时间惩罚，推动尽快完成任务 ----
    step_penalty = -0.05

    # ---- 接近着陆：距离近且速度慢时给与小额奖励 ----
    approach_bonus = 0.0
    if next_dist < 0.5 and next_speed < 1.0:
        approach_bonus = 0.5

    # ---- 成功着陆大奖励 ----
    contact_success = (nleft and nright and
                       (next_speed < 0.5) and
                       (abs(nx) < 0.3) and
                       (abs(ny) < 0.3) and
                       (abs(nangle) < 0.1))
    success_bonus = 50.0 if contact_success else 0.0

    # ---- 猛烈着陆惩罚 ----
    crash_condition = nleft and nright and ((next_speed > 1.5) or (abs(nangle) > 0.3))
    crash_penalty = -10.0 if crash_condition else 0.0

    # ---- 双腿接触持续奖励（鼓励稳定接触） ----
    contact_continuous = 0.0
    if nleft and nright:
        speed_factor = max(0.0, 1.0 - next_speed / 2.0)      # 速度越慢，系数越高
        angle_factor = max(0.0, 1.0 - abs(nangle) / 0.5)     # 角度越小，系数越高
        contact_continuous = 0.2 * speed_factor * angle_factor

    # 汇总
    total_reward = (shaping +
                    angle_penalty +
                    angvel_penalty +
                    fuel_penalty +
                    descending_penalty +
                    step_penalty +
                    approach_bonus +
                    success_bonus +
                    crash_penalty +
                    contact_continuous)

    components = {
        "shaping": shaping,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "fuel_penalty": fuel_penalty,
        "descending_penalty": descending_penalty,
        "step_penalty": step_penalty,
        "approach_bonus": approach_bonus,
        "success_bonus": success_bonus,
        "crash_penalty": crash_penalty,
        "contact_continuous": contact_continuous
    }

    return float(total_reward), components
```
