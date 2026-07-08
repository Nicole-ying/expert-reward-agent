# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------------------------------------------------------
    # 1. 势能函数 Phi(s) —— 越小表示越接近成功着陆状态
    #    Phi(s) = - ( w_dist * d + w_vel * v + w_angle * |angle| + w_angvel * |angvel| )
    #    d = sqrt(x^2 + y^2) , v = sqrt(vx^2 + vy^2)
    # ------------------------------------------------------------------
    px, py = float(obs[0]), float(obs[1])
    vx, vy = float(obs[2]), float(obs[3])
    angle = float(obs[4])
    angvel = float(obs[5])

    dist_now = (px * px + py * py) ** 0.5
    speed_now = (vx * vx + vy * vy) ** 0.5
    phi_now = -(
        2.0 * dist_now
        + 0.5 * speed_now
        + 0.2 * abs(angle)
        + 0.1 * abs(angvel)
    )

    npx, npy = float(next_obs[0]), float(next_obs[1])
    nvx, nvy = float(next_obs[2]), float(next_obs[3])
    nangle = float(next_obs[4])
    nangvel = float(next_obs[5])

    dist_next = (npx * npx + npy * npy) ** 0.5
    speed_next = (nvx * nvx + nvy * nvy) ** 0.5
    phi_next = -(
        2.0 * dist_next
        + 0.5 * speed_next
        + 0.2 * abs(nangle)
        + 0.1 * abs(nangvel)
    )

    # 势能塑形奖励（使用 gamma=1.0，符合无折扣任务）
    r_shaping = phi_next - phi_now

    # ------------------------------------------------------------------
    # 2. 成功状态奖励 —— 已经稳定着陆在目标垫上则每步给予正奖励
    #    条件：距离近、速度小、双足接触、姿态正
    # ------------------------------------------------------------------
    left_contact = float(next_obs[6])
    right_contact = float(next_obs[7])
    if (dist_next < 0.2 and speed_next < 0.2
            and left_contact > 0.5 and right_contact > 0.5
            and abs(nangle) < 0.2):
        r_success_state = 1.0
    else:
        r_success_state = 0.0

    # ------------------------------------------------------------------
    # 3. 失败惩罚 —— 机体越界或严重坠毁（阈值宽松，只捕获明显过错）
    # ------------------------------------------------------------------
    if abs(npx
