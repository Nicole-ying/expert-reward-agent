# Response Record

# 设计理由

**正常模式，修改一个组件：`failure_penalty`（失败检测）**  
**问题**：当前 `distant_stall`（远距离静止失败）的检测条件要求 `y_next < GROUND_Y_CLOSE`（即必须贴近地面）。但从训练反馈看，全部 episode 均为 `terminated`，而 `failure_penalty` 的 `active_rate=0%`，说明 agent 学会了在任意高度（而不是仅贴近地面）减速到完全静止，触发环境内部的 `body_not_awake_or_settled` 终止，却未触发我们的失败惩罚。这使得 agent 可以在远离目标垫的地方“安全”结束 episode，躲避 `survival` 惩罚和 `failure_penalty`，只收获少量 `progress`，导致总分偏低（‑11.6）。  
**修改**：移除 `distant_stall` 的高度条件，使其在任何高度下，只要远离目标垫、速度接近零、姿态平稳、无脚部接触，就判定为失败。同时将速度阈值收紧到 `0.1`（几乎完全静止），避免在正常飞行中误判。一次性惩罚从 ‑30 降低到 ‑20，以符合主信号校准（主信号 per‑step ≈ 0.28，80 步正收益约 22，‑20 占 90% 略高但仍在可以接受的惩罚负担内）。  
**预期**：agent 将无法再通过任意位置静止来结束 episode，被迫持续朝目标垫移动，最终触地着陆（触发 `landing_success_bonus`）或真正坠毁。这将推动 `landing_success_bonus` 的有效性，最终提升得分。  
**等级**：Level 2（结构变换 — 修改失败条件，让僵尸组件活起来）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- 超参数 ----
    PROGRESS_WEIGHT = 20.0
    SURVIVAL_PENALTY = -0.08
    FAIL_PENALTY = -20.0               # 出界 / 坠毁 / 远距离静止 一次性惩罚（降低一点）

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.03

    ACTION_FUEL_PENALTY = -0.01

    # 成功着陆检测参数（继续收紧以杜绝 exploit）
    SUCCESS_DIST_THRESH = 0.15
    SUCCESS_SPEED_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1
    LANDING_SUCCESS_BONUS = 150.0

    # 出界/坠毁阈值
    X_BOUNDARY = 1.0
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8
    CRASH_IMPACT_VEL = 1.5

    # ---- 1. 进展信号 ----
    x_cur, y_cur = obs[0], obs[1]
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 一次性成功着陆奖励 ----
    prev_dist = dist_cur
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    prev_angle = abs(obs[4])
    prev_contact = max(obs[6], obs[7]) > 0.5

    cur_speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5
    cur_angle = abs(body_angle_next)
    cur_contact = max(left_contact, right_contact) > 0.5

    prev_success = (prev_dist < SUCCESS_DIST_THRESH and prev_speed < SUCCESS_SPEED_THRESH and
                    prev_angle < SUCCESS_ANGLE_THRESH and prev_contact)
    cur_success = (dist_next < SUCCESS_DIST_THRESH and cur_speed < SUCCESS_SPEED_THRESH and
                   cur_angle < SUCCESS_ANGLE_THRESH and cur_contact)

    landing_success_bonus = LANDING_SUCCESS_BONUS if (cur_success and not prev_success) else 0.0

    # ---- 6. 终止事件推断（失败检测） ----
    out_of_bounds = abs(x_next) > X_BOUNDARY

    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    # 【修改】远距离静止失败：去除高度限制，任何位置的“停滞”都判为失败，避免 agent 在空中静止结束 episode。
    distant_stall = (
        (dist_next > SUCCESS_DIST_THRESH) and   # 远离目标垫
        cur_speed < 0.1 and                     # 几乎完全静止
        cur_angle < SUCCESS_ANGLE_THRESH and    # 姿态稳定
        (not cur_contact)                       # 无脚部接触
    )

    failure_penalty = FAIL_PENALTY if (out_of_bounds or crash or distant_stall) else 0.0

    # 合并奖励
    total_reward = (progress + survival + stability + fuel +
                    landing_success_bonus + failure_penalty)

    components = {
        'progress': progress,
        'survival': survival,
        'stability': stability,
        'fuel': fuel,
        'landing_success_bonus': landing_success_bonus,
        'failure_penalty': failure_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: `failure_penalty` active_rate 0%，因为 distant_stall 条件要求低空才生效，agent 学会在空中静止终止，白嫖了无惩罚结束。
- **behavior**: agent 在远离目标垫位置减速至完全静止，由环境内部判断为 `settled` 并终止 episode，获得少量 progress 和负生存惩罚。
- **signal**: 缺少对任意高度远距离静止的明确失败信号，导致目标垫避让。
- **level**: Level 2（结构变换 — 改造 failure 检测条件，使停滞失败信号重新激活）
- **hypothesis**: 移除高度限制后，agent 无法在任何地方安全“抛锚”，必须继续向目标垫推进，最终触发成功着陆或真正失败，整体得分将回归 iter7 的水平。
- **risk**: 若 `cur_speed < 0.1` 的阈值在正常慢速飞行时偶尔触发（例如在强风中短暂悬停），可能误判失败并立即惩罚，但速度阈值已很保守；若仍有误触发，可在后续轮次微调为 `0.05` 或增加连续步数检查。
