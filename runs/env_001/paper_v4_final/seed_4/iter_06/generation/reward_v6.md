# 设计理由
本轮聚焦**信号覆盖审计**中暴露的僵尸组件 `landing_bonus` (active_rate 0.5%)：虽然部分 episode 触发了一次性 200 奖励，但整体引导太稀疏，agent 在到达目标垫上方后因差一点不满足所有严格条件而得不到奖励，导致分值受限于进度与生存惩罚的微小差额，停滞在 12−14 分区间。  
**干预层级**：Level 2 结构变换。将二值成功 bonus 替换为**连续着陆接近度奖励**，利用距离、速度、角度、脚接触四个维度构建门控乘积 → 几何平均因子，使 agent 在逐渐接近完美着陆的过程中每一步都能获得递增的激励，从而驱动“最终接触”完成着陆，而非在垫上空徘徊。  
**数学形式**：
- 四个因子均归一化到 [0,1]，采用 1−min(1, signal/threshold) 线性缩放，阈值设置在任务终止边界的 60‑80%：
  - `dist_factor`: 阈值 0.5（目标坐标原点附近区域）
  - `speed_factor`: 阈值 0.5
  - `angle_factor`: 阈值 0.3 rad
  - `contact_factor`: 0.5 + 0.5·max(左接触, 右接触) （无接触时 0.5，避免完全塌缩）
- 组合使用几何平均 `(dist * speed * angle) ** (1/3) * contact` 防止乘积塌缩为 0。
- 系数 `LANDING_PROX_WEIGHT = 2.0`，最大 per‑step 奖励 2.0（当所有因子=1 时），约为当前主信号 per‑step (0.202) 的 10 倍，但仅发生在着陆前极少数步；平均激励远低于此。总惩罚负担 (≈‑0.101) 被安全包容。
**系数校准**：阈值基于“接近目标但不完全着陆”的安全区域设定，确保 gate 在“不太完美但安全”状态保持 ≥0.3，避免塌缩。

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
    FAIL_PENALTY = -30.0               # 出界 / 坠毁 / 远距离静止 一次性惩罚

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.03

    ACTION_FUEL_PENALTY = -0.01

    # 连续着陆接近度奖励参数
    LANDING_PROX_WEIGHT = 2.0           # 最大 per-step 奖励
    DIST_THRESH = 0.5                   # 距离阈值
    SPEED_THRESH = 0.5                  # 速度阈值
    ANGLE_THRESH = 0.3                  # 角度阈值

    # 出界/坠毁阈值（保留原有失败检测）
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

    # ---- 5. 连续着陆接近度奖励 (替代稀疏 landing_bonus) ----
    # 距离因子：越接近原点越大，超出阈值 0
    dist_factor = 1.0 - min(1.0, dist_next / DIST_THRESH)
    # 速度因子
    speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5
    speed_factor = 1.0 - min(1.0, speed / SPEED_THRESH)
    # 倾角因子
    angle_factor = 1.0 - min(1.0, abs(body_angle_next) / ANGLE_THRESH)
    # 接触因子：有脚接触为 1，无接触为 0.5（避免塌缩为 0）
    contact_factor = 0.5 + 0.5 * max(left_contact, right_contact)

    # 几何平均防止任一因子 0 导致整体 0
    prox = (dist_factor * speed_factor * angle_factor) ** (1.0 / 3.0)
    landing_prox = LANDING_PROX_WEIGHT * prox * contact_factor

    # ---- 6. 终止事件推断（保留失败检测） ----
    out_of_bounds = abs(x_next) > X_BOUNDARY

    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    # 远距离静止失败（无脚接触，贴地，远离目标）
    distant_stall = (
        (y_next < GROUND_Y_CLOSE) and
        (dist_next > DIST_THRESH) and        # 使用与着陆奖励相同的阈值保持一致
        speed < SPEED_THRESH and
        abs(body_angle_next) < ANGLE_THRESH and
        (not (left_contact > 0.5 or right_contact > 0.5))
    )

    failure_penalty = FAIL_PENALTY if (out_of_bounds or crash or distant_stall) else 0.0

    # 合并奖励
    total_reward = (progress + survival + stability + fuel +
                    landing_prox + failure_penalty)

    components = {
        'progress': progress,
        'survival': survival,
        'stability': stability,
        'fuel': fuel,
        'landing_prox': landing_prox,
        'failure_penalty': failure_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测均被使用，无信号缺口；僵尸组件 `landing_bonus` (0.5%) 缺乏引导，agent 因稀疏奖励难以学习最终着陆动作。
- **behavior**: agent 在 127 步左右终止，接近目标但常因缺少脚接触或速度/角度略超阈值而得不到奖励，总分被生存惩罚压制。
- **signal**: 缺少平滑的着陆接近度信号，现有成功检测过于二值且严格，导致 landing 奖励几乎休眠。
- **level**: Level 2（结构变换：稀疏 bonus → 连续门控乘积 + 几何平均）
- **hypothesis**: 连续着陆激励会渐进强化接近完美着陆的所有子目标，驱动 agent 在最后阶段主动减速、调平、触地，显著提升成功率与得分。
- **risk**: 连续奖励可能诱发 agent 在着陆区边缘小幅振荡“刷分”，但速度与距离的同时约束使这种振荡不可持续；需监控 len 是否异常上涨。