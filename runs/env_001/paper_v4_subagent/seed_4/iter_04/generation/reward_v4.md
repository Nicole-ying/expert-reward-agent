# 设计理由
## 第 0 步：信号覆盖审计
- **终止模式分析**：全部 episode 以 `terminated` 结束，平均长度 72.75，12/20 为早期终局（<150 步且 score<-50）。推测主因是 crash 或出界——agent 在学会安全着陆前快速失败。
- **观测使用扫描**：所有 8 维观测均已使用，但 crash／出界检测组合过于严苛，使得 `failure_penalty` 的 `active_rate = 0`（僵尸组件）。同时 `stall_penalty` 被激活但很可能将远处低速行为罚至灭绝，迫使 agent 保持高速→加剧 crash，形成恶性循环。
- **信号缺口判断**：并非观测缺失，而是 **失败事件的检测逻辑失准**（校准问题），导致最重要的失败代价信号为 0。同时 `stall_penalty` 在早期探索阶段反向抑制减速，属于结构不合理。
- **僵尸组件**：`failure_penalty`（active_rate 0%）和 `stall_penalty`（虽为 failure_penalty 的一部分，但数学形式有害）。

## 行为诊断
- **agent 在做什么**：快速坠毁或飞出边界，因为失败时几乎没有代价（progress 可能为负，但 failure_penalty=0），agent 未能区分安全与危险动作。
- **干预目标**：恢复有效的失败惩罚信号，使 crash 和出界受到显著负反馈；同时移除有害的 `stall_penalty`，释放减速探索空间。
- **方向是否继续**：累积记录中 iter 3 引入软着陆+停滞惩罚后 len 暴跌，单次预判 ❌。连续 ❌ 未满 3 次，暂不重建骨架，但需要结构修正（修复失败检测 + 删除 stall）。

## Level 2 结构变换
- **删除**：`stall_penalty` 及相关参数（`FAIL_STALL_PER_STEP`, `STALL_DIST_THRESH`, `STALL_VEL_THRESH`）。
- **重写 `failure_penalty` 检测逻辑**：
  - **出界**保留：`abs(x_next) > 1.0` → -10。
  - **crash 重定义**：只要脚接触（`left_contact` 或 `right_contact == 1`），且任一危险信号成立（大倾角 `>0.5`、高冲击速度 `>1.0`、离目标远 `>0.5`），即判为 crash → -10。删除了原来必须同时满足 `close_to_ground` 且 `excessive_tilt/high_impact` 的狭窄组合，确保现实中的坠毁能被捕获。
- **原因**：僵尸化的 failure_penalty 是当前低分的根本原因；stall_penalty 在 agent 学会控制前便惩罚低速，诱导向高速→crash 的正反馈，必须移除。

## 系数校准
- 失败惩罚 `-10` 为一次性事件惩罚，平均到每步（约 73 步）约 `-0.137`，主信号 progress 的 per-step ≈ 0.045，惩罚幅度为主信号的约 3 倍（略高，但失败事件需要足够突出以改变行为）。考虑到早期大部分 episode 皆失败，惩罚占比较大属必要代价，随学习进步会逐步降低。
- 其他组件系数不变，满足约束。

# 代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_cur, y_cur = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ========== 超参数 ==========
    PROGRESS_WEIGHT = 3.0
    FAIL_BOUNDS = -10.0
    FAIL_CRASH = -10.0

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.05
    ACTION_PENALTY = 0.05

    SOFT_LANDING_WEIGHT = 0.5
    SOFT_POS_THRESH = 0.3
    SOFT_VEL_THRESH = 0.5          # 绝对值之和阈值
    SOFT_ANGLE_THRESH = 0.2

    SUCCESS_REWARD = 100.0
    SUCCESS_X_THRESH = 0.1
    SUCCESS_Y_THRESH = 0.1
    SUCCESS_VEL_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1

    X_BOUNDARY = 1.0
    CRASH_ANGLE = 0.5
    CRASH_VEL = 1.0
    CRASH_DIST = 0.5

    # ========== 1. 主进展信号 (improvement_delta) ==========
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ========== 2. 失败惩罚（边界 + 不安全着陆）==========
    # 出界
    out_of_bounds = abs(x_next) > X_BOUNDARY
    boundary_penalty = FAIL_BOUNDS if out_of_bounds else 0.0

    # 坠毁：有脚接触 + 危险状态（倾角/速度/距离任意超标）
    crash = False
    contact = (left_contact == 1.0) or (right_contact == 1.0)
    if contact:
        if (abs(body_angle_next) > CRASH_ANGLE or
            abs(y_vel_next) > CRASH_VEL or
            abs(x_vel_next) > CRASH_VEL or
            dist_next > CRASH_DIST):
            crash = True
    crash_penalty = FAIL_CRASH if crash else 0.0

    failure_penalty = boundary_penalty + crash_penalty

    # ========== 3. 姿态稳定惩罚 ==========
    stability_penalty = (-ANGLE_PENALTY * (body_angle_next ** 2) 
                         - ANG_VEL_PENALTY * (ang_vel_next ** 2))

    # ========== 4. 动作效率惩罚（离散动作） ==========
    action_penalty = 0.0
    if action != 0:  # 非 no_engine 动作消耗燃料
        action_penalty = -ACTION_PENALTY

    # ========== 5. 软着陆塑造 (joint_condition_proxy, 连续) ==========
    pos_factor = max(0.0, 1.0 - dist_next / SOFT_POS_THRESH)
    vel_abs_sum = abs(x_vel_next) + abs(y_vel_next)
    vel_factor = max(0.0, 1.0 - vel_abs_sum / SOFT_VEL_THRESH)
    angle_factor = max(0.0, 1.0 - abs(body_angle_next) / SOFT_ANGLE_THRESH)
    soft_landing = SOFT_LANDING_WEIGHT * pos_factor * vel_factor * angle_factor

    # ========== 6. 成功着陆奖励（稀疏大额） ==========
    landed = (abs(x_next) < SUCCESS_X_THRESH and
              abs(y_next) < SUCCESS_Y_THRESH and
              abs(x_vel_next) < SUCCESS_VEL_THRESH and
              abs(y_vel_next) < SUCCESS_VEL_THRESH and
              abs(body_angle_next) < SUCCESS_ANGLE_THRESH and
              contact)
    success_reward = SUCCESS_REWARD if landed else 0.0

    # ========== 汇总 ==========
    total_reward = (progress + failure_penalty + stability_penalty +
                    action_penalty + soft_landing + success_reward)

    components = {
        'progress': progress,
        'failure_penalty': failure_penalty,
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty,
        'soft_landing': soft_landing,
        'success_reward': success_reward
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: `failure_penalty` 因 crash 检测过于严苛而完全未触发，导致 agent 在大量坠毁/出界时无失败代价，len 暴跌；`stall_penalty` 反向压制减速探索，助推高速坠毁。
- **behavior**: 快速坠毁或出界（平均 72 步），失败集无有效负反馈。
- **signal**: 失败事件惩罚缺失（active_rate=0），应重建检测逻辑；低速抑制信号（stall）对早期学习有害。
- **level**: Level 2
- **hypothesis**: 放宽 crash 条件使坠毁/出界受到 -10 强信号，移除 stall 释放减速空间，agent 将学会规避危险状态，逐步延长 episode，提升成功率，进而拉升总分。
- **risk**: crash 条件放宽可能误判少数正常接触为失败（如距离稍远但安全的支撑脚触地），短期给予不必要惩罚；但随着训练推进，agent 会倾向于在更优状态接触，误判率下降。可通过后续微调阈值缓解。