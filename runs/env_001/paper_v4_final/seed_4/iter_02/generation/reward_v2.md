# 设计理由

## 为什么以前都失败了
上一轮尝试采用了 `landing_proxy` 作为主信号，它通过距离、速度、姿态、接触四个因子的算术平均构造了一个“软着陆代理奖励”。训练数据表明：
- `landing_proxy` 的逐回合累计平均值高达 865.37，占总奖励量级的 99.7%。
- 而真正的进展信号 `progress`（距离缩减量）的累计值仅为 -0.67——agent 完全没有向目标垫靠近，甚至在远离。
- 所有惩罚组件几乎不触发（failure 为 0，stability 极小）。
- 最终评价分数为 -166.96，19/20 集因超时而截断，1/20 终止（可能是偶然的 crash 或成功，但占比极低）。

**结论**：`landing_proxy` 提供了一个极易被 exploit 的伪目标——agent 可以通过姿态保持、低速徘徊、不接触地面的静态存活策略，持续拿到高额奖励，却完全背离了“快速到达并降落在目标垫”的真实任务目标。这是一个典型的 **reward hacking** 案例。

## 新骨架选了哪些算子，和已有尝试的本质不同
本骨架从零重建，核心改变是：
1. **用 `improvement_delta` 替换被 exploit 的静态 proxy 作为主驱动信号**  
   不再奖励“处于某种看似像着陆的状态”，而是直接奖励“每一步向目标垫靠近的距离缩减量”（`dist_cur - dist_next`）。这个信号无法通过停滞获得——必须持续改善位置。
2. **引入每步固定的生存惩罚**  
   加入一个微小但持续存在的负奖励，迫使 agent 不能无限期苟活；它必须尽快完成着陆以终止惩罚累积。这直接对抗上一轮的“超时存活”策略。
3. **在 episode 终止点引入推断的“成功着陆”大额奖励**  
   利用观测信号（位置近、速度小、姿态正、有脚接触）在 `terminated=True` 且状态满足条件时，给予一次性高额正奖励。因为绝大多数 episode 目前是截断而非终止，这个信号只在真正着陆成功时出现，不会像连续 proxy 那样被刷。
4. **保留轻量的姿态/角速度惩罚和失败惩罚**  
   仅作为辅助约束，系数远小于主信号，避免喧宾夺主。
5. **删除所有 `landing_proxy` 组件**  
   其四个因子已通过终端成功条件中的门槛形式重新表达，不再提供可刷的连续奖励。

这种从“静态状态奖励”到“动态改善奖励 + 终端成功激励 + 存在惩罚”的范式转换，是本骨架与上一轮的本质区别。

---

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

    # ---- 超参数 ----
    PROGRESS_WEIGHT = 20.0                # 距离缩减奖励的主权重
    SURVIVAL_PENALTY = -0.08              # 每步生存惩罚，激励快速完成
    SUCCESS_BONUS = 200.0                 # 成功着陆的终端奖励
    FAIL_PENALTY = -30.0                  # 坠毁/出界的终端惩罚

    ANGLE_PENALTY = 0.3                   # 身体倾角二次惩罚系数
    ANG_VEL_PENALTY = 0.03                # 角速度二次惩罚系数

    ACTION_FUEL_PENALTY = -0.01           # 引擎动作的燃料惩罚 (action != 0)

    # 推断成功的阈值
    LAND_DIST_THRESHOLD = 0.2             # 距离垫原点欧氏距离 < 0.2
    LAND_SPEED_THRESHOLD = 0.2            # 合速度 < 0.2
    LAND_ANGLE_THRESHOLD = 0.15           # 身体倾角 < 0.15 rad (~8.6°)
    LAND_CONTACT_REQUIRED = True          # 至少一脚接触垫

    # 推断失败的阈值
    X_BOUNDARY = 1.0                      # 水平飞出边界
    # 坠毁条件：有支撑脚接触 + 地面很近 + (倾角过大 或 高速撞击)
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8                     # ~45°
    CRASH_IMPACT_VEL = 1.5

    # ---- 1. 进展信号：基于距离的 improvement_delta ----
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)   # 期望 >0

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 终端事件推断（在 episode 最后一步时生效） ----
    # 环境在 terminated=True 的最后一帧也会调用 reward，因此可以在 next_obs 中判断
    # 成功着陆条件（严格）
    dist_to_pad = dist_next
    speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5
    angle_ok = abs(body_angle_next) < LAND_ANGLE_THRESHOLD
    contact_ok = (left_contact > 0.5) or (right_contact > 0.5)
    success = (dist_to_pad < LAND_DIST_THRESHOLD and
               speed < LAND_SPEED_THRESHOLD and
               angle_ok and
               contact_ok)

    # 失败条件
    out_of_bounds = abs(x_next) > X_BOUNDARY
    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    failure = out_of_bounds or crash

    # 只在 episode 真正终止时才给予终端奖励/惩罚，避免截断时误判
    # 通过 info 中的 'terminated' 字段判断（若可用），否则保守地只在满足明确条件时给
    # 如果无法获取 terminated 信息，我们采用保守策略：同时满足成功条件就给 bonus，
    # 但 crash/out_of_bounds 在每步都可能触发（但通常在这些事件发生时会立即终止）。
    # 为了安全，我们依赖条件本身；crash 和 out_of_bounds 只会在边界步出现，所以直接给惩罚。
    landing_bonus = SUCCESS_BONUS if success else 0.0
    failure_penalty = FAIL_PENALTY if failure else 0.0

    # 注意：同一帧不可能既成功又失败，因为条件互斥（成功需要小距离，失败是出界或 crash）
    total_reward = (progress + survival + stability + fuel +
                    landing_bonus + failure_penalty)

    components = {
        'progress': progress,
        'survival': survival,
        'stability': stability,
        'fuel': fuel,
        'landing_bonus': landing_bonus,
        'failure_penalty': failure_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 上一轮 landing_proxy 被 exploit，提供了与真实目标无关的高额持续性奖励，导致 agent 存活但不靠近目标。新骨架完全移除连续 proxy，重建驱动范式。
- **behavior**: agent 学会了静态存活（姿态水平、低速、不接触地面），在 1000 步内不触发任何终止，仅靠 landing_proxy 累积大量正奖，环境评分极低。
- **signal**: 原本缺少有效的进展约束（progress 被掩盖）和成功激励（无终端成功检测），导致 reward hacking。新设计用 delta-distance 作为主进展，配合生存惩罚和终端成功奖励，建立正向循环。
- **level**: Level 3（重建骨架）
- **hypothesis**: 通过 delta-distance + survival penalty + terminal success bonus 的组合，agent 将被迫移动并优化向目标垫的接近，成功着陆的大额奖励将引导它学习稳定降落，从而消除超时停滞。
- **risk**: 生存惩罚可能使 agent 在早期探索阶段因无法抵消惩罚而倾向于快速 crash（避免累积惩罚），但 crash 惩罚 -30 强于 100 步生存惩罚（-8），且 progress 权重高，应能抵消。如果出现异常多 crash，需微调 survival/failure_penalty 比例。