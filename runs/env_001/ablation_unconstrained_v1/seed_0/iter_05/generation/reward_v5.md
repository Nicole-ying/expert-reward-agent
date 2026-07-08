## 分析

### 1. Agent 发生了什么？

**核心问题：Agent 学会了"悬停采集代理奖励"的局部最优，而非着陆完成任务。**

关键证据：
- **proximity_bonus 占比 88.2%**（147.6/167.4 magnitude），是一条始终激活、几乎不区分好坏的恒定信号。`0.2/(1+new_dist)` 在 d≈0.35 时约 0.148/step，累积 1000 步即 ~148。
- **landing_proxy active_rate = 0.0%**：乘积式中 `contact_avg` 始终为零（无支撑腿接触），导致整个着陆信号从未触发。Agent 没有任何梯度引导它完成着陆。
- **progress_reward 仅 3.5%**（5.89 episode sum）：`5.0 * (old_dist - new_dist)` 系数太小，且 agent 在悬停后几乎不做位移。
- **stability_penalty 仅 -5.8%**：说明 agent 保持极低速度/低角度来规避惩罚，印证了"悬停不动"的行为。
- **全部 20 个 episode 被 truncate**（无一次 terminate）：Agent 从未成功着陆或坠毁，只是在 1000 步中存活并收集 proximity_bonus。
- **score = -19.28，远低于 component 总和 ~140**：说明外部真实评分与当前奖励严重不对齐——shaped reward 大量支付给了无意义的悬停行为。

### 2. 哪个组件最值得干预？

**最值得干预：proximity_bonus（必须重构）和 landing_proxy（必须修复）。**

- **proximity_bonus**：当前作为恒定 per-step 奖励，创建了"存活即收益"的错误激励。正确的做法是将其改为**增量式进度奖励**（potential-based shaping），只在靠近目标时支付、悬停时为零。
- **landing_proxy**：因缺少 floor（`contact_avg` 为零时整项为零）导致 0% 激活率。必须添加类似 best 函数中的 `(0.1 + 0.9 * contact)` 结构，确保无接触时仍有弱梯度引导下降和稳定。

### 3. 上一轮改了什么？

上一轮在 best 基础上做了这些修改：
- 将 `r_dist = 1.0/(1+d)` 拆成 `progress_reward = 5.0*(old_dist-new_dist)` + `proximity_bonus = 0.2/(1+new_dist)`
- 去掉了 landing_proxy 的 floor（从 `(0.1+0.9*contact)` 变成直接 `contact_avg`）
- 新增了 `time_penalty = -0.003`

**效果**：proximity_bonus 退化为悬停采集器，landing_proxy 彻底失效，progress_reward 太弱无法驱动行为。得分从 best（未知但更高）跌至 -19.28。

---

**修改方案**：以 best 结构为基础，做三个有证据的改动：
1. **用强进度信号替代恒定 proximity**：`r_progress = 20.0 * (old_dist - new_dist)`，悬停时为零。
2. **恢复 landing_proxy 的 floor**：`(0.1 + 0.9 * contact)`，确保无接触时仍提供梯度。
3. **移除 time_penalty**：它对行为没有塑造作用，只是恒定噪声。
4. **调整 landing 系数**：从 8.0 提到 10.0，因为去掉了独立 r_dist 后需要更强的终点吸引。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Previous state ---
    old_x = obs[0]
    old_y = obs[1]
    old_dist = (old_x**2 + old_y**2)**0.5

    # --- Next state ---
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    new_dist = (x_pos**2 + y_pos**2)**0.5

    # 1. Progress — strong incentive to reduce distance, zero when hovering
    #    Replaces constant proximity bonus; only pays for actual approach.
    r_progress = 20.0 * (old_dist - new_dist)

    # 2. Stability — gentle penalty for violent motion
    r_stability = -(
        0.01 * (abs(x_vel) + abs(y_vel)) +
        0.10 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )

    # 3. Landing proxy — continuous gradient toward stable touchdown
    #    Floor of 0.1 ensures signal before any leg contact, fixing 0% activation.
    proximity = max(0.0, 1.0 - new_dist / 0.5)
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.4)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)
    contact   = (left_contact + right_contact) / 2.0
    r_landing = 10.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)

    total_reward = r_progress + r_stability + r_landing

    components = {
        "progress_reward": r_progress,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }

    return float(total_reward), components
```