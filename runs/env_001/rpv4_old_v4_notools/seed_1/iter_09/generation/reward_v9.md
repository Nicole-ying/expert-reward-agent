### 1. 诊断报告

**evidence**：
- final score = -24.39，episode 长度 919.6，truncated 16/20，terminated 仅 4/20。agent 几乎做不到成功着陆，大量 episode 因超时截断。
- 组件分析：**touchdown_reward** 均值为 51.1，但 active_rate 仅 2.1%，magnitude_share 却高达 59.8%。结合 checkpoint 趋势（cp20%=272, cp70%=628, cp100%=154），说明 agent 偶尔获得高额触地奖励，但极不稳定，极可能是在反复“碰地—弹起—盘旋—再碰地”的 exploit 循环。  
- **landing_quality_reward** 在最终评估中完全为 0（active_rate 0.0%），说明它的触发条件过严，从未被 policy 满足，无法提供任何有效引导。  
- **proximity_progress** 持续激活，但 progress 本身只管距离缩小，无法制止上述触碰式 exploit。  
- 整体趋势：score 在训练中剧烈波动（cp30%= -70, cp70%= 59, cp90%= -156），表明奖励信号存在严重的双重吸引（exploit vs 真正成功），学习不稳定。

**behavior_diagnosis**：
当前策略学会通过偶尔触地获得巨大单步奖励来对冲持续的 penalty，但无法稳定着陆。它更倾向于漂浮并等待触地机会，而不是精确定点和减速。这是一种不健康的 exploit 行为，由 **可重复获取的高额触地奖励** 驱动。最终导致大量 truncation 而非成功。

**signal_completeness**：
缺失：
- **终端成功信号**：没有明确的“任务完成”一次性奖励。agent 无法区分“偶然碰地”和“真正成功着陆”。
- **防止重复触地的机制**：当前触地奖励为每次腿接触均按质量给 reward，鼓励 agent 反复制造接触。
- **着陆过程中的温和正反馈**：landing_quality 过严，完全失效；缺少一个能在稳定接触期间持续给予合理奖励的组件。

**selected_level**：L2  
理由：必须改变 touchdown 的数学形态（从“持续质量加成”转为“首次触地事件 + 稳定接触温和奖励”），并引入成功 bonus。单纯的系数微调（L1）无法消除 exploit 结构。

**selected_intervention**：
- 删除：原来的 `touchdown_reward`（每次腿接触的质量加成）和 `landing_quality_reward`（从未触发）。
- 新增：`first_touchdown_bonus` 一次性奖励（首次腿接触转移事件），消除反复触地动机。
- 新增：`contact_reward` 稳定接触温和奖励（仅当双足接触且速度/姿态低时给每步 +2），提供着陆过程中的稠密正反馈。
- 新增：`success_bonus` 终端成功奖励（+200），通过 info 检测成功条件，给明确的任务完成信号。
- 保留并微调：`proximity_progress`、`attitude_penalty`、`speed_penalty` 保持原有形态，仅 speed penalty 的近地系数适当加大（从 -0.1 调至 -0.15）以鼓励更快减速。

**falsifiable_hypothesis**：
- 一次性首次触地将消除“碰地—飞起—再碰地”的循环 exploit；成功 bonus 为最终目标提供强力锚点；稳定接触温和奖励填补了 landing_quality 空窗，引导 agent 在触地后趋于稳定。三者结合应使策略从“盘旋碰地”转向“快速下降—柔和初触地—稳定着陆”，从而提升 terminated 率并提高得分。
- 若下一轮 score 大幅改善（>30 分），主要改善来自终局成功信号和接触奖励的增加。

**expected_next_round**：
策略将更果断地下降，初触地后倾向维持双足接触并快速减速以获取 `contact_reward`，最终达成成功条件以收获 `success_bonus`。episode 长度将显著缩短，terminated 比例明显提高，平均 score 升至正区间。

---

### 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress    : dense progress towards origin (target at (0,0))
      attitude_penalty      : penalty for tilt and angular velocity
      speed_penalty         : stage-gated penalty on linear velocity (higher near ground)
      first_touchdown_bonus : one-time reward for initial leg contact
      contact_reward        : per-step reward while both legs gently touch the ground
      success_bonus         : large terminal reward for successful landing (via info)
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty (stage-gated) ----------
    y = next_obs[1]                   # vertical coordinate
    vx = next_obs[2]
    vy = next_obs[3]
    speed_sq = vx ** 2 + vy ** 2
    if y < 0.5:                       # near ground – require stronger braking
        speed_penalty = -0.15 * speed_sq
    else:                             # in flight – allow faster descent
        speed_penalty = -0.01 * speed_sq

    # ---------- first touchdown bonus (event, not per-step) ----------
    old_leg_contact = (obs[6] + obs[7]) > 0.5
    new_leg_contact = (next_obs[6] + next_obs[7]) > 0.5
    first_touchdown = (not old_leg_contact) and new_leg_contact
    first_touchdown_bonus = 30.0 if first_touchdown else 0.0

    # ---------- contact reward (both legs down, gentle conditions) ----------
    leg_contact_stable = (next_obs[6] + next_obs[7]) > 1.5
    speed = (speed_sq) ** 0.5
    angle_mag = abs(angle)
    if leg_contact_stable and speed < 0.3 and angle_mag < 0.2:
        contact_reward = 2.0
    else:
        contact_reward = 0.0

    # ---------- success bonus (terminal, via info) ----------
    success = bool(info.get('success', False))
    success_bonus = 200.0 if success else 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + first_touchdown_bonus
                    + contact_reward
                    + success_bonus)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "first_touchdown_bonus": first_touchdown_bonus,
        "contact_reward": contact_reward,
        "success_bonus": success_bonus,
    }
    return float(total_reward), components
```