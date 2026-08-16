# Response Record

**1. 诊断报告**

- **evidence**
  - `score=-10.42`，评分为负且离目标 200 很远；同时 `soft_landing_bonus` 均值高达 `320.48`，占奖励量级的 88.8%，而 `active_rate` 仅 18.5%，说明该奖励只在少量步触发但单步或持续累积值极大。
  - `proximity_progress` 均值 22.12，几乎全程活跃，引导作用被淹没。
  - Checkpoint 趋势：`soft_landing_bonus` 在 cp60% 后快速膨胀（82→214），而同期得分先升（cp80% 42.2, cp90% 67.2）后暴跌至 cp100% -36.4，提示奖励 exploit 在后期爆发。
  - `terminated=15/20`, `truncated=5/20`, `len=487.75`：大量 episode 以自我终止结束（着陆或坠毁），但平均步长较长，存在盘旋、延迟着陆。

- **behavior_diagnosis**
  策略学会了利用 `soft_landing_bonus`：一旦任意位置着陆（哪怕偏离靶心），只要腿触地并保持低速/小倾角，每步都能获得 4~8 的持续奖励，累积出极高的总分。而原始环境对时间、燃料或非靶心着陆有惩罚，导致原始评分被拉低甚至为负。Agent 没有动力去追求靶心精确着陆，只求尽快触地然后“赖着不动”。

- **signal_completeness**
  缺失对 “最终成功降临目标垫中心” 的明确奖励。当前的 `soft_landing_bonus` 是一个**持续过程奖励**，且完全不包含位置精度因子，鼓励了“随便着陆 + 赖地”的行为。需要将着陆奖励改造为**一次性事件型**，并根据位置、速度、姿态给出差异化的质量和精确度信号。

- **selected_level**：L2  
  不能仅靠 L1 调系数解决 exploit，必须改变 `soft_landing_bonus` 的数学形态，从持续过程奖励变为首次着陆事件的一次性奖励，并嵌入位置精度因子。

- **selected_intervention**
  目标组件 `soft_landing_bonus`：
  - 探测着陆事件：`old_any_contact → new_any_contact`。
  - 发生着陆时，根据以下因素计算一次性奖励：
    - 水平位置偏差：`exp(-|x|/0.3)`，鼓励对准中心。
    - 速度抑制：`1/(1+10*(|vx|+|vy|))`，惩罚硬着。
    - 姿态抑制：`1/(1+5*|angle|)`，惩罚歪斜。
    - 腿数因子：`(n_contacts/2)`，轻微偏好双腿同时触地。
  - 非着陆步，该组件返回 0。

- **falsifiable_hypothesis**
  改为事件型高质量着陆奖励后，agent 无法再通过“随便着陆 + 赖地”刷分；它必须精准降落在靶心且做到软着陆，才能获得大量奖励。这将引导策略学习真正的靶心软着陆，从而提高原始环境评分。

- **expected_next_round**
  下一轮：
  - `soft_landing_bonus` 的 episode_sum_mean 会下降到 ~100–200 级别的单次值，active_rate 由接触事件的稀疏性决定。
  - 得分将摆脱负值平台，向 200 趋近，但可能因稀疏奖励导致早期探索不足，需后续观察是否需要增加中间引导信号。
  - 终止率保持高位，episode 长度或下降，因为不再有动机滞空。

**2. 完整 Python 代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress : dense signal – progress towards origin (target pad at (0,0))
      attitude_penalty   : penalty for tilt and angular velocity
      soft_landing_bonus : event-based – rewarded only on first leg contact,
                           scaled by landing quality (position, velocity, angle)
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- soft landing bonus (event-based) ----------
    # Detect first leg contact
    old_any_contact = (obs[6] + obs[7]) > 0.5
    new_any_contact = (next_obs[6] + next_obs[7]) > 0.5
    landing_event = (not old_any_contact) and new_any_contact

    if landing_event:
        x = next_obs[0]            # horizontal offset from target centre
        vx = abs(next_obs[2])
        vy = abs(next_obs[3])
        angle_abs = abs(angle)

        # position factor – exponentially decay with |x|, width ~0.3
        pos_factor = 2.718281828 ** (-abs(x) / 0.3)

        # velocity penalty – low speeds desired
        vel_factor = 1.0 / (1.0 + 10.0 * (vx + vy))

        # angle penalty
        angle_factor = 1.0 / (1.0 + 5.0 * angle_abs)

        # leg factor (1 or 2)
        leg_factor = next_obs[6] + next_obs[7]  # 1 or 2

        # max bonus = 200 (aligns with environment target score)
        soft_landing_bonus = 200.0 * pos_factor * vel_factor * angle_factor * (leg_factor / 2.0)
    else:
        soft_landing_bonus = 0.0

    total_reward = proximity_progress + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```
