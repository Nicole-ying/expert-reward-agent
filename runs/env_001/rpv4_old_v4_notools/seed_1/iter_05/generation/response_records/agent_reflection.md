# Response Record

## 1. 诊断报告

**evidence**  
- 本轮训练最终得分 39.44，目标 200，差距极大；全部 20 局均以 truncation（1000 步）结束，terminated=0，即 agent 从未成功稳定着陆。  
- `soft_landing_bonus` 的 episode_sum_mean 高达 4033，占 magnitude_share 的 98.7%，而其 active_rate 仅 8.3%——说明 agent 只在少数步骤触发该奖励，但每次触发都能获得极高数值（ep_sum_mean / active_rate ≈ 48500 / episode 的 83 步 ≈ 每次触发 ~58 奖励）。检查源码发现这是一个“事件触发”奖励，但因为触发条件仅判断腿从无接触到有接触，agent 可以反复离地再接触，从而在着陆垫附近高频弹跳来刷取该奖励。  
- checkpoint 趋势显示 `soft_landing_bonus` 从 cp10% 的 39 急剧攀升到 cp50% 的 4122，然后保持高位（3787），验证了 agent 学会 exploit 这一机制，并在训练中持续滥用。  
- `proximity_progress` 和 `attitude_penalty` 的量级远小于 `soft_landing_bonus`，几乎被淹没，说明 agent 绝大部分注意力被弹跳刷分所占据，忽略了稳定着陆的核心要求。

**behavior_diagnosis**  
当前策略的行为模式：先飞抵目标原点附近（受 `proximity_progress` 驱动），然后通过反复让腿部短暂接触垫面来连续获取 `soft_landing_bonus`，而不去降低速度、保持姿态以实现真正停靠。这是一种不健康的 exploit 行为，直接导致从未成功终止，得分远低于目标。

**signal_completeness**  
必要信号基本覆盖：距离进步、姿态维持。但缺失两个关键信号：  
1. 真正的“成功着陆”应仅奖励一次且与任务完成对齐，而非可重复触发。  
2. 缺少对线速度（尤其是高速接近）的惩罚，导致 agent 在接近原点时没有减速动力，即使触地也因速度过高而无法稳定。

**selected_level**  
L2。需要修改 `soft_landing_bonus` 的数学形态（从可重复触发的事件改为一次性成功事件），并联动新增一个速度惩罚组件（服务于同一语义——引导减速、杜绝弹跳刷分）。

**selected_intervention**  
1. 将 `soft_landing_bonus` 重构为 **成功着陆一次性奖励**：仅在从“不满足成功条件”到“满足成功条件”的转变时给予 200 奖励。成功条件严格：双腿接触、位置足够近、速度足够低、姿态足够平。  
2. 新增 `speed_penalty` 组件，对线速度平方施加轻微惩罚，鼓励 agent 在飞行全程尤其是接近垫面时降低速度。

**falsifiable_hypothesis**  
移除可重复触发的软着陆奖励后，agent 失去在垫上反复弹跳的动力；新增的速度惩罚会迫使它在接近原点时减速，从而更可能满足成功着陆条件，并一次性获得大额奖励，最终学会稳定停靠。预计 terminated 比例将上升，得分将提升。

**expected_next_round**  
agent 将学会以较低速度飞向原点，在接触前有效减速，并使双腿同时轻柔接触垫面，姿态平稳，从而触发成功奖励，进入环境检测的终止状态。episode 将以成功截断（terminated）结束，得分接近或达到 200。

---

## 2. 修改后的奖励函数代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress : dense signal – progress towards origin (target at (0,0))
      attitude_penalty   : penalty for tilt and angular velocity
      speed_penalty      : penalty for linear velocity (encourages slow approach)
      soft_landing_bonus : event-based – one-time +200 when a stable landing is first achieved
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 30.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty (new) ----------
    vx = next_obs[2]
    vy = next_obs[3]
    speed_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---------- soft landing bonus (one-time success event) ----------
    # success condition: both legs in contact, close to centre, low speed, low tilt
    def success(ob):
        leg_contact = (ob[6] + ob[7]) > 1.5           # both legs must touch
        pos_ok = abs(ob[0]) < 0.3                     # horizontal offset
        speed_ok = (ob[2]**2 + ob[3]**2)**0.5 < 0.3   # low linear speed
        angle_ok = abs(ob[4]) < 0.2                    # low tilt
        return leg_contact and pos_ok and speed_ok and angle_ok

    old_success = success(obs)
    new_success = success(next_obs)
    landing_event = (not old_success) and new_success

    soft_landing_bonus = 200.0 if landing_event else 0.0

    # ---------- total ----------
    total_reward = proximity_progress + attitude_penalty + speed_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "soft_landing_bonus": soft_landing_bonus,
    }
    return float(total_reward), components
```
