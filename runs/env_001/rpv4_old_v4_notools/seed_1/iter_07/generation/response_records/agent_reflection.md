# Response Record

## 诊断报告

**evidence**  
- 最终 score = -39.7，episode 长度=1000（全部 truncated），无成功终端。  
- checkpoint 中 cp50% 时 score 曾达到 114.6，但随后退回负值；leg_contact_reward 在 cp40‑50% 出现 269/213 的高值（active_rate 14‑19%），最终 eval 归零。  
- proximity_progress 占 magnitude share 96%，全程主导，说明 agent 被单一「靠近」信号驱动，缺乏着陆引导。  
- soft_landing_bonus 全程未触发，leg_contact_reward 当前为标量接触标志 ×2.0，无姿态/速度条件，历史上曾被 exploit 后消失。  
- attitude_penalty、speed_penalty 量级过小，无法有效约束靠近后的行为。

**behavior_diagnosis**  
策略当前的行为是：朝目标区域趋近（proximity_progress 持续为正），但在接近后不尝试着陆，而是保持悬停/振荡以避免腿部接触（leg_contact_reward=0），最终因超时被截断。中期曾因 leg_contact_reward 获得大量奖励，但该奖励无稳定性约束，导致高速冲撞式接触，随后被抑制或衰退，反而让 agent 学到「接触地面有害」。整体是不健康的徘徊策略。

**signal_completeness**  
缺失引导稳定着陆的连续奖励。现有 soft_landing_bonus 过于稀疏，leg_contact_reward 仅反映接触与否，不评估着陆质量，缺失「安全着陆」信号。

**selected_level**  
L2 – 修改 leg_contact_reward 的数学形态，将其替换为基于着陆质量的连续奖励，同时微调 proximity_progress 权重以平衡，整体服务于同一语义：为着陆阶段提供精细梯度信号。

**selected_intervention**  
- 删除原 `leg_contact_reward`，新增 `touchdown_reward`：只在有腿接触时，以 `(1 - dist/0.3) * (1 - speed/0.3) * (1 - angle/0.2)` 刻画着落质量，系数 10.0，clip 到 [0, 1] 再乘系数，避免负值。  
- 将 `proximity_progress` 的系数从 30.0 降至 20.0，以减少其对靠近后行为的压倒性引导，让着陆奖励起效。

**falsifiable_hypothesis**  
增加基于质量的着陆连续奖励，能在 agent 接近目标区域时提供梯度指引，使其在低速、小角度状态下主动尝试腿接触，从而增加稳定着陆概率，提升最终得分并可能提前终止。若下一轮 leg_contact 的 active_rate 回升且 soft_landing_bonus 仍然为 0，则说明接触条件仍太过严格，需要进一步放宽或增强瞬间奖励。

**expected_next_round**  
策略将学会在接近后减速、调整姿态并尝试腿接触，episode 长度可能缩短，success 率提升，score 大幅改善（期望 >50），leg_contact 的 active_rate 显著回升。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Components:
      proximity_progress  : dense signal – progress towards origin (target at (0,0))
      attitude_penalty    : penalty for tilt and angular velocity
      speed_penalty       : penalty for linear velocity (encourages slow approach)
      touchdown_reward    : dense reward for high-quality leg contact during soft landing
      soft_landing_bonus  : event-based – one-time +200 when a stable landing is first achieved
    """
    # ---------- proximity progress ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)

    # ---------- attitude penalty ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.6 * (angle ** 2) - 0.12 * (angular_vel ** 2)

    # ---------- speed penalty ----------
    vx = next_obs[2]
    vy = next_obs[3]
    speed_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---------- touchdown reward (leg contact quality) ----------
    # Only reward leg contact when the vehicle is stable (low speed, low tilt, near target)
    leg1 = next_obs[6]
    leg2 = next_obs[7]
    leg_contact = (leg1 + leg2) > 0.5   # at least one leg touching
    if leg_contact:
        dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
        speed = (vx ** 2 + vy ** 2) ** 0.5
        ang = abs(angle)
        # quality: 1.0 for each condition fully satisfied, linearly down to 0
        quality = max(0.0, 1.0 - dist / 0.3) * max(0.0, 1.0 - speed / 0.3) * max(0.0, 1.0 - ang / 0.2)
        touchdown_reward = 10.0 * quality
    else:
        touchdown_reward = 0.0

    # ---------- soft landing bonus (one-time success event) ----------
    def success(ob):
        leg_contact_s = (ob[6] + ob[7]) > 1.5           # both legs must touch
        pos_ok = abs(ob[0]) < 0.3                       # horizontal offset
        speed_ok = (ob[2]**2 + ob[3]**2)**0.5 < 0.3     # low linear speed
        angle_ok = abs(ob[4]) < 0.2                      # low tilt
        return leg_contact_s and pos_ok and speed_ok and angle_ok

    old_success = success(obs)
    new_success = success(next_obs)
    landing_event = (not old_success) and new_success
    soft_landing_bonus = 200.0 if landing_event else 0.0

    # ---------- total ----------
    total_reward = (proximity_progress
                    + attitude_penalty
                    + speed_penalty
                    + touchdown_reward
                    + soft_landing_bonus)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "speed_penalty": speed_penalty,
        "touchdown_reward": touchdown_reward,
        "soft_landing_bonus": soft_landing_bonus,
    }
    return float(total_reward), components
```
