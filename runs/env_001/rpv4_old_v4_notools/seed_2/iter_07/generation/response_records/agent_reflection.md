# Response Record

### 1. 诊断报告

**evidence**  
- 最终评估：score=167.9，len=495.4，terminated=18/20。分数未达200，episode接近500步上限，说明飞行器可能行进缓慢、着陆后停留过久。  
- stable_contact_bonus magnitude_share=68.3%，ep_sum_mean=339.5，active_rate=34.3%（约170步获得奖励），是主要信号来源；contact_bonus（事件奖励）也有贡献但稀少。  
- progress（接近信号）ep_sum_mean仅3.96，active_rate>99%但量级极小，说明每一步几乎都有微小距离进步，但累积值低——agent移动太慢。  
- checkpoint趋势显示：总成绩在中后期波动大，stable_contact_bonus和contact_bonus剧烈波动，progress和penalties平稳，表明行为主要由接触奖励驱动，接近信号未能有效塑造速度。  

**behavior_diagnosis**  
- 策略学会了缓慢下降并着陆，着陆后持续保持低垂直速度、双腿接触以获取持续stable_contact_bonus，直至步数耗尽或终止。这导致episode长度接近上限，且速度慢、效率低。  
- 该行为不健康：它拖延了任务完成，未能追求“尽可能快”的主目标，同时也可能浪费燃料（缺乏惩罚）。终止率高说明最终完成了接触，但时间成本过高是主因。  

**signal_completeness**  
- 有接近信号（progress）、速度/姿态惩罚、接触事件、稳定接触持续奖励。缺少明确的“紧迫性”或“时间惩罚”来驱动快速完成。progress的权重不足以对抗稳定停留的收益。  
- velocity_penalty和orientation_penalty存在但量级很小，未严重干扰。  

**selected_level**  
- **L1**（调系数，不改数学式）。理由：当前各组件方向基本正确（接近信号、接触事件、稳定奖励都有），力度失衡是主因。差距<20%（32/200=16%），但len过长表明主要是时间效率问题，L1调整权重有望改善，风险较低。  

**selected_intervention**  
- **增加progress系数**：从4.0 → 8.0，让接近信号更强，驱动更快的下降过程。  
- **降低stable_contact_bonus**：从2.0 → 0.5，减少着陆后持续停留的激励，使agent不会过度追求“保持接触赚分”。  
- 其他组件保持不变（velocity_penalty、orientation_penalty、contact_bonus）。  

**falsifiable_hypothesis**  
- 更强的接近梯度会推动agent更快移动到目标区；降低稳定接触持续奖励会缩短着陆后的滞留时间，降低episode长度。两项合力使agent在更少步数内完成着陆，减少时间/燃料消耗，从而提升环境外部得分。如果下一轮len明显下降且score上升，即验证假设。若出现速度过快导致失控着陆或接触减少，则需后续微调velocity_penalty。  

**expected_next_round**  
- episode长度应降到400步以下，平均行进速度提升；stable_contact_bonus的ep_sum_mean下降，progress均值略升；总外部得分提高（预期180+），同时接触事件仍保持高触发。  

---

### 2. 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to center (used for progress shaping, not for penalty)
    dist_prev = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # --- progress: strong shaping towards target  (L1: increased coefficient 4→8)
    progress = 8.0 * (dist_prev - dist_next)

    # --- velocity penalty: penalize high speed to encourage safe landing
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.03 * (vx ** 2 + vy ** 2)

    # --- orientation penalty: discourage large tilt
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.03 * (next_obs[5] ** 2)

    # --- contact event reward: bonus for newly touching a leg to the platform
    leg_left_prev = obs[6]
    leg_right_prev = obs[7]
    leg_left_next = next_obs[6]
    leg_right_next = next_obs[7]

    contact_event = 0.0
    if leg_left_next > 0.5 and leg_left_prev < 0.5:
        contact_event += 1.0
    if leg_right_next > 0.5 and leg_right_prev < 0.5:
        contact_event += 1.0
    contact_bonus = 5.0 * contact_event

    # --- stable contact reward: small per-step reward for maintaining contact with low vertical speed
    # (L1: reduced from 2.0 to 0.5 to discourage lingering)
    contact_active = leg_left_next + leg_right_next
    stable_contact_bonus = 0.0
    if contact_active > 0.5 and abs(vy) < 0.2:
        stable_contact_bonus = 0.5

    # --- combine
    total_reward = (
        progress
        + velocity_penalty
        + orientation_penalty
        + contact_bonus
        + stable_contact_bonus
    )

    components = {
        "progress": progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_bonus": contact_bonus,
        "stable_contact_bonus": stable_contact_bonus,
    }

    return float(total_reward), components
```
