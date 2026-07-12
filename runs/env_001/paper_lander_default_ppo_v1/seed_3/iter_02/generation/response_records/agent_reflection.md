# Response Record

`evidence`：当前score=231.37已超target=200（115.7%），terminated=13/20、truncated=7/20，episode长度589步，无早期失败。proximity_reward均值1036.57占signed_share 76.8%，contact_bonus均值310.23占23.0%，三项penalty合计约-3.03（占比<0.3%）几乎无行为塑造力。上一轮无历史修改。

`behavior_diagnosis`：策略已能成功导航着陆并保持双脚接触，但35%的episode被截断而非自然终止，可能因agent在目标平台附近微幅振荡未能触发body_not_awake_or_settled，或评估框架存在时间上限；proximity主导的per-step奖励使agent缺乏"完全静止"的边际激励。

`signal_completeness`：goal_proximity（proximity_reward）和soft_landing（三项near-gated penalty + contact_bonus）职责均存在且可达。penalty组件符号正确、数学形态合理，但系数过小导致幅度可忽略。contact_bonus数学形态正确（near-gated乘积），但相对proximity的占比可加强以更好引导双脚稳定接触。

`selected_level`：Level 1。contact_bonus组件职责、符号和数学形态均合理，仅相对强度可微调以更强调双脚同时接触这一成功着陆关键条件。penalty组件虽幅度可忽略但系数已较小且agent已表现出良好速度/姿态控制，无需调整。

`selected_intervention`：仅将contact_bonus系数w_contact从1.0提升至2.0，其他所有组件保持不变。

`falsifiable_hypothesis`：提高contact_bonus权重使双脚同时接触的边际收益从+1.0/步升至+2.0/步（与proximity在goal处的+2.0/步等权重），应强化agent对双脚同时着地的偏好，促使agent在接近目标后更快达成稳定双脚接触状态，从而减少因微幅振荡导致的截断，提高自然终止率。

`expected_next_round`：contact_bonus的episode_sum_mean和signed_share应上升（预计从23%升至35-40%），terminated率应从65%上升，truncated率下降，score保持或略有提升（因更早稳定接触可减少无效步数），episode_length可能小幅缩短。

`main_risk`：contact_bonus权重翻倍可能在极少数误触场景（非目标平台表面双脚接触）给予过多奖励，但由于near门控已限制远距离时的contact_bonus幅度，且物理上非平台表面难以同时触发双脚接触，实际风险很低。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract next state signals
    x = next_obs[0]
    y = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to target platform center (goal is (0,0))
    distance = (x**2 + y**2) ** 0.5

    # ---------- main learning signal: goal proximity ----------
    w_prox = 2.0
    # Bounded positive reward: maximum 2.0 at distance 0, decays smoothly
    prox_reward = w_prox / (1.0 + distance)

    # ---------- near-factor for soft-landing constraints ----------
    near = 1.0 / (1.0 + distance)   # 1 when close, ~0 when far

    # ---------- velocity penalty ----------
    w_vel = 0.1
    vel_penalty = -w_vel * near * (x_vel**2 + y_vel**2)

    # ---------- body angle penalty ----------
    w_angle = 0.2
    angle_penalty = -w_angle * near * (angle**2)

    # ---------- angular velocity damping ----------
    w_angvel = 0.1
    angvel_penalty = -w_angvel * near * (ang_vel**2)

    # ---------- contact bonus (both feet on ground) ----------
    w_contact = 2.0
    both_contacts = left_contact * right_contact   # 0 or 1
    contact_reward = w_contact * near * both_contacts

    # ---------- total reward ----------
    total_reward = prox_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        "proximity_reward": prox_reward,
        "velocity_penalty": vel_penalty,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angvel_penalty,
        "contact_bonus": contact_reward
    }

    return float(total_reward), components
```
