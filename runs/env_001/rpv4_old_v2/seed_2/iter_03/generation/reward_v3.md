1. `evidence`：iter2将success_proxy从几何平均改为加和，导致其每步均值激增到6.114并100%激活，同时goal_proximity无界累积到-3768，len从68暴涨到947，truncated增多到13/20，score反而恶化；历史最佳iter1中goal_proximity主导（per‑step约-2.16），contact_quality和success_proxy几乎死掉，策略快速终结（len≈68，全terminated）。

2. `behavior_diagnosis`：iter2的agent学会了在所有位置都能获得大量success_proxy正值，从而用无界goal_proximity惩罚“买”生存时间，但并未向目标靠拢，多数回合被截断；iter1的agent则因无界goal_proximity惩罚过强，迅速坠毁或失败终止，没有机会学习着陆。

3. `signal_completeness`：任务需要的接近引导、稳定着陆与燃料约束已具备，但无界goal_proximity在两种版本中都成了阻碍——要么压灭其他信号导致短命，要么被其他虚高信号中和导致无意义延长；必须将其改为有界形式，使其提供梯度但不主导生存决策。

4. `selected_level`：Level 2 — 修改goal_proximity的数学形态，将其从无界线性惩罚改为有界惩罚，以消除全覆盖惩罚压制探索和诱导虚假补偿的根源。

5. `selected_intervention`：唯一目标是goal_proximity，将 `comp_prox = w_prox * ( -abs(x) - abs(y) )` 改为 `comp_prox = w_prox * ( - (x**2 + y**2) / (1.0 + x**2 + y**2) )`，并调整权重 `w_prox = 5.0`（使远处步惩罚约‑5，近处接近0）。

6. `falsifiable_hypothesis`：有界goal_proximity会让远处惩罚不再无限累积，agent不会因远离目标而快速死亡，同时无法靠虚假正值无限延长回合，从而被迫去探索接近目标的动作；这将提升len适度增加，terminated减少，score向0改善。

7. `expected_next_round`：len从68上升至200‑400，truncated出现但比例<50%，terminated比例下降但仍有成功终止（settled），score负值绝对值减小（>-121），goal_proximity magnitude_share下降，success_proxy和contact_quality开始激活。

8. `main_risk`：如果goal_proximity惩罚太弱，agent可能转为懒散悬停或轻微飘离而不追求着陆，导致len接近截断上限且仍得不到正success_proxy；需后续观察是否需增强success_proxy的几何平均阈值或接触奖励。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observation (using next_obs, which represents the state after action)
    x       = next_obs[0]   # horizontal offset from target pad center
    y       = next_obs[1]   # vertical offset from target pad (height)
    vx      = next_obs[2]   # horizontal velocity
    vy      = next_obs[3]   # vertical velocity
    angle   = next_obs[4]   # body tilt angle (rad)
    angvel  = next_obs[5]   # angular velocity
    left_c  = next_obs[6]   # left leg contact flag (0 or 1)
    right_c = next_obs[7]   # right leg contact flag (0 or 1)

    # ------------------------------------------------------------------
    # Component A – goal proximity (bounded, soft guidance)
    # ------------------------------------------------------------------
    # Changed from unbounded -abs(x)-abs(y) to a bounded penalty that
    # saturates far from the pad.
    w_prox  = 5.0
    dist_sq = x**2 + y**2
    comp_prox = w_prox * ( - dist_sq / (1.0 + dist_sq) )

    # ------------------------------------------------------------------
    # Component B – velocity penalty
    # ------------------------------------------------------------------
    w_vel = 1.0
    comp_vel = w_vel * ( -vx**2 - vy**2 )

    # ------------------------------------------------------------------
    # Component C – angle penalty
    # ------------------------------------------------------------------
    w_ang = 2.0
    comp_angle = w_ang * ( -angle**2 - angvel**2 )

    # ------------------------------------------------------------------
    # Component D – contact quality (stability constraint)
    # ------------------------------------------------------------------
    # Reward simultaneous leg contact, gated by position and velocity.
    w_contact = 10.0

    pos_thresh = 0.3
    vel_thresh = 0.3

    gate_x   = max(0.0, 1.0 - abs(x) / (pos_thresh + 1e-8))
    gate_y   = max(0.0, 1.0 - abs(y) / (pos_thresh + 1e-8))
    gate_vx  = max(0.0, 1.0 - abs(vx) / (vel_thresh + 1e-8))
    gate_vy  = max(0.0, 1.0 - abs(vy) / (vel_thresh + 1e-8))

    leg_contact = left_c * right_c
    comp_contact = w_contact * leg_contact * gate_x * gate_y * gate_vx * gate_vy

    # ------------------------------------------------------------------
    # Component E – success proxy (geometric mean of near‑perfect conditions)
    # ------------------------------------------------------------------
    w_proxy = 20.0

    eps_x     = 0.1
    eps_y     = 0.1
    eps_vx    = 0.15
    eps_vy    = 0.15
    eps_angle = 0.15

    score_x     = max(0.0, 1.0 - abs(x) / eps_x)
    score_y     = max(0.0, 1.0 - abs(y) / eps_y)
    score_vx    = max(0.0, 1.0 - abs(vx) / eps_vx)
    score_vy    = max(0.0, 1.0 - abs(vy) / eps_vy)
    score_angle = max(0.0, 1.0 - abs(angle) / eps_angle)
    score_contact = (left_c + right_c) * 0.5

    prod = (score_x * score_y * score_vx * score_vy * score_angle * score_contact)
    if prod > 1e-12:
        comp_proxy = w_proxy * (prod ** (1.0 / 6.0))
    else:
        comp_proxy = 0.0

    # ------------------------------------------------------------------
    # Component F – thrust penalty (efficiency)
    # ------------------------------------------------------------------
    w_thrust = 0.05
    comp_thrust = -w_thrust * float(action != 0)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_reward = (comp_prox + comp_vel + comp_angle +
                    comp_contact + comp_proxy + comp_thrust)

    components = {
        "goal_proximity": comp_prox,
        "velocity_penalty": comp_vel,
        "angle_penalty": comp_angle,
        "contact_quality": comp_contact,
        "success_proxy": comp_proxy,
        "thrust_penalty": comp_thrust,
    }

    return float(total_reward), components
```