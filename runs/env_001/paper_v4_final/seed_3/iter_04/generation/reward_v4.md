# 设计理由
前 3 轮迭代在 `progress + angle gate + lateral penalty + angvel penalty + landing bonus` 骨架下连续改进，但 score 仍深陷负值（-71），且所有 episode 都被 terminated（len≈70），说明 agent 在试图接近原点时频繁坠毁或被终止，始终无法完成着陆。  
当前 landing bonus 的 active rate 仅 0.7%——几乎只在少数步触发，绝大多数时间梯度为零，完全无法引导 agent 学习“如何柔和降落到目标平台上”。同时 lateral drift penalty（惩罚 vx²）可能抑制必要的水平纠偏动作，使 agent 更难对准着陆点。

重建思路：
- **放弃稀疏 landing bonus**，改用全局连续的 **soft landing proxy**：基于高度（|y|）、垂直速度、姿态三个连续因子的几何平均，使 agent 从下降早期就能获得“接近地面、低速、姿态端正”的正向梯度。
- **将 lateral drift penalty 替换为横向位置惩罚**：惩罚 x²，鼓励 agent 回到中线但不抑制必要的水平移动。
- 保留有效的 progress（距离减少）+ angle gate 结构，作为主进展信号。
- 保留轻量角速度惩罚以平滑动作。
- 系数校准让 landing 信号在途中贡献约 0.3–0.6 per step，与 progress（≈0.15）叠加，形成正向总奖励；位置与角速度惩罚负担极轻，不压制主信号。

新骨架本质区别：用 **连续 joint_condition_proxy（几何平均）** 替代原来“依赖接触且过于严格”的乘积，提供了从任意状态到完美着陆的稠密梯度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # observation indices: 0:x, 1:y, 2:vx, 3:vy, 4:angle, 5:ang_vel, 6:left_contact, 7:right_contact
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]

    # ---------- 1. progress towards origin (distance decrease) ----------
    dist_old = (x**2 + y**2) ** 0.5
    dist_new = (nx**2 + ny**2) ** 0.5
    progress = dist_old - dist_new
    w_progress = 10.0

    # ---------- 2. attitude gate: suppress progress when angle is dangerous ----------
    abs_angle = abs(n_angle)
    angle_gate = 1.0 - 0.8 * (2.718281828 ** (20.0 * (abs_angle - 0.15)) / 
                              (1.0 + 2.718281828 ** (20.0 * (abs_angle - 0.15))))

    # ---------- 3. lateral position penalty (encourage centering) ----------
    w_lat_pos = 0.15
    lateral_pos_penalty = -w_lat_pos * (nx ** 2)

    # ---------- 4. angular velocity penalty (smooth rotation) ----------
    w_angvel = 0.05
    angvel_penalty = -w_angvel * (n_ang_vel ** 2)

    # ---------- 5. continuous landing proxy (geometric mean) ----------
    k_y = 1.0
    k_vy = 10.0
    k_ang = 20.0
    f_y   = 1.0 / (1.0 + k_y   * abs(ny))
    f_vy  = 1.0 / (1.0 + k_vy  * abs(nvy))
    f_ang = 1.0 / (1.0 + k_ang * abs(n_angle))
    landing_factor = (f_y * f_vy * f_ang) ** (1.0 / 3.0)
    w_land = 2.0
    landing_reward = w_land * landing_factor

    # ---------- combine ----------
    total_reward = (w_progress * progress * angle_gate
                    + lateral_pos_penalty
                    + angvel_penalty
                    + landing_reward)

    components = {
        "progress_gated": w_progress * progress * angle_gate,
        "lateral_pos_penalty": lateral_pos_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_reward": landing_reward
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 全部观测维度已使用，但原 landing 信号因过于严格的 contact 门控几乎无梯度（active 0.7%），信号覆盖有缺口；重建用连续高度‑速度‑姿态因子补齐。
- **behavior**: agent 向原点移动但坠毁频繁（len~70, 全 terminated），无法完成软着陆。
- **signal**: 缺失从远处到着陆的全过程稠密梯度引导，原稀疏 landing bonus 无法提供有效学习信号。
- **level**: Level 3
- **hypothesis**: 全局连续的 soft landing proxy 会让 agent 在每个 step 都能感知“接近地面且稳定”的价值，从而学会减速、调姿、对准后平稳触地。
- **risk**: landing reward 可能在某些区域变得过大，使 agent 为追求 landing 奖励而牺牲前进速度；如发生此现象需降低 w_land。