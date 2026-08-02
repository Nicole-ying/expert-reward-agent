# 设计理由

修改对象：新增 `fuel_penalty` 组件。  
当前奖励函数缺失引擎使用惩罚，但环境事实明确指出次要目标是节约燃料。观测／动作审计显示所有观测维度已被使用，且无信号缺口，但 action 信号（Discrete 4）未被纳入奖励，导致 agent 可能过度使用引擎而只能从主着陆目标中获得信号，忽略了效率问题。  
本次修改为一个轻微的常驻惩罚，仅对非 `no_engine` 动作（action ≠ 0）施加 -0.05，使 agent 在保持精准着陆的同时倾向于减少不必要的引擎点火。  

系数校准：  
- `progress` 的 per‑step 约 0.049（13.0/264），`landing_bonus` 的 per‑step 约 0.212（56.0/264），主信号 per‑step ≈ 0.261。  
- 新惩罚 per‑step 最坏约为 0.05（若每步都用引擎），比率为 0.19，低于 0.3× 的限制。  
- 总惩罚负担（含几乎为零的角度、角速度惩罚）远低于 0.5× 主信号。  
其他组件保持不变，确保当前优良的着陆策略不受干扰。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, l_contact, r_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nl_contact, nr_contact = next_obs

    # 1. Main progress signal: distance reduction to target pad
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_obs - dist_next

    # 2. Stability constraints (quadratic penalties on next state)
    angle_penalty = -0.1 * (nangle ** 2)
    angvel_penalty = -0.05 * (nangvel ** 2)

    # 3. Contact-gated soft landing attractor (tightened conditions)
    proximity = max(0.0, 1.0 - dist_next / 0.5)
    speed = abs(nvx) + abs(nvy)
    speed_factor = max(0.0, 1.0 - speed / 0.3)
    angle_factor = max(0.0, 1.0 - abs(nangle) / 0.15)

    contact_gate = float(nl_contact or nr_contact)
    landing_attractor = proximity * speed_factor * angle_factor * contact_gate

    # 4. Fuel penalty: discourage unnecessary engine usage
    fuel_penalty = -0.05 if action != 0 else 0.0

    w_progress = 10.0
    w_attractor = 2.0

    total = (w_progress * progress +
             angle_penalty + angvel_penalty +
             w_attractor * landing_attractor +
             fuel_penalty)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": w_attractor * landing_attractor,
        "fuel_penalty": fuel_penalty
    }

    return float(total), components
```

# 诊断摘要
- **audit**: 所有观测维度已被使用，信号齐全；landing_bonus 在接触阶段提供强奖励符合成功着陆模式，但缺少燃料效率激励。
- **behavior**: agent 已学会利用接触门控的着陆奖励实现精准快速着陆，平均 264 步完成，20/20 成功终止。
- **signal**: landing_bonus 占比过高（79.9%）但行为合理；缺失对引擎使用的激励，导致次要燃料目标被完全忽略。
- **level**: Level 2
- **hypothesis**: 新增轻量燃料惩罚（-0.05/步）将引导 agent 在不损害着陆精度的前提下减少无效引擎点火，从而提升整体任务贴合度。
- **risk**: 极小的惩罚可能略微延长 episode 长度或使 agent 犹豫，但系数足够低，主着陆梯度仍占主导，风险可控。