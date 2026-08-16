# Response Record

## 诊断报告

**evidence**  
- `score=-471.6, len=968, term=1/20, trunc=19/20`：几乎所有 episode 都截断，从未成功着陆。  
- `landing` 组件全程 `active_rate=0%`, `ep_sum_mean=0`：无论训练哪个阶段，着陆奖励从未触发，说明“双接触+近中心+低速”的门槛过高，agent 收不到任何与着陆相关的正反馈。  
- 距离奖励 `ep_sum_mean=2749`，但速度惩罚 `-916` 且 `active_rate=100%`：距离奖励在全局累积，速度惩罚也在全局施加，导致 agent 在远离目标时也被迫减速，极大抑制了向目标的移动。checkpoint 趋势显示速度惩罚在 cp80% 曾降到 -389，那时总分改善至 -154.6，但最终回弹至 -1111，说明速度惩罚的全局性导致训练后期策略崩溃。  
- 姿态惩罚已收敛到很低（-4 左右），燃料惩罚稳定在 -150 量级，不是主要问题。  
- 历史检查（`read_past_reflection(6)` 等）表明之前几轮主要围绕调整系数和增加 landing bonus，但从未解决“landing 完全不触发”和“全局速度惩罚阻碍接近”这两个根因。

**behavior_diagnosis**  
当前策略学会了向中心靠近（距离奖励很高），但由于速度惩罚无处不在，agent 不敢以足够的速度飞向目标。最后停留在“较近但不触地”的中等距离，速度偏低以避免惩罚，却无法完成着陆。episode 几乎全部截断，无任何成功着陆，正是“可接近但无法停靠”的停滞行为。

**signal_completeness**  
- 缺少“接近并减速”的**渐进式着陆引导信号**：landing 仅靠一次性 bonus，门槛太高，变成死信号。  
- 距离奖励是**状态值**（`1/(1+dist)`），在目标附近驻留可持续获奖，不会强制 agent 完成最终着陆。  
- 速度惩罚是**全局**的，缺乏阶段门控，导致探索阶段不必要的约束。

**selected_level**：**L2**  
理由：单纯调系数（L1）无法改变 landing 死信号和速度惩罚的全局压制；需要修改数学形态以引入“连续接近奖励”和“阶段门控”，且增加势能差形式避免驻留奖励。

**selected_intervention**  
1. **距离奖励**：由状态值改为**势能差**（`potential(next)-potential(now)`），迫近给正、远离给负，消除驻留奖励。  
2. **速度惩罚**：从全局改为**近区门控**——仅在 `dist<1.5` 时施加，允许早期快速接近。  
3. **新增连续接近奖励**：当 `dist<0.8` 且 `velocity<0.8` 时给予基于距离与速度比例的奖励，让 agent 在未接触时也能获得着陆引导，解决 `active_rate=0` 问题。保留一次性 200 超奖。

**falsifiable_hypothesis**  
- 势能差奖励会强化“每一步都更接近目标”的行为，避免在远处累积巨额距离奖励。  
- 近区速度惩罚让 agent 在接近目标后才开始减速，不再被全局惩罚压抑运动。  
- 连续接近奖励填补“无接触→有接触”之间的信号断裂，agent 将首次获得着陆相关反馈，active_rate 大幅上升，训练中会出现成功着陆的 episode。

**expected_next_round**  
策略将学会：第一阶段快速飞向中心（不受速度罚）；第二阶段在目标附近减速；第三阶段在低速近中心时获得持续的正奖励，最终触发着陆。预计 `score` 明显上升（>0），`landing active_rate` 脱离 0，部分 episode 成功着陆并拿到一次性 200。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- extract current and next state ---
    x, y = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    angvel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- helpers ---
    dist = (x ** 2 + y ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    velocity_mag = (vx_next ** 2 + vy_next ** 2) ** 0.5

    # === 1. distance potential difference ===
    w_dist = 30.0
    potential = 1.0 / (1.0 + dist)
    potential_next = 1.0 / (1.0 + dist_next)
    distance_reward = w_dist * (potential_next - potential)

    # === 2. velocity penalty — gated, only when close to target ===
    w_vel = 3.0
    speed_gate = 1.0 if dist_next < 1.5 else 0.0
    velocity_penalty = -w_vel * velocity_mag * speed_gate

    # === 3. attitude stabilization ===
    w_angle = 1.0
    w_angvel = 0.5
    attitude_penalty = -w_angle * (angle_next ** 2) - w_angvel * (angvel_next ** 2)

    # === 4. fuel penalty ===
    if action == 0:
        fuel_cost = 0.0
    elif action == 2:   # main engine
        fuel_cost = 0.2
    else:               # side engines
        fuel_cost = 0.1
    fuel_penalty = -fuel_cost

    # === 5. soft landing approach + one-shot bonus ===
    approach_reward = 0.0
    if dist_next < 0.8 and velocity_mag < 0.8:
        w_approach = 100.0
        norm_dist = dist_next / 0.8
        norm_vel = velocity_mag / 0.8
        approach_reward = w_approach * (1.0 - norm_dist) * (1.0 - norm_vel)

    landing_one_shot = 0.0
    if left_contact and right_contact and dist_next < 0.2 and velocity_mag < 0.5:
        landing_one_shot = 200.0

    landing_reward = approach_reward + landing_one_shot

    # --- total ---
    total_reward = (distance_reward +
                    velocity_penalty +
                    attitude_penalty +
                    fuel_penalty +
                    landing_reward)

    components = {
        "distance": distance_reward,
        "velocity": velocity_penalty,
        "attitude": attitude_penalty,
        "fuel": fuel_penalty,
        "landing": landing_reward,
    }

    return float(total_reward), components
```
