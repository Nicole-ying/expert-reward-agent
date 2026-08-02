# 设计理由

本轮是第 9 轮迭代，属于**正常模式**，仅修改一个组件。  

### 当前问题诊断

从训练反馈看，所有 episode 都在约 68 步内 early-terminated，score ≈ -114。  
最终策略产生的生成奖励总和仅约 5.26（landing 4.53 + progress_delta 1.18 − fuel 0.46），而环境内置的存活惩罚约 −1.65/step × 68 ≈ −112，完全压倒了生成奖励，导致 policy 无法学习任何有意义的行为，直接坠毁。  

检查组件信号：  
- `landing` 的 `approach_reward` 权重为 1.0，接近阶段各种 factor 经常在 0.2–0.5 之间，乘积后 per-step 贡献很小。  
- `progress_delta` 的门控 `gate = gate_angle * gate_vel * gate_angvel` 采用**三元乘积**，即使每个子门控下限为 0.1，乘积也会塌缩到 0.001，导致 `w_progress=8.0` 实际能产生的奖励极低（均值仅 0.017/step）。这使得正向距离改善信号几乎完全消失，agent 得不到任何减速/调姿的激励。  

### 修改内容

**改动组件：`progress_delta` 的门控结构**  
- **旧**：`gate = gate_angle * gate_vel * gate_angvel`（乘积，最小值 0.001）  
- **新**：`gate = (gate_angle * gate_vel * gate_angvel) ** (1.0/3.0)`（几何平均，最小值 0.1）  

### 变换依据

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 乘积门控 `f1*f2*f3`，最小值 0.001 | active_rate 高，但信号值极低，无法对抗环境惩罚 | **几何平均** `(f1*f2*f3)^(1/3)` | 提升门控下界 100 倍，同时保留“所有维度协调”的引导，不产生分离的持续获奖 |

### 系数校准

- 门控几何平均后最小值为 **0.1**（原来 0.001），`progress_delta` per-step 在距离缩短时预期可达 **0.8 ∼ 4.0**，能够有效与环境惩罚竞争。  
- 不修改 `landing` 或 `fuel_penalty`，避免一次改变过多变量，方便验证单独的门控修改效果。  
- 惩罚负担（只有 fuel_penalty，active_rate 3.4%）很低，满足设计校准要求。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ------------------- unpack observations -------------------
    x,  y  = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle      = obs[4]
    angvel     = obs[5]
    left_leg   = obs[6]
    right_leg  = obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle  = next_obs[4]
    n_angvel = next_obs[5]
    n_left   = next_obs[6]
    n_right  = next_obs[7]

    # ------------------- helper quantities -------------------
    dist      = (x**2  + y**2)  ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    next_vel_abs  = (nvx**2 + nvy**2) ** 0.5

    # ------------------- thresholds & weights -------------------
    w_progress = 8.0
    w_fuel = 0.2

    th_angle  = 0.5
    th_vel    = 1.0
    th_angvel = 2.0

    w_approach   = 1.0
    w_touchdown  = 10.0

    # ------------------- 1. distance-improvement progress signal -------------------
    delta_dist = dist - next_dist   # positive when approaching target

    gate_min = 0.1
    gate_angle  = max(gate_min, 1.0 - abs(n_angle)  / th_angle)
    gate_vel    = max(gate_min, 1.0 - next_vel_abs   / th_vel)
    gate_angvel = max(gate_min, 1.0 - abs(n_angvel)  / th_angvel)
    # geometric mean keeps gate >= 0.1 even when one dimension is poor
    gate = (gate_angle * gate_vel * gate_angvel) ** (1.0/3.0)

    progress_delta = w_progress * max(0.0, delta_dist) * gate

    # ------------------- 2. landing (approach + touchdown) -------------------
    contact_next = (n_left + n_right) / 2.0

    pos_factor    = max(0.0, 1.0 - abs(nx) / 0.5)
    height_factor = max(0.0, 1.0 - abs(ny) / 0.5)
    vel_factor    = max(0.0, 1.0 - next_vel_abs / 0.5)
    angle_factor  = max(0.0, 1.0 - abs(n_angle) / 0.3)
    angvel_factor = max(0.0, 1.0 - abs(n_angvel) / 0.5)

    touchdown_reward = 0.0
    approach_reward  = 0.0

    if contact_next > 0.1:
        quality = pos_factor * height_factor * vel_factor * angle_factor * angvel_factor * contact_next
        touchdown_reward = w_touchdown * quality
    else:
        dist_factor   = max(0.2, 1.0 - next_dist / 1.5)
        h_factor      = max(0.2, 1.0 - abs(ny) / 1.5)
        vel_factor_a  = max(0.2, 1.0 - next_vel_abs / 0.8)
        angle_factor_a = max(0.2, 1.0 - abs(n_angle) / 0.5)
        angvel_factor_a = max(0.2, 1.0 - abs(n_angvel) / 1.0)
        approach_reward = w_approach * dist_factor * h_factor * vel_factor_a * angle_factor_a * angvel_factor_a

    landing = approach_reward + touchdown_reward

    # ------------------- 3. fuel penalty -------------------
    engine_on = 1.0 if action != 0 else 0.0
    fuel_penalty = -w_fuel * engine_on

    # ------------------- total reward -------------------
    total_reward = progress_delta + landing + fuel_penalty

    components = {
        'progress_delta':   progress_delta,
        'landing':          landing,
        'fuel_penalty':     fuel_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号齐全，但乘积门控导致 progress_delta 补偿系数实质上塌缩到 0.001 级别，使生成奖励远低于环境惩罚。
- **behavior**: agent 快速坠毁（len≈68），policy 完全被环境负奖励支配，无法学习减速或调姿。
- **signal**: progress_delta 的 gate 塌缩，缺乏有效的密集正向引导，无法对抗 −1.65/step 的环境惩罚。
- **level**: Level 2
- **hypothesis**: 几何平均门控使 progress_delta 的最小 per-step 奖励从 0.008 提升到 0.8，agent 将获得足够的密集正向反馈来学习减速、调姿并延长存活，从而为后续的 landing 组件创造接触机会。
- **risk**: 若几何平均使 reward 过于宽松，agent 可能在中度偏离目标时仍获得不错奖励，从而徘徊而延迟着陆；但 landing 组件会在接近条件下提供更强奖励，应能最终驱策着陆。