# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander: approach target, decelerate near pad, keep level,
    settle softly on both legs.  Uses only observable signals; no info/terminal flags.
    """
    # ── state extraction ──
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ── 1. goal proximity: improvement delta ──
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5
    distance_progress = dist_old - dist_new

    # ── 2. velocity damping: quadratic penalty with distance-gated weight ──
    # Weight ramps from ~0.01 (far away, encourage fast approach) to ~0.15 (near
    # target, enforce soft touchdown).  Uses rational scaling instead of linear
    # division to give a smoother transition zone.
    speed_sq = vx ** 2 + vy ** 2
    proximity_weight = 0.01 + 0.14 / (1.0 + 3.0 * dist_new ** 2)
    velocity_damping = -proximity_weight * speed_sq

    # ── 3. orientation penalty: quadratic on angle + angular velocity ──
    orientation_penalty = -0.08 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 4. soft-landing bonus: joint-condition proxy with rational factors ──
    # All three factors use 1/(1 + k * x^2) — rational decay, gentler tails than
    # exponential, less prone to collapsing to zero far from the target.
    both_contact = left_contact * right_contact          # binary 0/1
    dist_factor = 1.0 / (1.0 + 5.0 * dist_new ** 2)
    vel_factor  = 1.0 / (1.0 + 10.0 * speed_sq)
    angle_factor = 1.0 / (1.0 + 15.0 * angle ** 2)

    soft_landing = 0.5 * both_contact * dist_factor * vel_factor * angle_factor

    # ── combine ──
    total = distance_progress + velocity_damping + orientation_penalty + soft_landing

    components = {
        "distance_progress": distance_progress,
        "velocity_damping": velocity_damping,
        "orientation_penalty": orientation_penalty,
        "soft_landing_bonus": soft_landing,
    }

    return float(total), components
```

# reward_v1 设计说明

**selected task_family / dynamics_subtype**  
`navigation_goal_reaching` / `goal_approach_and_soft_contact` – 飞行器必须从上方某处出发，在目标垫上稳定着陆。

**selected reward roles**  
根据 `reward_role_decomposition` 选择了四个职责，覆盖主驱动和安全约束，省略了 v1 暂不需要的效率项。

| role | 实现组件 | 信号来源 |
|------|----------|----------|
| `goal_proximity` (mandatory) | `distance_progress` | `obs[0:2]`, `next_obs[0:2]` |
| `velocity_damping` (mandatory) | `velocity_damping` | `next_obs[2:4]`, `next_obs[0:2]` |
| `orientation_penalty` (mandatory) | `orientation_penalty` | `next_obs[4:6]` |
| `early_settlement_bonus` (conditional) | `soft_landing_bonus` | `next_obs[0:4]`, `next_obs[4]`, `next_obs[6:8]` |

**role‑to‑signal mapping 与公式算子**  

- **goal_proximity** → `improvement_delta`：`dist_old - dist_new`，每步提供正向梯度，推动飞向原点。这是唯一正确的选择 —— `-distance`（sample_00/04）永远为负且无方向性梯度。  

- **velocity_damping** → `dense_state_signal`（二次惩罚 + 距离门控权重）：`-proximity_weight * speed_sq`，其中 `proximity_weight = 0.01 + 0.14/(1 + 3*dist²)`。与原始 sample_02 的 `speed/(1+dist)` 不同，这里使用：
  - **二次速度惩罚** (`speed_sq`) 而非线性 —— 轻微速度几乎不受罚，高速受到强罚（凸化形式）
  - **有理距离门控** (`1/(1+3*dist²)`) 而非线性除法 —— 过渡区更平滑
  - 远距离时权重 ≈0.01（鼓励快速接近），近目标时权重 ≈0.15（强制减速）

- **orientation_penalty** → `quadratic_penalty`：`-0.08*angle² - 0.04*angvel²`。比原始 sample_02 稍轻的权重（0.08 vs 0.1, 0.04 vs 0.05），减少对导航阶段的干扰。

- **early_settlement_bonus** → `joint_condition_proxy`（有理衰减因子）：用三个有理函数 `1/(1+k*x²)` 构造 soft-landing 信号。与原始 sample_02 的指数形式 `exp(-x²/σ)` 的关键区别：
  - **有理函数尾部更重**：远离零点时衰减更慢，提供更远的梯度引导
  - **不易塌缩**：即使一个因子很小（如距离远），其他因子仍能维持非零梯度
  - **无需求幂运算**：`1/(1+k*x²)` 比 `2.718**exp` 计算更稳定

**与原 sample_02 的关键差异**

| 维度 | 原 sample_02 | 新版本 |
|------|-------------|--------|
| goal_proximity | `dist_old - dist_new` (same) | 不变 — 这是正确的选择 |
| velocity_damping 形式 | 线性 `speed/(1+dist)` | 二次 `speed_sq` + 有理距离门控 |
| velocity_damping 远距离行为 | 线性衰减 | 二次衰减（更轻的远距离惩罚） |
| soft_landing 因子族 | 指数 `exp(-x²/σ)` | 有理 `1/(1+kx²)` |
| soft_landing 尾部行为 | 快速衰减到 0 | 缓慢衰减，保持远距离梯度 |
| orientation 权重 | 0.1 / 0.05 | 0.08 / 0.04 |

**excluded roles 及原因**  

- `fuel_efficiency`：v1 默认先学会合理飞行和着陆，燃油效率优化留到后续。sample_00 加入后 active_rate 仅 9.5%，agent 不敢点火。  
- `time_penalty`：缺少步骤数/剩余时间信号，且未授权使用 `training_progress`，不可实现。  
- `explicit_success_reward`：`info` 为空，无可用的 success/failure flag，不能依赖。  

**为什么没有 terminal_success_reward / terminal_failure_penalty**  
环境卡片明确 `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`，`info` 固定为空。任何依赖 `info["success"]` 或 `info["termination_reason"]` 的写法都会失败，因此完全放弃稀疏终端奖励，转而用稠密的 `soft_landing_bonus` 近似成功状态。

**留到后续迭代的职责**  

- `fuel_efficiency`：发动机点火惩罚（离散动作代价）。  
- 更精细的健康门控（如 `soft_health_gate` 从姿态/角速度构建，在主奖励恶化时乘性抑制）。  
- 动态课程权重（`curriculum_weighting`）用于 late‑stage 精细控制。  

**训练后应重点观察的 failure modes**  

1. **有理因子梯度过弱**：`1/(1+kx²)` 在远离零点时梯度较小（比指数更平），可能导致 soft_landing_bonus 的早期引导不足。应检查 active_rate 是否显著低于原版的 43.5%。  
2. **二次速度惩罚过轻**：远距离时 `speed_sq * 0.01` 可能不足以抑制极端速度，导致飞出边界。应检查 terminated 率。  
3. **悬停不前**：`distance_progress` 和速度阻尼冲突，agent 可能宁可静止也不接近目标。需观察 episode_length 是否异常长。  
4. **单腿支撑/不稳定**：`soft_landing_bonus` 只奖励双腿接触，若 agent 满足于单腿可能无法获得 bonus。  
5. **过早坠落到垫外**：agent 未控制水平漂移，需结合 `goal_proximity` 和速度阻尼观察。
