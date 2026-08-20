# Response Record

现在我已掌握全部数据。以下是诊断和修改方案。

---

## 诊断报告

### evidence

| 数据点 | 数值 | 含义 |
|---|---|---|
| contact_guidance magnitude_share | **99.2%** | 几乎独占奖励信号 |
| progress magnitude_share | 0.3% | 对学习几乎无贡献 |
| terminated ratio | 7/20 (35%) | 多数 episode 无法完成 settling |
| episode length | 781 | 偏长，agent 在徘徊/逗留 |
| contact_guidance ep_sum | 252.86 | 在 ~530 active steps 中持续领取 |
| checkpoint cp70% 突变 | contact 0→248, score -16→106 | agent 突然发现 contact 奖励并大规模 exploit |
| cp80%→cp100% | contact↓ (271→167), score↑ (131→224) | 后期逐渐学会更快 settle，降低 contact 积累 |

### behavior_diagnosis

当前策略的行为模式：**"接触徘徊者"**。agent 学会了飞到目标附近并用双腿接触平台（68% active rate），但随后**逗留在接触状态持续领取 per-step 奖励**，而非快速完成 settling 终止 episode。这直接导致：

- 13/20 episode 被 truncate（达到最大步数仍未 settle）
- contact_guidance 累积到 252.86，占总奖励 99.2%
- progress 信号（0.3%）完全被淹没，无法提供导航引导

**根因**：`contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok` 是一个**纯状态奖励**——只要处于"近目标 + 低速度 + 有接触"状态，每一步都领取奖励。这正好匹配证据表模式：**"持续事件被重复领取 → 持续→状态转移事件"**。

### 干预方案（L2：单一语义修改）

**语义方向**：将 contact_guidance 从「持续状态奖励」改为「状态转移事件奖励」，消除逗留激励。同时适度提升 progress 权重以补偿减弱后的 contact 信号。

具体改动：
1. **contact_guidance**：拆分为极小 per-step 保底信号（`0.02 * contact_next * ...`）+ 新接触达成奖金（`5.0 * contact_gained * ...`），其中 `contact_gained = max(0, contact_next - contact_prev)` 只在接触从 0→1 的转移步触发
2. **velocity_ok**：从仅检查 vy 改为检查总速度 `(vx²+vy²)^0.5`，水平速度不稳同样无法安全着陆
3. **progress 权重**：`0.5 → 2.0`，提升导航引导力

---

## 修改后代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── distance to target (target at origin (0,0)) ──
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # ── velocity components ──
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # ── orientation penalty: body angle and angular velocity ──
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # ── contact guidance (transition-based: reward achieving contact, not lingering) ──
    contact_prev = float(obs[6] + obs[7])           # 0.0 .. 2.0
    contact_next = float(next_obs[6] + next_obs[7]) # 0.0 .. 2.0
    contact_gained = max(0.0, contact_next - contact_prev)  # fires only on 0→1 transitions

    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    # total speed must be low for safe contact (both horizontal and vertical)
    speed = (vx**2 + vy**2) ** 0.5
    velocity_ok = max(0.0, 1.0 - speed / 0.5)

    # (a) small per-step signal to avoid complete sparsity
    contact_dense = 0.02 * contact_next * proximity_factor * velocity_ok
    # (b) transition bonus: reward the act of making new contact
    contact_bonus = 5.0 * contact_gained * proximity_factor * velocity_ok

    contact_guidance = contact_dense + contact_bonus

    # ── combine all components ──
    total_reward = (2.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 2.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return float(total_reward), components
```

### 预期效果

| 指标 | 改前 | 预期改后 |
|---|---|---|
| contact_guidance ep_sum | 252.86 | ~15–30（不再积累） |
| contact_guidance magnitude_share | 99.2% | ~40–60% |
| progress magnitude_share | 0.3% | ~30–50% |
| terminated ratio | 35% | ↑（无逗留激励，更快 settle） |
| episode length | 781 | ↓（settle 更快） |

核心逻辑：agent 不再因为"保持接触"而持续获利——只有**达成新接触的瞬间**获得显著奖励。这会推动 agent 尽快完成双腿着陆并 settle，而非在接触状态中逗留。同时 progress 权重提升确保导航信号不被淹没。
