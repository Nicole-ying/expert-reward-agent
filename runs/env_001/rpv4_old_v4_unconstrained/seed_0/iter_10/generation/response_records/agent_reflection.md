# Response Record

数据已经非常清晰。让我进行完整诊断。

---

## 诊断报告

### Evidence

| 证据来源 | 关键数据 |
|---|---|
| **v9 final score** | 185.79（目标200），len=**680.1**（过长），terminated=11/20，truncated=**9/20**（45%超时） |
| **v9 组件份额** | proximity=**56.5%**（289.5），contact=39.9%（204.2），progress=**0.5%**（2.5），vel_damping=0.3%（-1.4） |
| **progress 全程微弱** | 所有 checkpoint 在 0.26~2.70 之间，与 proximity（60~415）差 **100倍** |
| **cp70% contact 骤升** | contact 从 0→241.9（首次发现地面），但 cp80% 暴跌至 43.3 → 训练不稳定，agent 在"长时间贴地"和"短时着陆"间摇摆 |
| **vel_damping 几乎死亡** | active_rate 从 cp10% 的 46% 跌至 cp30%-cp60% 的 6-10%，最终仅 19.5%，ep_sum=-1.4 — agent 学会了无视速度惩罚 |
| **v4-v8 教训** | 离散 landing_bonus 全部被 exploit（v4: 97.4% magnitude，v8: 94.7%），v9 用连续 proximity 解决了 exploit，但引入了新问题：hovering |
| **cp80% 峰值** | score=220.9（已超目标），但无法维持 → 奖励结构不支持稳定收敛 |

### Behavior Diagnosis

**当前策略行为：慢速徘徊 + 偶尔着陆。**

v9 成功消除了离散 exploit，但创造了"hovering equilibrium"：agent 发现只需待在着陆垫附近（dist≈0.5~1.0）就能稳定收取 proximity_reward（~0.25~0.37/step），无需真正着陆。这解释了：
- **len=680**：agent 在空中徘徊收集 proximity
- **9/20 truncated**：徘徊过久导致超时
- **progress 微弱**：agent 没有动力快速接近——progress 每步贡献 ~0.004，proximity 贡献 ~0.4，相差 100 倍
- **vel_damping 无效**：agent 学会了在低速下运行以规避惩罚（safe_speed=0.25 太容易满足）

**核心矛盾**：proximity 是「存量」奖励（只要靠近就能无限累积），progress 是「流量」奖励（接近后就用完了）。当存量信号 100 倍于流量信号时，agent 理性选择是慢慢蹭过去。

### 干预方案

1. **progress_reward ×5**：2.0 → **10.0**。使"靠近"的每步边际收益从 ~0.004 提升到 ~0.02，与 proximity 可比。
2. **proximity 上限降低**：0.5 → **0.35**。减少 hovering 收益，缩小存量/流量差距。
3. **vel_damping 加强+加宽**：系数 -0.5→**-1.5**，near_pad sigma 0.5→**1.0**。使速度惩罚更早激活、更强力。
4. **新增 descent_bonus**：当水平对准着陆垫时奖励向下速度。直接对抗 hovering——悬停时此项为 0，下降时为正。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v10: Stronger progress + descent incentive to fix hovering.

    ROOT CAUSE OF v9 HOVERING:
    progress_reward (0.5% share) was 100x weaker than proximity_reward (56.5% share).
    Agent could collect ~0.3/step proximity just by staying near pad, vs ~0.004/step
    from progress. Rational choice: hover indefinitely, occasionally land.

    v10 CHANGES:
    - progress_reward: 2.0 → 10.0 (5x), makes "get closer" the dominant per-step signal
    - proximity_reward: 0.5 → 0.35 max, reduces hovering incentive while keeping gradient
    - vel_damping: wider sigma (1.0), stronger coefficient (1.5), prevents crash dives
    - NEW descent_bonus: rewards downward velocity when horizontally aligned with pad.
      Directly punishes hovering: 0 when hovering, positive when descending.
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    px = obs[0]
    py = obs[1]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ── 1. Progress: strong navigation drive (5x boost from v9) ──
    # v9: 2.0 gave ~2.5 ep_sum vs proximity's 289.5 → 100:1 ratio
    # v10: 10.0 targets ~12-15 ep_sum vs proximity's ~150 → 10:1 ratio
    progress_reward = 10.0 * (prev_dist - dist)

    # ── 2. Proximity: continuous gradient, moderated to reduce loitering ──
    # Reduced from 0.5 to 0.35 max. At center with contact: 0.35 + 0.6 = 0.95/step.
    # At dist=0.5: 0.35*exp(-0.25/0.8)=0.256. Far away (dist=2): ~0.002.
    proximity_reward = 0.35 * (2.718281828 ** (-dist ** 2 / 0.8))

    # ── 3. Orientation penalty: stable, proven, unchanged ──
    orientation_penalty = -0.15 * (angle ** 2) - 0.06 * (angvel ** 2)

    # ── 4. Velocity damping: wider activation + stronger coefficient ──
    # v9: sigma=0.5, coeff=0.5 → only active very close to pad, easily ignored
    # v10: sigma=1.0, coeff=1.5 → activates at dist=1.0 with 37% strength,
    #   creates natural braking gradient that counters descent_bonus near pad
    near_pad = 2.718281828 ** (-dist ** 2 / 1.0)
    safe_speed = 0.25
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -1.5 * near_pad * excess_speed

    # ── 5. Descent incentive: reward downward velocity when above pad ──
    # Key anti-hovering signal. When agent is horizontally aligned and descending,
    # this adds 0.1-0.5/step. When hovering (vy≈0), this is ~0.
    # Above_pad gates on horizontal alignment: at |x|=0.5 → 61%, at |x|=1.0 → 14%.
    # Natural balancing with vel_damping near pad:
    #   dist=0.5, speed=0.8: vel_damping=-0.64 vs descent=+0.40 → net negative → slow down
    #   dist=1.0, speed=1.0: vel_damping=-0.41 vs descent=+0.50 → net positive → descend OK
    above_pad = 2.718281828 ** (-x ** 2 / 0.5)
    descent_bonus = 0.5 * above_pad * max(0.0, -vy)

    # ── 6. Contact: ground interaction discoverability, unchanged ──
    contact_reward = 0.3 * (left_contact + right_contact)

    # ── 7. Action efficiency: gentle penalty, unchanged ──
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.04
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.02

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + proximity_reward
        + orientation_penalty
        + vel_damping
        + descent_bonus
        + contact_reward
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "descent_bonus": descent_bonus,
        "contact_reward": contact_reward,
        "action_reward": action_reward,
    }
    return float(total_reward), components
```
