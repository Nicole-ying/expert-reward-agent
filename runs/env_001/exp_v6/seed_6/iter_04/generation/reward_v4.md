## 诊断

### 1. Agent 发生了什么？

- **Score 129.76**，比 iter 2 的 156.45 下降了 26.69。
- Episode length 829.80，没有 crash，没有 early terminal。
- 这是一个「徘徊型」agent：它学会了飞到目标附近并悬停，但没有变得更精准。上一轮收紧阈值（dist 0.20→0.12, speed 0.10→0.05, angle 0.05→0.03）**没有降低 nonzero_rate（仍是 62%）**，反而让分数退步了。

### 2. 哪个组件是主因？

- **soft_landing_proxy**：mean=0.062，ratio_to_progress=**23x**，nonzero_rate=62%。它完全主导了 total_reward。progress 只有 0.003。
- 这是一个典型的 **binary proxy 过松** 问题：0.1 的固定奖励在 62% 的 step 上触发，agent 学会了「在宽松条件附近悬停」而非「精准着陆」。收紧阈值没有用——因为 agent 本来就已经在阈值范围内活动了。
- **根本问题不是阈值松紧，而是数学形态**：二值奖励不区分「勉强满足条件」和「完美着陆」，没有梯度引导 agent 做得更好。

### 3. 我之前改了什么？

- Iter 3：收紧 soft_landing_proxy 的三个阈值（层次 1）。结果：分数下降，nonzero_rate 未变。**收紧阈值这条路已经证明无效，不能再走。**

### 修改方案

**层次 2：改数学形式**——把 `soft_landing_proxy` 从二值触发（`if ... else 0`）改为连续乘积因子（`dist_factor * speed_factor * angle_factor * contact_factor`）。使用 `1/(1+kx)` 形式让每个因子 bounded 在 [0,1]，乘积提供平滑梯度。系数从 0.1（二值）调整为 0.15（连续 max），这是形态变化的自然结果。

预期效果：
- nonzero_rate → ~100%（但值域连续），mean 预计降至 0.01–0.03
- ratio_to_progress 从 23x 降至 ~3–10x
- 梯度引导 agent 从「勉强悬停」进化到「精准着陆」

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract states ----------
    # obs
    x, y = obs[0], obs[1]
    # next_obs
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. main learning signal: progress toward (0,0) ----------
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress_delta_reward = dist_obs - dist_next

    # ---------- 2. stability / smoothness penalty ----------
    vel_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    ang_vel_penalty = abs(ang_vel)

    w_vel = 0.001
    w_angle = 0.005
    w_angvel = 0.001
    stability_penalty = - (w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (CONTINUOUS) ----------
    # DIAGNOSIS: binary trigger at 0.1 with 62% nonzero_rate dominates progress 23x.
    # Agent exploits loose binary by hovering, not landing precisely.
    # FIX: replace binary if/else with continuous product of bounded factors.
    # Each factor uses 1/(1+kx) — smooth, bounded [0,1], gradient everywhere.
    # Coefficient 0.15 at max; typical hovering ~0.01-0.02, ratio ~3-7x progress.

    # distance: 1/(1+10*dist) — 0.5 at dist=0.1, 0.09 at dist=1.0
    dist_factor = 1.0 / (1.0 + 10.0 * dist_next)

    # speed: 1/(1+5*(|vx|+|vy|)) — 0.5 at total_speed=0.2, 0.17 at 1.0
    speed_factor = 1.0 / (1.0 + 5.0 * (abs(vx) + abs(vy)))

    # angle: 1/(1+20*|angle|) — 0.5 at |angle|=0.05, 0.33 at 0.1
    angle_factor = 1.0 / (1.0 + 20.0 * abs(angle))

    # contact: soft factor — 0.5 with no contact, 1.0 with both legs
    contact_factor = 0.5 + 0.25 * (left_contact + right_contact)

    soft_landing_proxy = 0.15 * dist_factor * speed_factor * angle_factor * contact_factor

    # ---------- total ----------
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy

    # ---------- components dict ----------
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```