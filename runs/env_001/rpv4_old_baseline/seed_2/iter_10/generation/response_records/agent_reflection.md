# Response Record

根据训练反馈，当前奖励函数的主要问题是：**score 始终为负（-10.1），与目标差距巨大**。原因在于：

- **landing_reward** 虽然 magnitude_share 极高（79.7%），但 active_rate 仅有 2.6%，几乎只在极短暂的着陆瞬间触发。大量 episode 没有机会获得该奖励。
- 而 **fuel_penalty** 和 **time_penalty** 几乎贯穿全程（active rate 72.3% 和 100%），每步持续扣分，轻松抵消掉成功 episode 的 landing reward，导致平均 score 为负。
- **approach_reward** 信号太弱（weight=5 导致平均每步只有 0.02），无法有效引导 agent 向目标移动。
- **speed_penalty** 对所有速度一视同仁，从一开始就惩罚必要的飞行运动，阻碍 agent 尝试接近目标。
- 整体上，负奖励的累积压倒了正奖励，agent 被“罚怕”而无法学会着陆。

因此，按照反思规则 (1)，需要**重写奖励框架**——把主力信号改为**可持续的接近目标正奖励**，并大幅削弱或移除远期惩罚，只在着陆阶段加入软着陆约束。

**修改原则**：
1. 将 `approach_reward` 权重增大（×30），使其成为主要的步级引导信号。
2. 将 `speed_penalty` 改为**仅在接近目标时生效**，鼓励远距离快速移动、近距离减速软着陆。
3. 保留 `landing_reward` 作为最终成功的大额奖励，并有限提高基础值，降低其苛刻程度。
4. `fuel_penalty` 下调到极轻（‑0.001），基本不起惩罚作用，不再阻碍引擎使用。
5. 完全移除 `time_penalty`。

以下为修改后的奖励函数：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    old_x, old_y = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    old_dist = (old_x ** 2 + old_y ** 2) ** 0.5
    new_dist = (x ** 2 + y ** 2) ** 0.5

    # Strong approach shaping: reward moving closer to target
    approach_reward = (old_dist - new_dist) * 30.0

    # Speed / angle penalty only when near the target, to encourage soft landing
    proximity_threshold = 1.5
    if new_dist < proximity_threshold:
        speed_penalty = -0.5 * (vx ** 2 + vy ** 2) - 0.3 * angvel ** 2 - 0.3 * angle ** 2
    else:
        speed_penalty = 0.0

    # Landing bonus: large fixed bonus minus quality deficits
    if left_contact > 0.5 and right_contact > 0.5:
        landing_quality = 50.0 - 20.0 * angle ** 2 - 10.0 * (vx ** 2 + vy ** 2) - 10.0 * angvel ** 2
        landing_reward = max(0.0, landing_quality)
    else:
        landing_reward = 0.0

    # Negligible fuel penalty, not dominating
    fuel_penalty = -0.001 if action in [1, 2, 3] else 0.0

    total = approach_reward + speed_penalty + landing_reward + fuel_penalty

    components = {
        "approach_reward": approach_reward,
        "speed_penalty": speed_penalty,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty,
    }
    return float(total), components
```
