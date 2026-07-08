# Response Record

## 诊断

### 0. 环境理解
CartPole 生存任务。`obs = [cart_pos, cart_vel, pole_angle, pole_angvel]`。目标是保持杆子直立 + 小车居中，每步存活获得 `original_env_reward=1.0`。最大 episode 长度 500。

### 1. Agent 发生了什么？
Agent 在 iter 2 已经满分（score=500, len=500），iter 3 添加 `survival_bonus=0.005` 后分数不变。**这不是 crash 也不是徘徊——任务已解决。** 问题在于奖励函数本身已经退化：

- `progress_reward` mean = **-0.000312**，几乎是零。势能塑形 `cost(obs)-cost(next_obs)` 在稳态时趋近于零——agent 已平衡，cost 不再变化。
- `survival_bonus` mean = 0.005，是常数，**梯度为零**。它在数值上主导了 total_reward（mean=0.00436），但完全没有学习价值。
- 奖励函数本质上是「常数偏置 + 微小噪声」。Agent 能满分靠的是环境自身的 `original_env_reward=1.0/step` 驱动，而非我们的 shaped reward。

### 2. 哪个组件是主因？
**`progress_reward` 的数学形态。** 势能塑形在稳态时梯度消失——这是它固有的性质，不是系数问题。`survival_bonus` 只是无关痛痒的常数偏置，ratio=5.0 但 nonzero_rate=100%，它无害但也无用。

### 3. 我上一轮改了什么？
Iter 2→3：添加了 `survival_bonus=0.005`。得分从 500 到 500，零改善。这次不再碰 survival_bonus，改 progress_reward 的数学形式。

---

**修改方案（层次 2：改数学形式）：** 把 progress_reward 从势能塑形 `cost(obs)-cost(next_obs)` 改为 bounded proximity `1/(1+k*dist)`。后者在最优解处仍提供正向梯度（越接近 0 奖励越高但永不饱和），且天然 bounded 在 [0,1]，不会数值爆炸。只改这一个组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward — v4.

    Diagnosis (iter 3): potential-based progress_reward collapsed to near-zero
    at steady state (mean -0.0003).  survival_bonus (0.005, constant) dominated
    total but provides zero gradient.  Agent solved via env's own 1.0/step,
    not via our shaped reward.

    Change (v4): switch progress_reward from potential-based shaping
      scale * (cost(obs) - cost(next_obs))
    to bounded proximity:
      scale * (w_angle * 1/(1+10*|angle|) + w_pos * 1/(1+5*|pos|))

    Bounded proximity provides gradient everywhere — even at the optimum,
    there is always incentive to get closer.  Naturally bounded in [0,1]
    per factor, preventing numerical issues.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.01   # calibrated for bounded proximity range ~0.07-0.11

    # unchanged from iter 2/3
    w_vel = 0.001
    w_angvel = 0.001
    survival_bonus = 0.005

    # -- bounded proximity progress reward --
    # Each factor ∈ (0, 1]; product with weight gives max w_angle at angle=0
    angle_proximity = 1.0 / (1.0 + 10.0 * abs(next_obs[2]))
    pos_proximity   = 1.0 / (1.0 + 5.0 * abs(next_obs[0]))
    progress_reward = scale_progress * (w_angle * angle_proximity + w_pos * pos_proximity)

    # -- light stability penalty (unchanged) --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    # -- total --
    total_reward = progress_reward + stability_penalty + survival_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "survival_bonus": survival_bonus,
    }

    return float(total_reward), components
```
