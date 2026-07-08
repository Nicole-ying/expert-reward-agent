## 诊断

### 1. Agent 发生了什么？
Score = 466/500（93.2%），episode_length = 466，early_terminal = 0。Agent 没有 crash，也没有徘徊——它存活得很好，但没能撑满 500 步。**这是一个被过度惩罚拖累的 agent**：它在正常运作，但稳定性惩罚占比过大，可能导致它在需要大幅 corrective action 时"犹豫"。

### 2. 哪个组件是主因？
- **stability_penalty**：ratio_to_progress_reward = **-3.16**，绝对值远超 0.5 警戒线。nonzero_rate = 100%，意味着每一步都在被惩罚速度。
- **progress_reward**：mean = -0.00124，轻微为负——agent 在稳态附近振荡，但平均来说状态在缓慢退化。信号太弱，被 penalty 淹没了。

根因：稳定性惩罚是主学习信号的 3 倍强，agent 收到的梯度主要来自"别动"而非"改善姿态"。虽然没 crash，但 penalty 压制了必要的 corrective movement，导致 agent 无法撑满 500 步。

### 3. 上轮改了什么？
这是第一轮（iter 1），没有上轮改动。本轮是首次修订。

### 修改方案
**层次 1：降低 stability_penalty 系数 10 倍。** 从 `w_vel=0.01, w_angvel=0.01` 降到 `0.001`，使 ratio 从 -3.16 降至约 -0.32，让 progress_reward 重新成为主导信号。不改 progress_reward 或其他组件，保持单一变更可归因。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 cartpole survival reward.
    Uses:
    - progress_reward: dense improvement towards stable upright and center.
    - stability_penalty: light penalty on linear/angular speeds.
    
    Iter 2 change: reduced stability_penalty weights 10x (0.01→0.001)
    because penalty ratio_to_progress was -3.16, drowning out the progress signal.
    """

    # -- hyperparameters --
    w_pos = 1.0
    w_angle = 10.0
    scale_progress = 0.1

    # Reduced 10x from 0.01 → 0.001 to fix penalty dominance (ratio was -3.16)
    w_vel = 0.001
    w_angvel = 0.001

    # -- cost helper --
    def cost(o):
        return w_pos * abs(o[0]) + w_angle * abs(o[2])

    # -- main progress reward --
    progress_reward = scale_progress * (cost(obs) - cost(next_obs))

    # -- light stability penalty (only on next step) --
    stability_penalty = -w_vel * abs(next_obs[1]) - w_angvel * abs(next_obs[3])

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```