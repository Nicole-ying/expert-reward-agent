# Response Record

Let me analyze the situation carefully.

## Diagnosis

**1. What happened?**
- Score: 251.91 (down from best 269.69 in iter 4)
- Episode length: 1195.85 (good, not crashing)
- The agent is moving and surviving, but score dropped after switching from `alive_bonus` to `gait_bonus`.

**2. Which component is the issue?**
- `progress_reward` ratio = 1.0 (main signal, fine)
- `stability_penalty` ratio = -0.067 (very small, fine)
- `gait_bonus` ratio = 0.576 — this is significant! It's contributing 57.6% of the main signal's magnitude.
- `gait_bonus` nonzero_rate = 0.997 — almost always active.

The problem: `gait_bonus` is too strong and always-on. The agent gets rewarded for alternating leg contact (which is good), but this reward is nearly constant (0.997 nonzero rate) — meaning the agent is already doing it naturally, and the bonus doesn't differentiate good vs bad behavior. It's just free reward that dilutes the learning signal.

**3. What did I change last time?**
- Iter 4 had `alive_bonus` (ratio 0.455) + `progress_reward` (ratio 1.180) + `stability_penalty` (-0.025) → score 269.69
- Iter 6 replaced `alive_bonus` with `gait_bonus` (ratio 0.576) → score 251.91

The `alive_bonus` version scored higher. The `gait_bonus` is actually worse — it's a constant reward that doesn't drive improvement.

**Best score was 269.69 with alive_bonus + progress_reward + stability_penalty.** I should revert to that skeleton but make a meaningful change.

Looking at the best configuration (iter 4):
- `alive_bonus` ratio = 0.455
- `progress_reward` ratio = 1.180
- `stability_penalty` ratio = -0.025

The `alive_bonus` at 0.455 is reasonable — it keeps the agent alive. But the score plateaued around 269. Let me think about what to improve.

The `progress_reward` is just `2.0 * forward_velocity`. This is a simple linear reward. Let me try a **bounded proximity** form for the progress signal to make it more robust, and keep the alive_bonus + stability_penalty.

Actually, let me think more carefully. The `original_env_reward` ratio is 0.202 (positive, same direction as progress_reward) — so alignment is fine.

The issue is that the agent has plateaued around 250-270. Let me try improving the progress signal's mathematical form. Instead of raw linear velocity, I'll use a bounded function that gives diminishing returns at high speed (to prevent the agent from just going fast and falling) and keeps the signal well-conditioned.

Let me also reconsider the gait bonus — it's not helping. Let me go back to alive_bonus but improve the progress signal.

**Plan:**
1. Revert to `alive_bonus` + `progress_reward` + `stability_penalty` (best skeleton)
2. Change the progress signal from linear `2.0 * velocity` to a bounded form: `progress_reward = 3.0 * tanh(forward_velocity)` — this gives smooth gradients, bounded output, and doesn't encourage reckless speed
3. Keep alive_bonus moderate and stability_penalty light

Let me verify this makes sense.

With `tanh(velocity)`, when velocity is ~1.0, tanh(1.0) ≈ 0.76. Scale 3.0 gives ~2.28. When velocity is 2.0, tanh(2.0) ≈ 0.96, scale 3.0 gives ~2.88. So the reward is bounded between 0 and 3, with diminishing returns at high speed — encouraging steady movement rather than speed at all costs.

Let me also check: the best score had progress_reward ratio of 1.180 with coefficient 2.0. Mean progress_reward was... let me estimate. If ratio_to_progress = 1.180 and progress_reward is the reference, then the mean of progress_reward itself... Actually ratio_to_progress_reward for progress_reward is always 1.0 (it's the reference). The 1.180 in iter 4 was the ratio of... wait, let me re-read.

Looking at iter 4: `progress_reward=1.180` — this is the mean value, not the ratio. So mean progress_reward was 1.180 with coefficient 2.0, meaning mean velocity was ~0.59.

With tanh scaling of 3.0, at velocity 0.59, tanh(0.59) ≈ 0.53, so reward ≈ 1.59. That's a bit higher than before. Let me use 2.5 * tanh(velocity) instead: at v=0.59, reward ≈ 1.33. Close to the original range.

Actually, let me just use a reasonable coefficient and let the training find the right behavior.

Let me also check: should I add a small alive_bonus? The best config had alive_bonus ratio 0.455 relative to progress. If progress mean is ~1.18, alive_bonus mean was ~0.54. With coefficient 0.5, that makes sense.

Let me go with:
- `progress_reward = 2.5 * tanh(forward_velocity)` — bounded, smooth, diminishing returns
- `alive_bonus = 0.5` — keep agent alive
- `stability_penalty = -0.5 * hull_angle^2 - 0.1 * angular_vel^2` — same as before

This is a meaningful change (changed mathematical form of progress signal from linear to bounded tanh) while reverting to the best skeleton.

## 诊断

**1. 发生了什么？**
- 得分从 best 269.69 (iter 4) 降到 251.91 (iter 6)
- Episode length 1195.85，没有 crash，agent 在正常运动
- 但得分退步了

**2. 哪个组件是主要原因？**
- `gait_bonus` ratio = 0.576，nonzero_rate = 0.997 → 几乎总是激活，相当于一个恒定的免费奖励
- 这个 bonus 不区分好行为和坏行为，只是白送 57.6% 的额外信号，稀释了主学习信号
- 对比 iter 4 的 `alive_bonus` ratio = 0.455，得分更高 → `alive_bonus` 比 `gait_bonus` 更有效

**3. 我上次改了什么？**
- iter 4（best 269.69）：`alive_bonus` + `progress_reward` + `stability_penalty`
- iter 6（当前 251.91）：把 `alive_bonus` 换成了 `gait_bonus` → 得分下降
- **结论：回到 alive_bonus 骨架，但做有意义的改进**

**改进方向：**
当前 `progress_reward = 2.0 * forward_velocity` 是线性无界信号。agent 可能为了追求高速度而牺牲稳定性。改用 bounded 形式 `tanh(velocity)` 让信号有界、梯度平滑、高速时收益递减——鼓励稳定前进而非疯狂加速。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励（bounded 形式，高速收益递减）
    forward_velocity = next_obs[2]  # horizontal_velocity
    # tanh 将速度映射到 (0,1)，系数 2.5 使典型速度 ~0.6 时奖励 ~1.3
    progress_reward = 2.5 * (forward_velocity / (1.0 + abs(forward_velocity)))
    
    # 存活奖励：鼓励 agent 保持存活状态
    alive_bonus = 0.5
    
    # 稳定性约束：惩罚身体倾斜和角速度
    hull_angle = next_obs[0]  # hull_angle
    hull_angular_velocity = next_obs[1]  # hull_angular_velocity
    stability_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_velocity ** 2)
    
    # 总奖励
    total_reward = progress_reward + alive_bonus + stability_penalty
    
    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components
```
