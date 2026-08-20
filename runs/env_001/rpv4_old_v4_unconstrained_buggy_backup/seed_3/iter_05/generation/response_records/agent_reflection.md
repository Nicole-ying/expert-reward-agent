# Response Record

## 诊断报告

### Evidence

| 维度 | iter_2 (v2, best) | iter_4 (v4, current) | 解读 |
|---|---|---|---|
| score | **72.90** | **-109.53** | v4 崩溃到比 v1 还差 |
| len | 253 | 68.55 | v4 agent 快速坠毁，不尝试停留 |
| contact_bonus ep_sum | 2980 | 142.4 | v2 可积累，v4 固定不变 |
| contact_bonus cp10%→cp100% | **125 → 2820** (增长 22x) | **162 → 142** (零增长) | v4 的一刀切奖励完全不随学习进步 |
| velocity_penalty ep_sum | -4.6 | -7.9 | v4 速度惩罚略高但因 episode 太短总量仍低 |

**Checkpoint 趋势对比（关键证据）**：

- **v2**：contact_bonus 从 cp10% 的 125 一路增长到 cp80% 的 5895，score 从 -72.9 涨到 +80.5。Agent 学会越待越久，奖励自然增长。
- **v4**：contact_bonus 在所有 checkpoint 中锁定在 142-162 区间，完全平坦。一刀切 +200 奖励无法区分 crash-landing 和 soft-landing，agent 学到的最优策略是"最快速度撞上平台拿 200 然后结束"。

### Behavior Diagnosis

**当前行为**：Agent 学会了高速俯冲撞击平台。len=68.55（远低于 v2 的 253），所有 episode 在 <150 步内终止。Agent 以最快速度冲向目标区域，触发一次 contact_bonus（+200），然后 episode 因 crash/body_settle 结束。因为没有停留激励（per-step settling 只有 +2，被速度惩罚淹没），agent 不做任何减速或姿态控制。

**根因**：v4 的 **一刀切 transition bonus** 创造了平坦的奖励地形。不管着陆质量如何，只要触地就拿 200。Agent 没有任何梯度去改善着陆质量。相比之下，v2 的 per-step bonus 天然随停留时间增长，形成了从"碰到就走"到"稳稳停住"的完整课程。

**为什么之前的修改（v3→v4）失败**：v3 试图修 goal_proximity（改 potential-based），但真正的问题是 contact_bonus 的 99.6% 主导。v4 试图用一刀切来解决主导问题，但杀死了奖励的缩放性——agent 失去了改善动力。

### 干预方案

**L2 修改**：将 contact_bonus 从"一刀切 transition + 微弱 settling"**回退到 per-step 奖励**（v2 骨架），同时：
- contact_bonus 权重从 100 → **25**（解决 99.6% 主导问题）
- velocity_penalty 权重从 0.15 → **0.4**（强化减速信号，防止高速撞击）
- goal_proximity 和 orientation_penalty 保持 v2/v4 的稳定形态

这保持单一语义方向：**让 contact 奖励随着陆质量缩放**。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5: per-step contact bonus (v2 skeleton) with reduced weight + stronger velocity damping

    Key diagnosis from v4 failure:
    - v4's one-time transition bonus (+200) created a flat reward landscape:
      contact_bonus stayed at ~142 across ALL checkpoints with zero growth.
      Agent learned "dive fast, grab bonus, die" (len=68.55, all early terminate).
    - v2's per-step bonus grew from 125→2820 during training — reward scaled
      naturally with landing quality, creating a curriculum from touch to settle.

    Changes from v4:
    1. contact_bonus: revert from one-time transition (+200) + per-step settling (+2)
       back to pure per-step bonus (v2 approach). Weight reduced 100→25 to address
       the 99.6% magnitude dominance that plagued v2.
    2. velocity_penalty: strengthened 0.15→0.4 (v4 had 0.15) to provide meaningful
       gradient against high-speed approaches, now that contact_bonus is smaller.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: direct distance penalty (v2/v4 approach, proven stable) ---
    # Consistent gradient toward (0,0) at every step.
    w_dist = 0.05
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = -w_dist * dist_nxt

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Strengthened (0.15→0.4) to deter high-speed crashes now that contact_bonus
    # is moderate. Gate ensures far-field maneuvering isn't overly penalized.
    w_v = 0.4
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Keep lander upright, suppress wobble. Unchanged from v2/v4.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Reverted from one-time transition (v4) back to per-step (v2 skeleton).
    # Weight 25 (vs v2's 100) keeps contact as the anchor signal while giving
    # velocity/orientation penalties room to shape behavior.
    # Per-step design means longer settled stays → higher reward, creating
    # natural curriculum from first touch to stable landing.
    w_contact = 25.0
    near_target = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    contact_bonus = w_contact if (near_target and both_legs) else 0.0

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components
```
