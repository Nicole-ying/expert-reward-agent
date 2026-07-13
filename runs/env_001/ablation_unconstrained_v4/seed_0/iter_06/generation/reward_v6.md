## 分析

### 1. Agent 发生了什么？

**Iter 5（当前）彻底失败**：score=-117.75, episode_length=68.35, **terminated=20/20**（所有episode都在150步内异常终止）。关键证据：

- `landing_reward` 的 **active_rate 仅 0.4%**——本质上从未触发。阈值过严（distance<0.3, speed<0.25, angle<0.15 rad 三者必须同时满足）形成了奖励悬崖，agent在早期探索中永远发现不了这个信号。
- `progress_delta`（+13.41 mean）和 `distance_penalty`（-19.95 mean）在对抗，但缺乏持续的近端正向引导，agent无法克服负向惩罚，快速撞毁。
- 这与搜索知识 `sparse_completion_proxy` 吻合：active_rate <1% 的一次性事件无法提供学习梯度。

**Iter 2（best, 108.12）相对成功**：episode_length=743，说明 `settle_reward`（+2.29 mean）提供了持续的近端正向信号，让agent能存活并靠近平台。但agent在目标附近徘徊743步却不彻底着陆——`settle_reward` 的阈值太宽松，让agent可以通过"hover"（近而不稳）来榨取奖励（对应 `high_reward_without_success` 症状）。

### 2. 哪个组件最值得干预？

核心矛盾在 **"近端引导信号"** 的形态：
- Iter 2 的 `settle_reward`：持续活跃→能引导接近，但过于慷慨→hover exploit
- Iter 5 的 `landing_reward`：阈值过严→完全稀疏，agent发现不了

**方案**：将单一的 settle/landing 奖励**拆分为两个层叠组件**——
- `proximity_reward`：纯距离的平滑函数 `w/(1+k*dist)`，**100%活跃**，提供永不消失的接近引导（避免iter 5的悬崖）
- `stable_landing_bonus`：仅在接近时由速度和角度软因子调制，作为**额外加成**而非单独门槛（避免iter 2的hover exploit，同时不给稀疏悬崖）

这从数学形态上不同于所有历史版本：iter 2 是单层 settle_reward，iter 5 是硬阈值乘积 landing_reward，本方案是**双层叠加**（平滑基座 + 门控加成）。

### 3. 历史修改回顾

| iter | 改动 | 结果 |
|------|------|------|
| 2 | 加入 settle_reward + velocity_damping | **108.12** (best)，但hover 743步 |
| 3 | settle_delta 替代 settle_reward | -121.20，快速失败 |
| 4 | settle_reward + landing_bonus 混合 | +14.06，部分恢复但不稳定 |
| 5 | 移除 settle_reward，硬阈值 landing_reward | **-117.75**，landing_reward 0.4%活跃，彻底失败 |

当前应**以 best（iter 2）结构为基础**，将 settle_reward 拆分重构为 proximity_reward + stable_landing_bonus 双层结构，同时保留 progress_delta、distance_penalty、velocity_damping 等已验证的辅助组件。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Splits iter 2's exploit-prone settle_reward into two-layer design:
    - proximity_reward: smooth distance-only signal (100% active, no cliff)
    - stable_landing_bonus: proximity-gated stability bonus (soft factors, extra credit)
    Prevents both the hover exploit (iter 2) and the sparse cliff (iter 5).
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (continuous, always active) ---
    k_prox = 7.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (primary approach driver)
    # ============================================================
    w_progress = 15.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (global baseline pull toward origin)
    # ============================================================
    w_dist = 0.3
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: proximity_reward (smooth, always-active approach incentive)
    # Replaces iter 2's settle_reward pure-proximity base.
    # 100% active, no threshold cliff: provides continuous gradient toward target.
    # ============================================================
    w_prox = 3.0
    proximity_reward = w_prox * curr_prox

    # ============================================================
    # Component D: stable_landing_bonus (proximity-gated stability extra)
    # Only meaningful near target; soft factors avoid hard-threshold cliff.
    # Multiplies proximity so the bonus scales with closeness.
    # ============================================================
    speed_ok = max(0.0, 1.0 - curr_speed / 0.5)
    angle_ok = max(0.0, 1.0 - abs(nangle) / 0.2)
    contact_factor = 0.5 * (nleft_contact + nright_contact)
    w_stable = 4.0
    stable_landing_bonus = w_stable * curr_prox * speed_ok * angle_ok * (1.0 + contact_factor)

    # ============================================================
    # Component E: velocity_damping (global mild speed penalty, prox-gated)
    # ============================================================
    w_vel = 0.3
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component F: orientation_penalty (proximity-gated)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component G: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + proximity_reward
        + stable_landing_bonus
        + velocity_damping
        + orientation_penalty
        + engine_penalty
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "proximity_reward": float(proximity_reward),
        "stable_landing_bonus": float(stable_landing_bonus),
        "velocity_damping": float(velocity_damping),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), components
```