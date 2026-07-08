## 诊断

### 0. 环境理解
2D 飞行器着陆任务，目标：到达原点 (0,0)、减速、保持姿态、用双腿平稳着陆。观测 8 维，动作 4 离散（无引擎/左姿态/主引擎/右姿态）。

### 1. Agent 发生了什么？
- **Score = -4345.46**，episode_length = 383，early_terminal = 30%
- `original_env_reward` mean = -5.79/step（大量负分来自引擎燃料消耗），ratio_to_progress = -40
- `landing_reward` nonzero_rate = 0.0012%，几乎从未触发
- **结论：agent 在 crash**。v8 的 potential-based shaping + 稀疏 landing bonus 完全失效。Progress reward 虽然为正（0.143），但 agent 学到的策略产生了大量负环境分（烧燃料 + 碰撞），说明奖励函数 misaligned。

### 2. 哪个组件是主要原因？
- `landing_reward`：nonzero_rate 0.0012%，等于不存在，对学习无帮助
- `progress_reward`（potential-based shaping）：gamma=0.99 的势能塑形公式本身没问题，但单靠它没有足够引导完成着陆——没有接近目标时的精细化激励，也没有接触激励
- **根因：v8 删掉了历史最佳代码中行之有效的 approach_quality 和 attitude_penalty，代之以不触发的 landing_reward**

### 3. 我之前改了什么？
- Iter 7 从 `approach_quality + attitude_penalty + progress` 换成了 `landing_reward + progress`（potential-based），得分从 ~140 暴跌到 -4345
- **Revert 规则触发**：best_reward (146.36) 远高于 current (-4345)，必须回退到最佳骨架

---

### 修改方案
**回退到历史最佳骨架**（`progress + approach_quality + attitude_penalty`），在此基础之上做**一个新修改**：添加连续接触激励 `contact_reward`。

**理由**：历史最佳骨架让 agent 学到了靠近目标、减速、保持姿态（episode_length=1000 满时长，不 crash），但缺少"真正着陆"的激励——approach_quality 奖励"接近+低速+正姿"导致 agent 在目标上方悬停而不触地。添加小量连续接触奖励直接填补这个梯度缺口，引导 agent 完成最后一步：双腿着地。

系数 0.02 使得双腿着地时 contact_reward ≈ 0.04，远小于 approach_quality（~0.094），不足以破坏已有行为，但能提供朝向接触的梯度。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lander trajectory optimization.

    Reverts to the best-performing skeleton (progress + approach_quality +
    attitude_penalty) and adds a small continuous contact incentive to
    encourage actual touchdown instead of hovering above the target.
    """

    # ----- Unpack observations -------------------------------------------------
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # ----- Component 1: Progress towards target (0, 0) -------------------------
    dist_current = (x ** 2.0 + y ** 2.0) ** 0.5
    dist_next = (next_x ** 2.0 + next_y ** 2.0) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # ----- Component 2: Approach quality (continuous shaping) -------------------
    # Three bounded factors [0,1] multiplied: proximity, low speed, upright.
    # Product form ensures all conditions matter — but also means hovering
    # near the target with low speed and good attitude yields reward without
    # landing.  The new contact_reward (component 4) fills that gap.
    prox_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed = (next_x_vel ** 2.0 + next_y_vel ** 2.0) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    approach_quality = prox_factor * speed_factor * angle_factor

    w_approach = 0.15
    approach_quality_reward = w_approach * approach_quality

    # ----- Component 3: Attitude penalty (gentle uprighting force) --------------
    attitude_penalty = -0.01 * abs(next_angle)

    # ----- Component 4: Contact incentive (NEW — the single modification) --------
    # Small continuous reward per leg in contact.  Provides gradient toward
    # touchdown that the approach_quality product alone lacks.  Max ~0.04 when
    # both legs touch, modest vs approach_quality (~0.094 active mean).
    contact_reward = 0.02 * (next_left + next_right)

    # ----- Total ----------------------------------------------------------------
    total_reward = progress + approach_quality_reward + attitude_penalty + contact_reward

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality_reward,
        "attitude_penalty": attitude_penalty,
        "contact_reward": contact_reward,
    }

    return float(total_reward), components
```