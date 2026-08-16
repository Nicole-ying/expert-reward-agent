# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Potential‑based shaping reward for the 2D lander task.
    The potential penalises distance, speed, tilt and angular velocity,
    and rewards leg‑contact, all in a single continuous function.
    The reward is the improvement in potential from obs to next_obs.
    """
    # Weights – can be tuned later
    w_dist  = 5.0
    w_vel   = 1.0
    w_angle = 2.0
    w_angvel= 1.0
    w_contact = 2.0

    def potential(o):
        x = o[0]
        y = o[1]
        vx = o[2]
        vy = o[3]
        angle = o[4]
        ang_vel = o[5]
        left_c  = o[6]
        right_c = o[7]

        distance = (x**2 + y**2)**0.5
        speed    = (vx**2 + vy**2)**0.5
        contact  = left_c + right_c   # 0, 1 or 2

        # Combine all objectives into a single scalar (higher is better)
        return -(w_dist * distance
                 + w_vel * speed
                 + w_angle * abs(angle)
                 + w_angvel * abs(ang_vel)
                 - w_contact * contact)

    pot_current = potential(obs)
    pot_next    = potential(next_obs)

    reward_delta = pot_next - pot_current

    # For numerical stability we can clamp the delta, but it’s usually small
    reward = reward_delta

    components = {
        "progress_delta": reward
    }

    return float(reward), components
```

# reward_v1 设计说明

## 1. 选择的任务族与动力学子类型
- **task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- 核心目标：将 2D 着陆器平稳降落在目标垫上，位置、速度、姿态和双腿接触都要满足终止条件。

## 2. 选择的奖励职责 (reward roles)
根据 `reward_role_decomposition`，v1 同时包含了全部 **mandatory** 角色，但通过一个统一的 potential‑based shaping 实现，避免了多个独立组件间的尺度冲突与信号对抗：
- `goal_proximity` – 以距离项驱近目标点。
- `velocity_damping` – 惩罚线速度幅值，迫使减速。
- `orientation_stabilization` – 惩罚机体倾角与角速度，保持竖直。
- `soft_contact_encouragement` – 奖励双腿触地（增加 contact 总和，间接鼓励同时触地）。

**条件职责**（`fuel_efficiency_penalty`、`final_settlement_bonus`）留待后续迭代加入，v1 阶段让 agent 先学会安全着陆。

## 3. 职责‑信号映射
| role | 使用的信号 | 备注 |
|---|---|---|
| goal_proximity | `obs[0], obs[1]` (x, y) | 欧氏距离 |
| velocity_damping | `obs[2], obs[3]` (vx, vy) | 合速度 |
| orientation_stabilization | `obs[4], obs[5]` (angle, angvel) | 倾角与角速度 |
| soft_contact_encouragement | `obs[6], obs[7]` (left/right contact) | 离散接触，相加成 0‑2 |

全部信号均在观测空间内声明，无缺失。

## 4. 公式算子选择
- **主框架**: **potential_based_shaping** (`expert_reward_context.md` 3.3)  
  `reward = potential(next_obs) - potential(obs)`  
  因为任务有一个明确的单调进展方向（距离→0、速度→0、姿态稳定、接触增加），且用单一 potential 将多目标融合为一个连续、每步都有梯度的信号，避免了多组件间的权重博弈。
- **potential 内部**：
  - 距离、速度、角度、角速度：**dense_state_signal (penalty‑quadratic / absolute)**  
    直接使用原始量级的线性/绝对值惩罚，不平方，避免梯度爆炸。
  - 接触：**dense_state_signal (positive linear)**  
    奖励接触腿数，作为 potential 的加分项，而非单独事件。
- **未使用**：
  - `improvement_delta` – 这里直接使用全状态 potential 的差分，效果等同。
  - `terminal_event` – 缺少显式成功/失败标志，且 potential 的连续反馈已经隐式惩罚了 crash 前兆。
  - `action_efficiency` – 离散动作，v1 不加入燃料惩罚，防止早期探索抑制。

## 5. 排除的角色与原因
- `terminal_success_reward`：环境没有 explicit_success_flag (`info` 为空)，无法可靠构造。
- `terminal_failure_penalty`：无 failure flag，且 crash 等失败会在终止前通过速度/角度惩罚带来负 delta，已有信号。
- `fuel_efficiency_penalty`：属于 conditional role，v1 暂不加入，避免阻碍初期引擎使用。
- `time_efficiency_reward`：缺少步数或剩余时间信号，不可用。

## 6. 设计时考虑的风险与缓解
- **高速撞击**：速度项在接近地面时（距离小）对 potential 影响变大，agent 会在下降末期主动减速。
- **悬停不降落**：距离项持续产生负 potential，若 agent 保持高度则不改善 potential，也无法获得正 delta，探索中会逐渐尝试下降。
- **单腿着陆**：contact 项鼓励双腿同时接触；若只有一条腿触地，potential 增量只有一半，仍有足够动力去调整触地。
- **姿态异常**：身体倾角和角速度的惩罚会阻止翻滚和过度摇摆。

## 7. 留待后续迭代的职责
- **fuel_efficiency_penalty**：待 agent 稳定着陆后，可在接近目标垫时引入轻微动作惩罚，优化燃料经济性。
- **final_settlement_bonus**：当前用连续 potential 已能提供稠密梯度；若后期需要更精确的终止激励，可加入 joint‑condition proxy 的 soft completion 信号。
- **dynamic curriculum**：不需要，v1 的 potential 已自然缩放。

## 8. 训练后应观察的 failure modes
- **飘出视口**：若总分始终很低且距离项持续负值，可能需增大 `w_dist` 或减小 `w_vel`，确保早期推动力足够。
- **冲过去又反弹**：若速度在接触前瞬间仍高，可考虑在距离较小时提高 `w_vel` 或加入 hinge。
- **悬停犹豫**：若 agent 长时间保持在高空，可后续增加高空惩罚或减小速度阻尼权重。
- **单腿摇摆**：若总是只有一条腿接触，可增大 `w_contact` 或加入更明确的乘积条件（如双腿接触且低速时才给激励）。

该设计遵循“环境事实→任务画像→奖励职责→职责‑信号映射→公式算子”的顺序，使用声明的观测信号，不依赖任何 `info` 或 `original_reward`，符合全部硬约束。
