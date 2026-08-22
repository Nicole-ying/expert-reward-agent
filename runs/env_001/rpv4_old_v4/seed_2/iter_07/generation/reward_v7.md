Now I have the full evolutionary picture. Let me compile the diagnosis.

---

## 诊断报告

### 1. Evidence（关键证据）

**六轮迭代演化轨迹（contact_guidance 为核心线索）：**

| iter | contact 公式 | contact ep_sum | contact active% | contact mag% | progress wt | score |
|:---:|---|---:|---:|---:|---:|---:|
| 1 | state-value (0.3×) | 252.9 | 68.0% | 99.2% | 0.5 | **171.4** |
| 2 | potential-diff (100×) | 155.5 | 81.0% | 99.9% | 2.0 | 157.1 |
| 3 | potential-diff (3×) | 3.8 | 0.9% | 32.5% | 5.0 | **-73.6** |
| 4 | state-value (0.3×) | 0.25 | 0.8% | 3.7% | 5.0 | **-52.5** |
| 5 | state-value (0.3×) | 328.9 | 66.4% | 99.2% | 1.0 | 162.3 |
| 6 | transition (100×) | **9810** | 12.1% | 100% | 1.0 | 101.8 |

**Iter 6 的致命信号：**
- `contact_guidance` ep_sum 从 iter_5 的 329 **暴增至 9810**（29×），但 score 从 162 跌至 102
- active_rate 仅 12.1%，但 magnitude_share = 100%：每次触发都产生巨大的 100 分奖励
- 这是典型的 **bouncing exploitation**：agent 在平台上反复弹跳，每次 gain→loss→gain 一条腿就赚 100 分
- 证据：ep_len=786，contact 触发 ~98 次（9810/100），即每 ~8 步弹跳一次

**历史教训（不可再犯）：**
- progress weight ≥ 5.0 → agent 全速冲向目标并坠毁（iter_3 全部 20 集 terminate，len=96；iter_4 全部 20 集 terminate，len=72）
- potential-diff / transition-based contact → 要么 magnitude 噪声支配（iter_2），要么信号消失（iter_3），要么弹跳 exploit（iter_6）
- **唯一可行的 contact 范式是 state-value**（iter_1/iter_5），但 iter_5 的 cp 趋势显示 lingering exploitation：cp70% score=235 → cp100% score=157，同期 contact 从 28 升至 332

### 2. Behavior Diagnosis

当前 iter_6 的策略学会了**弹跳欺骗**：在目标平台附近反复获得/失去腿部接触，每 ~8 步赚取 100 分 transition bonus。这导致训练奖励（9810）与外部评分（102）严重脱钩。agent 没有学会稳定着陆，而是学会了 bouncing。

### 3. Signal Completeness

| 职责 | 状态 | 问题 |
|---|---|---|
| goal_proximity (progress) | 存在但被淹没 | 0.0% magnitude_share，被 contact 完全压制 |
| velocity_management | 存在但无力 | 仅惩罚 vy²，未阻止水平滑行 |
| orientation_stabilization | 存在但无力 | 独立惩罚太弱（-0.05/步 vs contact 100/次） |
| safe_contact | **方向错误** | transition-based 鼓励弹跳而非稳定 |

**缺失的约束：** contact 奖励缺少对水平速度（vx）和姿态角度（angle）的门控。iter_1/5 的 `velocity_ok` 只检查了 `|vy|`，agent 可以通过水平滑行或倾斜姿态来收集 contact 奖励而不触发 termination。

### 4. Selected Level: **L2**

**理由：** 
- 需要改 contact 的数学形态（从 transition-based 回到 state-value，并扩充 velocity gate 和添加 angle gate）
- L1 调系数无法修复 transition-based 的结构性问题（弹跳 exploit）
- L3 不需要——state-value 范式（iter_1）已验证可行，只需修补其 lingering exploit 漏洞

### 5. Selected Intervention

**回到 iter_1 的 state-value contact，但修补两个漏洞：**

1. **velocity gate 从 `|vy|` 扩展为 `speed = (vx²+vy²)^0.5`**：阻止水平滑行时收集 contact 奖励
2. **新增 angle gate `angle_ok`**：阻止倾斜姿态时收集 contact 奖励

同时保持 iter_1 验证过的参数：progress=0.5，contact=0.3。

### 6. Falsifiable Hypothesis

当前 exploit 的机制：agent 保持 vy≈0 但 vx≠0（水平滑行），或保持倾斜，同时反复弹跳赚取 transition bonus。

新 gate 的预期效果：要获得 contact 奖励，agent 必须同时满足 (a) 靠近目标 (b) 低速——水平和垂直 (c) 姿态接近水平。这三个条件同时满足时，`body_not_awake_or_settled` 应快速触发 termination，**agent 只能领取寥寥数步的 contact 奖励即终止**，消除 lingering 动机。

**可证伪预测：** 下一轮 contact_guidance active_rate 应从 12% 回升到 40-70%（state-value 持续激活），但 ep_sum 应从 9810 回落到 100-400 范围；score 应回升到 140-180。

### 7. Expected Next Round

策略应恢复为**飞向目标 → 减速 → 稳定着陆 → 快速终止**的行为模式。terminated 比例应上升（当前 8/20），ep_len 应缩短。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to target (target at origin (0,0)) ---
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # --- velocity components ---
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # --- orientation penalty: body angle and angular velocity ---
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # --- contact guidance: state-value with comprehensive settling gates ---
    # Reverting from iter_6's transition-based approach (which caused bouncing
    # exploitation: ep_sum=9810, active=12%) back to state-value (iter_1/5).
    #
    # Key L2 change from iter_1/5: the settling gate now checks total speed
    # (vx AND vy, not just |vy|) plus body angle. This closes two exploitation
    # loopholes:
    #   1. Horizontal sliding: iter_1/5 only gated |vy|, so agent could slide
    #      sideways (vx≠0, vy≈0) and collect full contact reward.
    #   2. Tilted stance: iter_1/5 had no angle gate on contact, so agent could
    #      stay tilted on the platform collecting reward without settling.
    #
    # With all three gates (proximity + speed + angle) simultaneously tight,
    # the agent only earns contact reward when it is truly settled — which
    # should trigger body_not_awake_or_settled termination within a few steps.
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0, 1.0, or 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    speed = (vx**2 + vy**2) ** 0.5
    velocity_ok = max(0.0, 1.0 - speed / 0.5)       # speed < 0.5 for any reward
    angle_ok = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)  # |angle| < 0.3 for any reward
    contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok * angle_ok

    # --- combine all components ---
    # Progress weight=0.5 matches iter_1 (score=171, best ever).
    # Weights >0.5 (iter_4/5) caused rushing-and-crashing or lingering.
    total_reward = (0.5 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 0.5 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```