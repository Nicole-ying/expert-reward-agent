---

## 1. 诊断报告

### evidence

| 数据来源 | 关键发现 |
|---|---|
| `read_training_feedback(2)` | score=235.03 > 目标200，terminated=19/20，len=425.2。`landing_quality` magnitude_share=87.8%，主导但不失控 |
| `read_checkpoint_trend(2)` | score 从 cp10%=-115 → cp100%=245.7，单调改善。`landing_quality` ep_sum 从cp70%的735峰值下降至cp100%的173——随策略改善（更快着陆），该组件收益自然下降，说明未被 exploit，而是随任务完成效率提升而"自我限额" |
| `read_memory()` | v1→v2：contact-gated sparse bonus → height-gated dense quality reward，得分从 -3 跳到 235。高度门控是本次突破的核心机制 |
| `read_reward_code(1)` vs `read_reward_code(2)` | v1 使用 `contact_score * velocity_factor * angle_factor`（三重乘积，active_rate=0.3%），v2 删除 contact 条件，改为 height_gate 线性 ramp。**但 v2 完全没有使用 obs[6]/obs[7]（接触信号）** |
| `get_component_history` | `attitude_penalty` 全程100%激活但 magnitude_share 仅 0.8%，`landing_speed` 68.9% 激活但 share 仅 0.4%。这两个约束组件几乎无影响力 |

### behavior_diagnosis

策略已学会在近地区域保持低速 + 水平姿态以获得 `landing_quality` 奖励，并成功触发 termination（19/20）。但奖励函数完全依赖高度门控的"近地品质"代理信号来判断着陆质量——**从未直接验证两腿是否真正接触垫面**。当前策略可能通过"低空悬停后自由落体"或单腿触地触发 settled 终止，而非规范的双腿软着陆。

### signal_completeness

对照环境卡片的 `reward_role_decomposition`：

| 职责 | 当前覆盖 | 缺失 |
|---|---|---|
| `goal_proximity` | ✅ `proximity_progress` | — |
| `attitude_stabilization` | ✅ `attitude_penalty` | — |
| `soft_landing_contact` | ⚠️ `landing_quality`（仅用高度+速度+角度代理，无接触验证） | **obs[6]/obs[7] 接触信号完全未使用** |
| `fuel_efficiency` | ❌ 无 | 可后续添加 |
| `time_efficiency` | ❌ 无 | 可后续添加 |

**核心缺口：接触信号。** 环境提供 `left_support_contact` 和 `right_support_contact`，是判断"安全双腿着陆"的最直接证据，但奖励函数完全忽略。

### selected_level

**L2** — 新增一个组件（`contact_bonus`），引入接触反馈这一全新语义方向。

选择理由：这不是系数调整能解决的问题——当前四个组件中没有任何一个涉及接触信号。必须新增组件来填补 `soft_landing_contact` 职责中的"接触验证"缺口。这仍然是一次单一语义修改。

### selected_intervention

**目标组件**：新增 `contact_bonus`

**具体修改**：
```python
both_legs_contact = min(next_obs[6], next_obs[7])
contact_bonus = 2.0 * both_legs_contact
```

- `min(obs[6], obs[7])` 确保仅当**双腿同时接触**时奖励激活（偏置单腿蹭地的行为得不到奖励）
- 不做高度门控叠加——接触本身已隐含近地条件（物理约束），叠加门控会进一步压缩本已稀少的激活窗口
- 系数 2.0：若接触持续 ~10 步，ep_sum ≈ 20，与 `proximity_progress`（~27）可比，不主导也不淹没

### falsifiable_hypothesis

**如果**添加双腿接触奖励，**那么**策略将获得直接信号来区分"近地悬停"与"真正触地"，**因此**：（1）着陆行为应更果断地完成触地而非在近地徘徊；（2）`contact_bonus` 的 active_rate 应在 2%~8% 之间（仅在着陆最后阶段触发）；（3）得分应维持或略高于当前 235，因为奖励信号更精确地指向任务目标。

### expected_next_round

| 指标 | 预期变化 | 理由 |
|---|---|---|
| `contact_bonus` active_rate | 2%~8% | 仅在触地最后阶段激活，约 10-30 步/ episode |
| `contact_bonus` ep_sum | 10~40 | 2.0 × 接触步数 |
| `landing_quality` magnitude_share | 从 87.8% 降至 75-85% | 新增组件分摊总 reward 量级 |
| score | 维持或略升（200-260） | 信号更精确，不应变差 |
| terminated | 维持 18-20/20 | 不应恶化 |

### main_risk

**最可能引入的新问题**：`contact_bonus` 的 active_rate 可能低于 2%（若接触仅持续 <5 步即被 settled 终止），导致变为"半死组件"。缓解：系数 2.0 已为此做了补偿，即使只激活 5 步也能贡献 10 ep_sum。若下轮 active_rate < 2%，应改用"接触瞬间一次性 bonus"（通过 `obs→next_obs` 的 contact 上升沿检测）。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v3 — 语义修改：新增双腿接触奖励，填补着陆反馈中缺失的接触验证信号。

    变更（相对 v2）：
      - 新增 contact_bonus：当双腿同时接触垫面时给予正向奖励
      - 其余组件保持不变
    """

    # ── 组件 A：接近进度（改善量信号） ──
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 20.0 * (old_dist - new_dist)  # 靠近>0, 远离<0

    # ── 组件 B：姿态惩罚（全程） ──
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ── 高度门控因子 ──
    # y_pos 从 0.5→0 时 gate 从 0→1 线性 ramp
    y_pos = next_obs[1]
    height_gate = max(0.0, min(1.0, (0.5 - y_pos) / 0.5))

    # ── 组件 C：近地速度惩罚 ──
    speed_penalty = -0.5 * (next_obs[2] ** 2 + next_obs[3] ** 2)
    landing_speed = height_gate * speed_penalty

    # ── 组件 D：近地着陆品质奖励 ──
    speed_mag = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_quality = 2.718281828 ** (-4.0 * speed_mag)      # 速度越低越接近 1
    angle_quality = 2.718281828 ** (-4.0 * abs(angle))     # 角度越小越接近 1
    landing_quality = 1.5 * height_gate * speed_quality * angle_quality

    # ── 组件 E：双腿接触奖励（新增） ──
    # 仅当左右支撑腿同时接触垫面时激活，提供接触验证的直接信号
    both_legs_contact = min(next_obs[6], next_obs[7])
    contact_bonus = 2.0 * both_legs_contact

    # ── 总奖励 ──
    total_reward = (proximity_progress + attitude_penalty + landing_speed
                    + landing_quality + contact_bonus)

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "landing_speed": landing_speed,
        "landing_quality": landing_quality,
        "contact_bonus": contact_bonus
    }

    return float(total_reward), components
```