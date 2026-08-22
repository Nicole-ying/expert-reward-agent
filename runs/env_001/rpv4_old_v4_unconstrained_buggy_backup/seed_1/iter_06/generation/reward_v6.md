现在我已掌握所有数据。让我进行综合分析。

---

## 诊断报告

### 1. Evidence

| 来源 | 关键数据 |
|---|---|
| **跨迭代演化** | v1(-2.98, len=813) → v2(67.23, len=561) → v3(128.44, len=574) → **v4(135.22, len=285)** → v5(-113.47, len=68) |
| **v4 组件平衡** | proximity_progress=44.36 (22.5%), soft_landing_bonus=161.37 (76.2%), attitude=-2.60 (1.2%) — 有倾斜但可工作 |
| **v5 组件平衡** | proximity_progress=55.88 (**94.7%**), soft_landing_bonus=1.02 (**1.7%**), attitude=-2.24 (3.7%) — 严重失衡 |
| **v5 终止与长度** | terminated=**20/20** (全部提前终止=坠毁), len=**68.4** (v4 的 1/4) |
| **v5 checkpoint score** | 全程在 [-120, -111] 之间，完全平坦无改善 — 训练陷入死局 |
| **v5 soft_landing_bonus checkpoint** | active_rate 全程 1.2%-1.8%，ep_sum 始终 ~1.0 — 几乎是个死组件 |
| **v4 checkpoint 不稳定性** | soft_landing_bonus cp100%=431.5 爆炸，但 cp90%=87.2 时 score=157.9 最佳 |
| **proximity_progress 历史** | v1:36.7 → v2:32.3 → v3:32.6 → v4:44.4 → **v5:55.9** — 持续膨胀 |

### 2. Behavior Diagnosis

**v5 的策略行为：高速俯冲坠毁。** 

证据链清晰：proximity_progress 占 94.7% 的量级份额，agent 唯一学到的行为是「尽快缩短与目标的欧氏距离」。接近目标的最快方式是什么？垂直俯冲。没有任何组件有效惩罚高速——soft_landing_bonus 已萎缩到 1.7% 且 active_rate 仅 1.5%（意味着几乎从未触发过有意义的着陆奖励），attitude_penalty 微弱。20/20 的 episode 在 68 步内提前终止（坠毁/飞出边界）。

**根因分析：v4→v5 的三项修改互相放大，导致崩溃：**
1. proximity_progress 40→50 (+25%)：更强的俯冲激励
2. soft_landing_bonus 系数 4.0→2.5 + 温度 5→3：着陆奖励被双重削弱，从 76.2% 暴跌到 1.7%
3. 没有速度惩罚来制衡：proximity_progress 天然奖励高速接近，缺乏反向约束

**关键洞察：proximity_progress 和 soft_landing_bonus 之间存在根本性张力。** proximity_progress 奖励任何距离缩短（包括高速俯冲），而 soft_landing_bonus 只在双腿接触 + 低速 + 水平姿态时生效。当 proximity_progress 过强时，agent 学到「高速俯冲→快速缩短距离→大量奖励」，完全跳过软着陆阶段。需要一个桥梁组件来约束接近阶段的速度。

### 3. 修改方案

**L2 修改**：添加 `velocity_penalty` 组件（新数学形态），同时回退 v5 中被证明有害的系数变更。

| 修改项 | v5 值 | 新值 | 理由 |
|---|---|---|---|
| proximity_progress 系数 | 50.0 | **40.0** | 回退到 v4 值，降低俯冲激励 |
| soft_landing_bonus 系数 | 2.5 | **4.0** | 回退到 v4 值，恢复着陆信号强度 |
| soft_landing_bonus 温度 | 3.0 | **5.0** | 回退到 v4 值 |
| velocity_penalty | 无 | **新增** | 填补「接近引导」和「软着陆」之间的空白 |

**velocity_penalty 的语义角色**：对瞬时速度施加二次型惩罚。低速时几乎为零（不影响着陆阶段），高速时显著增长（抑制俯冲）。这使得 agent 必须在「接近目标（proximity_progress）」和「控制速度（velocity_penalty）」之间找到最优平衡——即快速但不失控地下降，为软着陆创造条件。

二次型优于线性：着陆阶段速度接近零时惩罚可忽略，而俯冲时 v² 快速增长，形成自然的速度上限。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v6 — Adds velocity penalty to bridge approach and soft landing.

    Changes from v5:
    - proximity_progress: 50.0 → 40.0 (revert to v4). v5's boost caused
      the agent to dive at full speed, ignoring landing quality entirely.
    - soft_landing_bonus: coefficient 2.5→4.0, temperature 3.0→5.0 (revert to v4).
      v5's weakened landing bonus collapsed to 1.7% magnitude share.
    - NEW velocity_penalty: quadratic penalty on instantaneous speed.
      This is the key semantic addition — it discourages the high-speed
      diving strategy that proximity_progress naturally incentivizes.
      Quadratic form means near-zero penalty at landing speeds, significant
      penalty at crash-inducing speeds.
    - attitude_penalty unchanged.
    """

    # ---------- proximity_progress: improvement in Euclidean distance to target ----------
    old_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    new_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    proximity_progress = 40.0 * (old_dist - new_dist)

    # ---------- velocity_penalty: discourage high-speed diving ----------
    # Quadratic: gentle at low speeds (landing), significant at high speeds (crashing)
    vx = next_obs[2]
    vy = next_obs[3]
    velocity_penalty = -0.3 * (vx ** 2 + vy ** 2)

    # ---------- attitude_penalty: keep tilt and spin under control ----------
    angle = next_obs[4]
    angular_vel = next_obs[5]
    attitude_penalty = -0.5 * (angle ** 2) - 0.1 * (angular_vel ** 2)

    # ---------- soft_landing_bonus: continuous per-step reward for safe contact ----------
    both_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0

    v_mag = abs(vx) + abs(vy)
    velocity_quality = 1.0 / (1.0 + 5.0 * v_mag)       # → 1.0 at v=0, → 0.17 at v=1
    angle_quality    = 1.0 / (1.0 + 5.0 * abs(angle))  # → 1.0 at θ=0, → 0.17 at |θ|=1

    soft_landing_bonus = 4.0 * both_contact * velocity_quality * angle_quality

    # ---------- total reward ----------
    total_reward = proximity_progress + velocity_penalty + attitude_penalty + soft_landing_bonus

    components = {
        "proximity_progress": proximity_progress,
        "velocity_penalty": velocity_penalty,
        "attitude_penalty": attitude_penalty,
        "soft_landing_bonus": soft_landing_bonus
    }
    return float(total_reward), components
```