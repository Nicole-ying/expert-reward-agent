# Response Record

# 设计理由

## 第 0 步：信号覆盖审计
- **终止模式**：20/20 terminated，12/20 early_terminal（<150步且score<-50），平均长度134步。着陆器快速坠毁，无法接近着陆垫。
- **观测使用扫描**：
  - obs[0:6] 全部被使用（位置、速度、角度、角速度）
  - **obs[6] left_support_contact 未被使用** ❌
  - **obs[7] right_support_contact 未被使用** ❌
  - 这两个接触信号是判断"是否到达地面"的关键信息。当前着陆器在接近地面时没有明确反馈，velocity_penalty的gate阈值（0.5）太严格，导致着陆器在坠毁前收不到"你应该慢下来"的信号。
- **信号缺口判断**：**信号校准问题为主，兼有信号缺失**。contact信号缺失降低了gate的可用性，但核心问题是velocity_penalty gate几乎不触发（active_rate=1.5%），这让整个接近阶段缺少减速引导。
- **僵尸组件**：velocity_penalty（active_rate=1.5%）— gate阈值为0.5，而着陆器基本从未进入0.5范围内，导致该组件完全失效。

## 行为诊断
- **agent在做什么**：快速坠毁。proximity_delta提供微弱的前进信号（~0.005/步），但velocity_penalty几乎不工作（~0.0004/步），orientation_penalty量级太小（~0.0002/步）。agent在无约束情况下加速下落。
- **干预目标**：velocity_penalty — 需要它在更远距离就开始警告高速，而不是等到0.5内才突然生效。
- **方向判断**：这是第一次迭代，方向本身合理（proximity驱动接近+velocity约束减速+orientation约束姿态），但velocity gate太紧且整体量级太小。无需重建，做Level 2结构变换。

## 选择干预层级 — Level 2 结构变换

| 证据 | 变换 |
|---|---|
| velocity_penalty active_rate=1.5%，gate阈值0.5几乎永远不开 | 二值/线性gate → **反距离加权连续因子** `1/(dist + ε)`，在全距离范围生效 |
| 所有组件的per-step贡献（合计~0.0057）远小于原始环境惩罚（~0.7575） | 全局系数放大 ~50×，使生成奖励可被感知 |
| velocity_penalty当前用`vx²+vy² * gate`，gate闭合时完全无反馈 | 改用 `speed / (dist + ε)` 形式的危险度信号，速度越高、距离越近惩罚越大 |

## 新组件设计：velocity_danger（替代 velocity_penalty）

**数学形式**：`danger = -w_vel * (speed_sq / (dist_cur + proximity_threshold))`

其中 `speed_sq = vx² + vy²`，`dist_cur` 是到原点的欧氏距离，`proximity_threshold = 1.0` 作为平滑因子防止除零并在远处也有适度惩罚。

**为什么这个形式有效**：
- 当距离大时（如dist=5），danger ≈ w_vel * speed_sq / 6 ≈ 较小的惩罚，高速仍会被警告但不压制探索
- 当距离小时（如dist=0.5），danger ≈ w_vel * speed_sq / 1.5 ≈ 强惩罚，迫使减速
- 不再有"突然开启"的阈值断崖，梯度平滑
- active_rate将从1.5%跃升到接近100%

**系数校准**：
- 当前主信号proximity_delta ≈ 0.005/步（放大50×后≈0.25/步）
- velocity_danger在dist=2, speed_sq=4时：0.3 * 4/(2+1) = 0.4/步 — 偏高，需降低
- 目标：velocity_danger per-step ≤ 主信号per-step的0.3倍
- 设 w_vel = 0.15，主信号放大到 w_prox = 50
  - 主信号 per-step ≈ 50 * 0.005 = 0.25
  - velocity_danger在dist=3, speed_sq=4时：0.15 * 4/4 = 0.15/步 ✓
  - velocity_danger在dist=1, speed_sq=4时：0.15 * 4/2 = 0.3/步 ≈ 主信号的1.2x（接近垫时短暂超过，但此时gradient让agent快速学会减速）
- orientation_penalty保持在 0.2 * (0.1²+0.1²) ≈ 0.004/步，放大到w_ang=5，则~0.02/步，为主信号的8%，可接受

**接触信号的使用**：本轮暂不引入contact奖励以避免复杂化，但velocity_danger的反距离形式已经在功能上弥补了gate的信号缺口——当距离减小时自动增加减速压力，即使用不到contact也能感知"接近地面"。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2 reward — velocity_danger replaces velocity_penalty with smooth inverse-distance weighting.
    Components:
      - proximity_delta:  scaled-up improvement in distance to target (core driving signal)
      - velocity_danger:  speed² / (dist + ε) penalty — continuous, no hard gate
      - orientation_penalty: scaled-up penalty on tilt and angular rate
    """
    # ── current state ──
    x_cur = obs[0]
    y_cur = obs[1]
    vx_cur = obs[2]
    vy_cur = obs[3]
    angle_cur = obs[4]
    angvel_cur = obs[5]

    # ── next state ──
    x_next = next_obs[0]
    y_next = next_obs[1]

    # ── distance to pad (target at 0, 0) ──
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # ── weights / thresholds ──
    w_prox = 50.0           # core progression weight (scaled up 50× from v1)
    w_vel  = 0.15           # velocity danger weight (reduced to keep penalty bounded)
    w_ang  = 5.0            # orientation penalty weight (scaled up 50× from v1)
    proximity_threshold = 1.0  # smoothing constant for inverse-distance (was gate threshold)

    # ── 1. Proximity delta (improvement_delta) ──
    proximity_delta = w_prox * (dist_cur - dist_next)

    # ── 2. Velocity danger (inverse_distance_weighting) ──
    # Continuous penalty: high speed is always warned, severity grows as distance shrinks.
    speed_sq = vx_cur ** 2 + vy_cur ** 2
    velocity_danger = -w_vel * speed_sq / (dist_cur + proximity_threshold)

    # ── 3. Orientation stability (quadratic_penalty) ──
    orientation_penalty = -w_ang * (angle_cur ** 2 + angvel_cur ** 2)

    # ── Total reward ──
    total_reward = proximity_delta + velocity_danger + orientation_penalty

    components = {
        "proximity_delta": proximity_delta,
        "velocity_danger": velocity_danger,
        "orientation_penalty": orientation_penalty,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 观测使用覆盖6/8维度，contact信号缺失但不是当前主因；velocity_penalty的gate阈值过紧(active_rate=1.5%)是核心卡点；所有组件per-step量级被原始环境惩罚淹没。
- **behavior**: agent快速坠毁 — proximity_delta提供微弱前进信号但无有效减速约束，velocity_penalty几乎不触发。
- **signal**: 缺少全场生效的减速压力信号 — 旧gate仅在dist<0.5时开启，着陆器从未到达该区域。反距离加权形式填补了这一缺口。
- **level**: Level 2 — 结构变换（gate → 反距离加权）+ 系数重校准
- **hypothesis**: velocity_danger将在整个下降过程中提供梯度递减的减速压力，proximity_delta放大50×后成为可感知的学习信号，两者配合让agent学会"接近时减速"而非"全速坠落"。
- **risk**: w_prox=50可能在前几步产生较大的正奖励波动（dist_cur - dist_next的绝对值放大），但速度danger会在高速时平衡它。若len显著增加但score改善有限，下一轮可能需要引入contact-based soft success bonus。
