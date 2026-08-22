## 诊断报告

### Evidence

| 证据 | 数据 | 含义 |
|---|---|---|
| **历史最优 iter_4** | score=155.35, len=514.8, terminated=19/20 | V4 连续接触奖励是唯一成功引导 agent 到达目标的版本 |
| **iter_4→iter_5 退化** | score 155→82, landing_contact 35→479(91.4%) | iter_5 的 sustain+transition(sustain=0.3, transition=4.0) 创造了巨大 exploit 漏洞 |
| **iter_5→iter_6 崩溃** | score 82→-116.65, landing_contact active 58.4%→1.2% | V6 势能差（potential-based）杀死了接触信号，agent 回到早期崩溃模式 |
| **iter_6 checkpoint 全平坦** | score cp10%–cp100% 全部 ≈ -118~-116 | RL 完全没有学习——所有组件平坦，无梯度信号 |
| **iter_6 landing_contact 几乎死亡** | active_rate=1.2%, cp10%–cp100% 全部 ≈ 4.3~5.0 | 99% 的步骤收不到任何正向奖励，agent 在纯惩罚下崩溃 |
| **iter_6 settling_stability 主导** | ep_sum=-14.66, magnitude_share=51.2% | 没有正向信号平衡时，速度惩罚吞噬一切 |
| **iter_6 terminated=20/20, len=68.35** | 与 iter_1/iter_2 相同模式 | agent 回到最早期的"自由落体→碰撞→终止"模式 |
| **iter_4 的 checkpoint 趋势** | cp80%=221.3 → cp100%=130.2 | 说明 late-training 存在接触奖励累积问题，但整体有效 |
| **iter_4 landing_contact** | ep_sum=35.2, active=17.5%, magnitude_share=41% | 适中——足够引导学习，未达到 iter_3(647)/iter_5(479) 的 exploit 级别 |

### behavior_diagnosis

当前策略（iter_6 V6）已经完全退化到早期失败模式：agent 不点火、不控制姿态、高速坠落到视口外或碰撞终止（len=68.35, 20/20 提前终止）。根本原因是 V6 的势能差公式 `max(0, Δpotential)` 要求 agent 必须先进入 near_pad+slow 区域才能获得任何 landing_contact 奖励，但 agent 从未学会接近目标垫板——因为唯一能引导它接近的信号（连续接触奖励）被移除了。形成了鸡生蛋蛋生鸡的死锁。

**核心教训**：
1. iter_4 的连续接触奖励（`2.0 * prox * speed * contact_frac`）是唯一成功引导 agent 到达目标垫板的信号结构
2. 后续所有试图"修复 exploit"的 L2 改动（iter_5 的 transition reward、iter_6 的 potential-based）都杀死了学习信号
3. 需要回到 iter_4 的成熟结构，仅做 L1 微调

### 干预方案

- **层级**：L1 — 仅调整系数，不改数学表达式
- **目标组件**：`landing_contact`
- **具体修改**：恢复 iter_4 的连续接触奖励结构，将系数从 2.0 降至 1.2
- **理由**：iter_4 在系数 2.0 时 cp80% 出现接触奖励尖峰（151.7），暗示存在轻度过激励。1.2 保留足够学习信号的同时降低徘徊激励。其他四个组件（goal_proximity, settling_stability, attitude_constraint, fuel_efficiency）保持 iter_4 原样——它们在 iter_4 中已被验证有效且平衡。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v7。

    核心策略：回到 iter_4（score=155.35，历史最优）的成熟结构，
    仅将 landing_contact 系数从 2.0 降至 1.2（L1 微调），
    以缓解 late-training 的接触奖励过度累积，同时保留足够的学习引导信号。

    设计原则（继承 v4）：
    - goal_proximity: 二次距离代价 → 引导接近（远处强、近处弱）
    - settling_stability: 二次速度代价 → 引导减速（高速强、低速弱）
    - landing_contact: 连续接触奖励 → 奖励"接近+慢速+接触"的着陆过程
    - attitude_constraint: 姿态代价 → 保持平稳
    - fuel_efficiency: 燃料代价 → 节省推力

    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """

    # ========== 信号提取（使用 next_obs 评估转移后状态） ==========
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ========== 辅助量 ==========
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5

    # proximity_gate: 1.0 在目标处，~0.1 在 distance≈3
    proximity_gate = 1.0 / (1.0 + 3.0 * distance)

    # ========== Component A: goal_proximity — 二次距离代价 ==========
    # 梯度 = -0.2 * distance，远处梯度强，近处梯度弱（天然 curriculum）
    goal_proximity = -0.1 * distance ** 2

    # ========== Component B: settling_stability — 二次速度代价 ==========
    # 梯度随 speed 和 proximity 增强，接近目标时减速约束更强
    settling_stability = -0.3 * speed ** 2 * (0.2 + 0.8 * proximity_gate)
    settling_stability += -0.08 * ang_vel ** 2 * (0.2 + 0.8 * proximity_gate)

    # ========== Component C: attitude_constraint — 姿态代价 ==========
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    # gate 范围：远距离 0.3，近距离 1.0
    attitude_gate = 0.3 + 0.7 * proximity_gate
    attitude_constraint = -0.4 * angle_error * attitude_gate
    attitude_constraint += -0.15 * ang_vel_mag * attitude_gate

    # ========== Component D: landing_contact — 连续接触奖励 ==========
    # v7：恢复 v4 的连续接触奖励结构，系数从 2.0 → 1.2（L1 微调）。
    # 仅在实际有支撑腿接触、靠近垫板、且速度足够低时发放奖励。
    # 三个条件共同防止 agent 在无关阶段获得误奖励：
    #   - near_pad: 距离 < 0.5，确保只在目标附近激活
    #   - slow_enough: 速度 < 1.0，防止高速撞击时领奖
    #   - contact_frac > 0: 至少一条腿接触，推动完成最后一步
    near_pad = distance < 0.5
    slow_enough = speed < 1.0
    contact_frac = (left_contact + right_contact) / 2.0  # 0.0 ~ 1.0

    if near_pad and slow_enough and contact_frac > 0.0:
        # 连续缩放因子：越近越好、越慢越好
        prox_factor = 1.0 - distance / 0.5      # 1.0@d=0, 0.0@d=0.5
        speed_factor = 1.0 - speed / 1.0        # 1.0@s=0, 0.0@s=1.0
        # v7: 系数 1.2（v4 为 2.0），降低徘徊激励
        landing_contact = 1.2 * prox_factor * speed_factor * contact_frac
    else:
        landing_contact = 0.0

    # ========== Component E: fuel_efficiency — 燃料代价 ==========
    fuel_efficiency = -0.03 * float(action != 0)

    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        settling_stability +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )

    components = {
        'goal_proximity': goal_proximity,
        'settling_stability': settling_stability,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), components
```