第 2 轮迭代训练已经结束，得分 235.033692（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(2)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

3. **查全局历史避免打转**——用 `read_memory()` 看所有迭代。**关键判断：你是不是在同一个语义功能上反复修改但无突破？**（注意：组件名字可能变了，但语义功能相同——比如 success_landing_bonus 和 success_bonus 都是"着陆成功信号"。）如果某个语义已经改了 2-3 轮都不好，不要再改它了——换一个语义方向。

4. **参考 best code 但要验证**——用 `read_reward_code(best_iter)` 看历史最佳奖励函数的代码。它的哪些组件可能促成了当时的最高分？但不要盲目照搬——结合 best 那轮的 feedback（`read_training_feedback(best_iter)`）看：它的得分分布、终止率、组件 activation 是否健康。best 只是记录最高分，不一定代表正确的引导方式。

5. **Checkpoint 趋势做交叉验证**——用 `read_checkpoint_trend(N)` 查看训练过程中各组件的演化轨迹。如果怀疑某阈值太宽导致 exploit，交叉对比约束组件（如 speed→velocity_damping）的 share 是否同步下降。多维度证据指向同一结论，干预才能精准。

### 修改原则

**奖励函数迭代本质是黑盒实验中的渐进微调，不是代码重构。**
每轮 RL 训练是昂贵且不可跳过的——你无法预知一个新奖励函数会引导出什么行为，只能训练完后看结果。这不是确定性系统，是黑盒优化。

**核心方法论：看到好趋势 → 先理解为什么好，再精准微调。**
对比高分和低分的奖励函数：它们的代码差异是什么？差异如何解释了行为变化？提炼出有效模式后，用最小的改动（L1）去强化好趋势、削弱坏趋势。大规模重写（L2/L3）等于丢弃已有实验证据，重新掷骰子——只在明确诊断出数学缺陷时才用。

具体层级：

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态。以下是常见证据模式与对应变换方向，供参考（不是规则，根据实际情况判断）：

| 证据模式 | 参考变换 | 验证标准 |
|---|---|---|
| 任务事件几乎不触发，缺少局部反馈 | 稀疏→连续过程证据 | active_rate 改善，不产生 proxy 徘徊 |
| 极端值支配奖励 | 无界→归一化有界 | 极端轨迹支配下降，方差下降 |
| 占据好状态即可持续获奖 | 状态值→改善量/势能差 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | 全局→阶段门控 | 早期探索与局部约束同时改善 |
| 多个小因子相乘导致塌缩 | 乘积→加性/几何平均 | 非零反馈增多，联合约束保留 |
| 持续事件被重复领取 | 持续→状态转移事件 | 重复积累下降，外部完成保持 |
| proxy 提高但外部任务不升 | 代理→任务完成对齐 | proxy 与外部分数重新同向 |
| 稠密 proxy 形成中分平台 | 全程 proxy→局部/转移任务信号 | 刷新 best，完成相关行为增加 |

常用数学性质参考：二值稀疏条件信用分配困难；连续乘积表达联合满足但可能塌缩；加权和反馈稠密但允许目标补偿；bounded 函数限制极端值但输入仍需按环境尺度归一化；门控只应在证据表明作用阶段错误时使用。

- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。
- **⚠️ 组件被 exploit 时必须干预**：如果一个组件 magnitude_share > 90%，即使它刚被激活，也必须修改它——加触发条件限制、加每 episode 次数上限、或收紧阈值。"刚激活所以只做 L1"不适用于 exploit 场景。
- **避免打转**：如果 `read_past_reflection` 显示你之前已经改过同一语义且结果不好，换方向。
- **接近目标时优先微调**：如果当前得分或 best 得分已经接近目标（差距 < 10%），**强烈推荐 L1 微调**（只调系数）。历史尝试中反复出现同一模式：分数已接近目标，却选择 L2 改组件形态，结果越改越差，连续多轮倒退。架构已有基本有效性时，大规模改动风险远大于收益。

### 你需要的信息都在这里

实验记录位于 `rpv4_old_v4/seed_1/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件
- `read_checkpoint_trend(N)` — 查看某轮训练中各组件随训练的演化轨迹（ep_sum + active_rate）

现在开始调查。

# 上一轮训练反馈 (iter_2, score=235.033692)
# Training Feedback

## Final-policy outcome
score=235.033692, len=425.200000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[82.814065, 278.428241]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 219.677227 | 87.8% | 87.8% | 71.7% |
| proximity_progress | 26.805481 | 10.7% | 11.0% | 96.7% |
| attitude_penalty | -2.012444 | -0.8% | 0.8% | 100.0% |
| landing_speed | -1.047543 | -0.4% | 0.4% | 68.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward v2 — 语义修改：着陆反馈从接触门控改为高度门控。

    变更：
      - 删除 soft_landing_bonus（contact-gated 三重乘积，active_rate=0.3%）
      - 新增 landing_speed：近地速度惩罚，高度门控
      - 新增 landing_quality：近地品质奖励，鼓励低速+水平姿态
      - proximity_progress 系数 30→20，为着陆信号留权重空间
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
    # 高度越低，速度惩罚越强；高空不约束速度
    speed_penalty = -0.5 * (next_obs[2] ** 2 + next_obs[3] ** 2)
    landing_speed = height_gate * speed_penalty

    # ── 组件 D：近地着陆品质奖励 ──
    # 鼓励低速 + 水平姿态，仅在近地区域生效
    speed_mag = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    speed_quality = 2.718281828 ** (-4.0 * speed_mag)      # 速度越低越接近 1
    angle_quality = 2.718281828 ** (-4.0 * abs(angle))     # 角度越小越接近 1
    landing_quality = 1.5 * height_gate * speed_quality * angle_quality

    # ── 总奖励 ──
    total_reward = proximity_progress + attitude_penalty + landing_speed + landing_quality

    components = {
        "proximity_progress": proximity_progress,
        "attitude_penalty": attitude_penalty,
        "landing_speed": landing_speed,
        "landing_quality": landing_quality
    }

    return float(total_reward), components
```

# Subagent 数据报告
# 训练数据报告 — iter_02

---

## 1. 行为事实

- **得分**：score=235.03，range=[82.81, 278.43]
- **终止方式**：terminated=19/20，truncated=1/20
- **回合长度**：len=425.20
- **Checkpoint 轨迹**：官方得分从 cp1=-115.0 → cp5=-0.5 → cp10=245.7（先下降至 cp3=-176.9 后持续上升，cp8 接近收敛）

---

## 2. 骨架演进表

| iter | score | len | term (ter/tr) | proximity_progress_ep_sum | proximity_progress_active | soft_landing_bonus_ep_sum | soft_landing_bonus_active | attitude_penalty_ep_sum | attitude_penalty_active | landing_quality_ep_sum | landing_quality_active | landing_speed_ep_sum | landing_speed_active |
|-----|--------|-----|------------|------------------------|-------------------------|------------------------|-------------------------|-----------------------|------------------------|----------------------|----------------------|--------------------|--------------------|
| 1 | -2.98 | 813.35 | 5/15 | 36.74 | 99.9% | 8.05 | 0.3% | -1.35 | 100.0% | — | — | — | — |
| 2 | 235.03 | 425.20 | 19/1 | 26.81 | 96.7% | — | — | -2.01 | 100.0% | 219.68 | 71.7% | -1.05 | 68.9% |

---

## 3. 数字标记

- **主导**：`landing_quality` magnitude_share=87.8% > 80%
- **死组件**：无（所有活跃率均 > 2%）
- **淹没**：无（无组件满足 active_rate > 90% 且 magnitude_share < 3%）
- **全覆盖惩罚**：`attitude_penalty` active_rate=100.0% 且 signed_share 为负（-0.8%）
- **量级对比**：
  - 步均值 = ep_sum / len
  - `attitude_penalty`: -2.01 / 425.20 = -0.00473
  - `landing_quality`: 219.68 / 425.20 = 0.51656
  - `landing_speed`: -1.05 / 425.20 = -0.00247
  - `proximity_progress`: 26.81 / 425.20 = 0.06306
  - **最大两个**：`landing_quality`(0.517) > `proximity_progress`(0.063)
  - **最小两个**：`attitude_penalty`(-0.005) > `landing_speed`(-0.002)
- **修改变化**（v1→v2）：
  - `attitude_penalty`：ep_sum Δ=-0.66，active_rate Δ=0.0pp
  - `proximity_progress`：ep_sum Δ=-9.93，active_rate Δ=-3.2pp
  - `soft_landing_bonus`：已删除
  - `landing_quality`：新增，ep_sum=219.68，active_rate=71.7%
  - `landing_speed`：新增，ep_sum=-1.05，active_rate=68.9%

---

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 各轮活跃率 | 方向切换 |
|---------|----------|---------|----------|---------|
| 接近/进度信号 | v1_proximity_progress → v2_proximity_progress | 36.74 → 26.81 | 99.9% → 96.7% | 改善量：无切换 |
| 着陆/接触信号 | v1_soft_landing_bonus → v2_landing_quality | 8.05 → 219.68 | 0.3% → 71.7% | 接触门控→高度门控：1次切换；稀疏bonus→连续奖励：1次切换 |
| 稳定性/约束信号 | v1_attitude_penalty → v2_attitude_penalty | -1.35 → -2.01 | 100.0% → 100.0% | 无切换 |
| 完成/成功代理 | v1_(无) → v2_landing_speed | 无 → -1.05 | 无 → 68.9% | 新增：高度门控惩罚 |

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断)
- **obs[0]**: x_position – 相对于目标垫中心的水平坐标，reward_usable: true
- **obs[1]**: y_position – 相对于垫着陆面的垂直坐标（高度），reward_usable: true
- **obs[2]**: x_velocity – 水平线速度，reward_usable: true
- **obs[3]**: y_velocity – 垂直线速度，reward_usable: true
- **obs[4]**: body_angle – 车体倾角，reward_usable: true
- **obs[5]**: angular_velocity – 角速度，reward_usable: true
- **obs[6]**: left_support_contact – 左支撑腿接触标志（连续 1.0 或 0.0），reward_usable: true
- **obs[7]**: right_support_contact – 右支撑腿接触标志，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- **action 0**: no_engine – 不点火，自由运动  
- **action 1**: left_orientation_engine – 左姿态引擎点火（产生逆时针力矩）  
- **action 2**: main_engine – 主引擎点火（产生向上推力，同时可能改变姿态）  
- **action 3**: right_orientation_engine – 右姿态引擎点火（产生顺时针力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination**: `body_not_awake_or_settled` 中的 “settled” 情况（身体静止且可能已着陆在垫上），通常意味着成功。但不能仅靠该标记，因为也可能包含觉醒失败等。
- **failure‑like termination**: `crash_or_body_contact`（与危险物体碰撞或身体部分接触非目标表面）、`horizontal_position_outside_viewport`（飞出水平边界）
- **ambiguous termination**: `body_not_awake_or_settled` 在没有额外信息时可能是成功也可能是长时间静止在空中，需要结合位置和接触信号判断。
- **truncation**: 未提及（环境中无）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields:（info 返回 `{}`，没有任何可用字段）
- forbidden_or_uncertain_info_fields: 没有 info 字段，不能依赖任何隐藏终止原因；成功/失败需通过状态和终止组合推断。

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]` (x), `next_obs[1]` (y) – 接近目标水平的水平和垂直距离
- **velocity**: `next_obs[2]` (vx), `next_obs[3]` (vy) – 抑制高速，鼓励着陆时低速
- **orientation**: `next_obs[4]` (angle), `next_obs[5]` (angular velocity) – 姿态稳定
- **contact**: `next_obs[6]`, `next_obs[7]` – 接触状态，可用于奖励正确着垫
- **action/engine**: `action` – 可用于惩罚不必要的引擎使用，且动作类型影响姿态和推力
- **other**: 可结合上述信号构建复合指标（如距离惩罚、摇晃惩罚）