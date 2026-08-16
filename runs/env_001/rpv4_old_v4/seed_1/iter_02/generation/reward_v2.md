---

## 1. 诊断报告

### evidence

| 数据源 | 关键发现 |
|---|---|
| `read_training_feedback(1)` | score=-2.98, terminated=5/20 (25%), truncated=15/20。`soft_landing_bonus` active_rate=0.3%（死组件），`proximity_progress` magnitude_share=82.9%（主导），`attitude_penalty` magnitude_share=2.5%（淹没） |
| `read_checkpoint_trend(1)` | 训练奖励从 cp10%=-72.4 升至 cp80%=164.9，随后回落至 cp100%=122.7；`soft_landing_bonus` ep_sum 在 cp80% 暴增至 256.5，但 final eval 仅 8.05——**典型的训练期 exploit**：agent 学会在训练中触发 contact bonus 刷分，但该行为不转化为真正的着陆能力 |
| `read_memory()` | 仅 1 轮迭代，score=-2.98，无历史可比较 |
| 环境卡片 | 着陆任务需要：接近目标、低速、水平姿态、双腿接触。无显式成功/失败 flag，info 为空 |

### behavior_diagnosis

Agent 当前行为：学会向目标靠近（`proximity_progress` 99.9% 激活，贡献 82.9% 信号），但**不会着陆**。75% 的 episode 被 truncate（超时），说明 agent 在目标附近徘徊但不完成着陆。`soft_landing_bonus` 仅在 0.3% 的 step 触发——agent 几乎从未接收到"如何软着陆"的反馈。训练过程中 checkpoint 数据显示 agent 曾大量触发 contact bonus（cp80%: 256 ep_sum），但这种行为无法泛化到评估 episode（eval 仅 8.05），说明 agent 利用训练中的噪声探索到了 contact bonus 的触发方式，但该信号太稀疏以至于无法形成稳定策略。

### signal_completeness

| 职责 | 状态 | 说明 |
|---|---|---|
| goal_proximity | ✅ 已覆盖 | `proximity_progress` 有效引导接近行为 |
| soft_landing_contact | ❌ 信号稀疏 | 三重乘积（contact×velocity×angle）导致 99.7% 的 step 为零，无梯度 |
| attitude_stabilization | ⚠️ 力度不足 | 全覆盖但 magnitude_share 仅 2.5%，被 proximity 淹没 |
| 近地速度约束 | ❌ 缺失 | 没有针对近地阶段的速度惩罚，agent 可能高速砸地 |

核心缺陷：**着陆反馈是接触门控而非高度门控**。接触事件（双腿同时触地）在 800+ 步 episode 中仅发生极少次数，导致信用分配完全失败。

### selected_level

**L2** — 改一个组件的数学形态。

理由：`soft_landing_bonus` 的数学结构（contact-gated 三重乘积）有明确的稀疏性缺陷，仅调系数（L1）无法解决 0×任何数=0 的问题。需要将门控机制从"接触触发"改为"高度触发"。

### selected_intervention

**语义方向**：将着陆反馈从接触门控改为高度门控。

具体修改：
1. **删除** `soft_landing_bonus`（contact-gated 稀疏信号）
2. **新增** `landing_speed` — 高度门控速度惩罚：当 `y_pos < 0.5` 时，按速度平方惩罚，鼓励近地减速
3. **新增** `landing_quality` — 高度门控正向品质奖励：近地时，低速度+水平姿态获得指数衰减奖励
4. **微调** `proximity_progress` 系数 30→20，为着陆信号留出权重空间

这服务于同一语义目标：**让 agent 在接近地面时获得稠密的着陆行为反馈**。

### falsifiable_hypothesis

如果高度门控有效，agent 将在近地区域接收到关于速度和姿态的连续梯度信号，从而学会在下降最后阶段减速+摆正姿态。这将表现为：
- `landing_speed` 和 `landing_quality` 的 active_rate 应远高于旧 `soft_landing_bonus` 的 0.3%（预期 > 20%，因为 y<0.5 时持续激活）
- terminated 比例应上升（agent 更频繁地完成着陆）
- checkpoint 趋势中的 exploit 峰值应消失（高度门控无法像接触门控那样被单次事件刷分）

### expected_next_round

| 指标 | 当前 | 预期 | 理由 |
|---|---|---|---|
| score | -2.98 | 正值（可能 50-150） | 着陆反馈改善 → 更多成功着陆 |
| terminated | 5/20 | 8-15/20 | 学会减速着陆 |
| `landing_speed` active_rate | N/A | 15-40% | 近地阶段持续激活 |
| `landing_quality` active_rate | N/A | 15-40% | 同上 |
| `proximity_progress` magnitude_share | 82.9% | 40-60% | 着陆组件分担信号 |
| 训练 cp 中的 exploit 峰值 | cp80%=256 | 消失或大幅减弱 | 高度门控无法被单次事件 exploit |

### main_risk

高度门控阈值 0.5 是硬编码的——如果 agent 在 y=0.5 附近徘徊（刚好在 gate 边界上），可能产生策略震荡。缓解措施：gate 用线性 ramp（0→1 平滑过渡），避免阶跃。另外，如果阈值 0.5 与实际着陆面高度不匹配（y=0 是垫面），可能导致 gate 过早/过晚激活。若下轮 `landing_speed` active_rate 异常高（>80%）且 score 仍低，说明阈值需调整。

---

## 2. 完整代码

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