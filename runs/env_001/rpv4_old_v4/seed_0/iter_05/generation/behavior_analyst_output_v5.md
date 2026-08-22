# 训练数据报告

## 1. 行为事实
- 得分：score=-59.19，range=[-113.83, -7.75]
- 终止方式：terminated 1/20（成功着陆），truncated 19/20（到达时间上限）
- 回合长度：len=977.9 步（环境最大长度 1000）
- Checkpoint 轨迹：官方得分 cp10%=-694.8 → cp20%=-533.5 → cp30%=-263.2 → cp40%=-132.8 → cp50%=-81.5 → cp60%=-71.6 → cp70%=-68.6 → cp80%=-70.1 → cp90%=-77.5 → cp100%=-59.0。整体持续上升，cp70–cp90 出现小幅回落后最终上升。

## 2. 骨架演进表
| iter | score | len | term (t/tr) | contact_reward_ep_sum | contact_reward_active | landing_proximity_reward_ep_sum | landing_proximity_reward_active | progress_reward_ep_sum | progress_reward_active | orientation_penalty_ep_sum | orientation_penalty_active | vel_damping_ep_sum | vel_damping_active |
|------|--------|------|-------------|------------------------|----------------------|--------------------------------|--------------------------------|------------------------|------------------------|---------------------------|----------------------------|--------------------|--------------------|
| 1    | -159.49 | 303.35 | 20/0 | 0.0 | 0.0% | - | - | 0.104 | 100.0% | -0.034 | 100.0% | 0.0 | 0.0% |
| 2    | -36.38 | 1000.0 | 0/20 | - | - | 2704.041 | 100.0% | 1.037 | 100.0% | -0.302 | 100.0% | -26.778 | 95.3% |
| 3    | -434.87 | 967.2 | 1/19 | - | - | 25.542 | 68.3% | -1.975 | 100.0% | -0.874 | 100.0% | 0.0 | 0.0% |
| 4    | -59.19 | 977.9 | 1/19 | - | - | 4731.133 | 100.0% | 0.837 | 100.0% | -0.558 | 100.0% | -21.756 | 92.1% |

（“-”表示该轮不存在该组件）

## 3. 数字标记
- **主导**：landing_proximity_reward magnitude_share=99.5%（>80%）
- **死组件**：无（所有组件 active_rate ≥ 92.1%）
- **淹没**：
  - progress_reward active_rate=100%，magnitude_share=0.0%
  - orientation_penalty active_rate=100%，magnitude_share=0.0%
- **全覆盖惩罚**：orientation_penalty active_rate=100%（>95%）且 signed_share=-0.0%（负），满足条件
- **量级对比**（每步均值 = episode_sum_mean ÷ len）：
  - 最大两个：landing_proximity_reward ≈ 4.838，progress_reward ≈ 0.00086
  - 最小两个：vel_damping ≈ -0.0222，orientation_penalty ≈ -0.00057
- **修改变化**（iter3 → iter4，Δ 值）：
  - landing_proximity_reward: ep_sum Δ +4705.59， active_rate Δ +31.7%
  - progress_reward: ep_sum Δ +2.81， active_rate Δ 0%
  - orientation_penalty: ep_sum Δ +0.32， active_rate Δ 0%
  - vel_damping: ep_sum Δ -21.76， active_rate Δ +92.1%

## 4. 语义演化（跨轮组件功能追踪）
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|--------|---------|
| 着陆/接触信号 | iter1: contact_reward → iter2: landing_proximity_reward → iter3: landing_proximity_reward → iter4: landing_proximity_reward | 0.0 → 2704.04 → 25.54 → 4731.13 | 接触标志条件→连续状态高斯（1次），状态高斯→改善量（2次），改善量→状态高斯（3次）<br>其中**状态值↔改善量切换 2 次** |
| 接近/进度信号 | iter1: progress_reward → iter2: progress_reward → iter3: progress_reward → iter4: progress_reward | 0.104 → 1.037 → -1.975 → 0.837 | 无（始终为改善量形式） |
| 姿态稳定 | iter1: orientation_penalty → iter2: orientation_penalty → iter3: orientation_penalty → iter4: orientation_penalty | -0.034 → -0.302 → -0.874 → -0.558 | 无 |
| 速度阻尼 | iter1: vel_damping → iter2: vel_damping → iter3: vel_damping → iter4: vel_damping | 0.0 → -26.778 → 0.0 → -21.756 | 无（始终为门控速度惩罚，仅激活状态变化） |