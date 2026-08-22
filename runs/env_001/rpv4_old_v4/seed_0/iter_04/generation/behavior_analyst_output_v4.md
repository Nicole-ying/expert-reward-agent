# 训练数据报告

## 1. 行为事实
- 得分：score=-434.871335, range=[-901.456076, -164.536235]
- 终止方式：terminated 1/20, truncated 19/20
- 回合长度：len=967.2
- Checkpoint 轨迹：官方得分从 cp1=-758.2 迅速上升至 cp2=-236.2，随后波动，cp5=-211.8 后又下降，最终 cp10=-514.5。总体先快速改善后明显恶化。

## 2. 骨架演进表

| iter | score | len | term | progress_ep_sum | progress_active | orient_ep_sum | orient_active | vel_damp_ep_sum | vel_damp_active | contact_ep_sum | contact_active | land_prox_ep_sum | land_prox_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -159.49 | 303.35 | 20T/0Tr | 0.104125 | 100% | -0.033500 | 100% | 0.000 | 0% | 0.000 | 0% | - | - |
| 2 | -36.38 | 1000.0 | 0T/20Tr | 1.036857 | 100% | -0.302478 | 100% | -26.778066 | 95.3% | - | - | 2704.041458 | 100% |
| 3 | -434.87 | 967.2 | 1T/19Tr | -1.975256 | 100% | -0.874189 | 100% | 0.000 | 0% | - | - | 25.541566 | 68.3% |

*注：`-` 表示该组件在该轮不存在。*

## 3. 数字标记
- **主导**：无组件 magnitude_share > 80％
- **死组件**：vel_damping（active_rate=0%）
- **淹没**：orientation_penalty（active_rate=100%, magnitude_share=2.1%）
- **全覆盖惩罚**：progress_reward（active_rate=100%, signed_share=-4.7%）；orientation_penalty（active_rate=100%, signed_share=-2.1%）
- **量级对比**（每步均值）：
  - 最大两个：landing_proximity_reward（0.0264），vel_damping（0.0）
  - 最小两个：progress_reward（-0.00204），orientation_penalty（-0.00090）
- **修改变化**（iter 3 vs iter 2）：
  - landing_proximity_reward: ep_sum Δ = -2678.5, active_rate Δ = -31.7 pp
  - progress_reward: ep_sum Δ = -3.01, active_rate Δ = 0 pp
  - orientation_penalty: ep_sum Δ = -0.57, active_rate Δ = 0 pp
  - vel_damping: ep_sum Δ = +26.78, active_rate Δ = -95.3 pp

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1: progress_reward → v2: progress_reward, landing_proximity_reward → v3: progress_reward, landing_proximity_reward | v1: 0.10 → v2: 2705.1 → v3: 23.6 | 状态值→改善量（landing_proximity_reward 在 v2→v3）：1次 |
| 着陆/接触信号 | v1: contact_reward → v2: landing_proximity_reward → v3: landing_proximity_reward | v1: 0.00 → v2: 2704.0 → v3: 25.5 | 接触条件连续→距离基连续→进度加权：2次 |
| 稳定性/约束信号 | v1: orientation_penalty, vel_damping → v2: orientation_penalty, vel_damping → v3: orientation_penalty, vel_damping | v1: -0.034 → v2: -27.1 → v3: -0.874 | 无切换（均为二次惩罚+门控速度衰减） |

*得分采用语义功能内所有相关组件的 episode_sum_mean 总和（近似）。*