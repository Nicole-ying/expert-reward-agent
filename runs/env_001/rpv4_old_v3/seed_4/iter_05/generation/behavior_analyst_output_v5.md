# 训练数据报告

## 1. 行为事实
- 得分：score=24.69，range=[-84.41, 209.99]
- 终止方式：terminated 10/20（成功），truncated 10/20（时间上限）
- 回合长度：len=754.05
- Checkpoint 轨迹：cp1=0.00 → cp2=0.00（持平）
- 早终：0/20（early_terminal <150 steps and score<-50）

## 2. 骨架演进表
| iter | score | len | term | pos_ep_sum | pos_active | contact_ep_sum | contact_active | vel_ep_sum | vel_active | angle_ep_sum | angle_active | angvel_ep_sum | angvel_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.74 | 341.75 | 15/5 | -88.13 | 100.0% | 555.99 | 31.9% | -5.30 | 100.0% | -0.44 | 100.0% | -1.24 | 92.7% |
| 2 | 64.24 | 924.05 | 10/10 | 3.55 | 99.9% | 305.31 | 12.0% | -1.50 | 99.8% | -0.55 | 100.0% | -0.37 | 99.2% |
| 3 | -21.40 | 1000.00 | 0/20 | 59.09 | 100.0% | 0.00 | 0.0% | -1.08 | 100.0% | -0.24 | 100.0% | -0.18 | 100.0% |
| 4 | 24.69 | 754.05 | 10/10 | 7.91 | 100.0% | 5.37 | 0.5% | -1.68 | 100.0% | -0.45 | 100.0% | -0.31 | 99.9% |

## 3. 数字标记

- **主导**：无组件 magnitude_share > 80%
- **死组件**：无组件 active_rate < 2%（contact=0.5%，接近但未达标）
- **淹没**：无组件同时满足 active_rate > 90% 且 magnitude_share < 3%
- **全覆盖惩罚**：无组件同时满足 active_rate > 95% 且 signed_share 为负
- **量级对比**（每步均值 = ep_sum ÷ len）：
  - 最大：contact=0.0071，pos=0.0105
  - 最小：angvel=-0.0004，angle=-0.0006
- **修改变化（iter_3 → iter_4）**：
  - pos：ep_sum Δ = -51.18，active_rate Δ = 0.0pp
  - contact：ep_sum Δ = +5.37，active_rate Δ = +0.5pp
  - vel：ep_sum Δ = -0.60，active_rate Δ = 0.0pp
  - angle：ep_sum Δ = -0.21，active_rate Δ = 0.0pp
  - angvel：ep_sum Δ = -0.13，active_rate Δ = -0.1pp

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1_pos → v2_pos → v3_pos → v4_pos | -88.13 → 3.55 → 59.09 → 7.91 | 状态值→改善量：1次切换（v1为平方距离惩罚，v2起改为delta） |
| 着陆/接触信号 | v1_contact → v2_contact → v3_contact → v4_contact | 555.99 → 305.31 → 0.00 → 5.37 | 无方向切换（始终稀疏bonus：both legs × distance_factor） |
| 稳定性/约束信号 | v1_vel/angle/angvel → v2_vel/angle/angvel → v3_vel/angle/angvel → v4_vel/angle/angvel | -6.98 → -2.41 → -1.49 → -2.45 | 无方向切换（始终门控惩罚：gate × squared penalty） |

**方向切换**列：接近信号在v1→v2发生1次"状态值→改善量"切换。着陆信号和稳定性信号均未发生方向切换。整体无振荡反思迹象。