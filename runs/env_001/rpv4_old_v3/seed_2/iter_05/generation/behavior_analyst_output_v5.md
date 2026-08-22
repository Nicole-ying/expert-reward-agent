# 训练数据报告 — 第4轮

## 1. 行为事实
- 得分：score=-114.38，range=[-138.75, -99.23]
- 终止方式：terminated 20/20，truncated 0/20
- 回合长度：len=68.45步
- Checkpoint 轨迹：40%: score=0.00, 80%: score=0.00（全程持平0）

## 2. 骨架演进表
| iter | score | len | term | progress_shaping_ep_sum | progress_shaping_active | action_cost_ep_sum | action_cost_active | angle_penalty_ep_sum | angle_penalty_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active | success_landing_bonus_ep_sum | success_landing_bonus_active | crash_penalty_ep_sum | crash_penalty_active |
|------|--------|-------|------|--------------------------|--------------------------|---------------------|--------------------|-----------------------|-----------------------|---------------------------------|---------------------------------|-------------------------------|-------------------------------|-----------------------|-----------------------|
| 1 | -155.98 | 154.40 | 20/20 | 1.582 | 100.0% | -1.804 | 58.4% | -1.141 | 100.0% | -0.067 | 5.3% | - | - | - | - |
| 2 | -119.51 | 68.30 | 20/20 | 5.583 | 100.0% | -0.071 | 5.2% | -0.205 | 100.0% | - | - | 5.000 | 0.1% | - | - |
| 3 | -203.45 | 154.15 | 20/20 | 1.488 | 100.0% | -1.472 | 47.7% | -2.460 | 100.0% | - | - | 0.000 | 0.0% | 0.000 | 0.0% |
| 4 | -114.38 | 68.45 | 20/20 | 5.597 | 100.0% | -0.097 | 7.1% | -0.087 | 100.0% | - | - | 30.000 | 0.4% | - | - |

## 3. 数字标记
- **主导**：success_landing_bonus (magnitude_share=83.4% > 80%)
- **死组件**：success_landing_bonus (active_rate=0.4% < 2%)
- **淹没**：angle_penalty (active_rate=100% > 90%, magnitude_share=0.2% < 3%)
- **全覆盖惩罚**：angle_penalty (active_rate=100% > 95%, signed_share=-0.2%)
- **量级对比（每步均值）**：
  - 最大两个：success_landing_bonus (0.438/step), progress_shaping (0.082/step)
  - 最小两个：action_cost (-0.00142/step), angle_penalty (-0.00127/step)
- **修改变化（vs iter3）**：
  - crash_penalty：移除（v3 ep_sum=0.0, active=0%）  
  - success_landing_bonus：ep_sum Δ=+30.0；active_rate Δ=+0.4%
  - progress_shaping：ep_sum Δ=+4.109；active_rate Δ=0%
  - angle_penalty：ep_sum Δ=+2.373；active_rate Δ=0%
  - action_cost：ep_sum Δ=+1.375；active_rate Δ=-40.6%

## 4. 语义演化（跨轮组件功能追踪）

### 着陆事件
- 组件轨迹：v1_landing_velocity_penalty → v2_success_landing_bonus → v3_success_landing_bonus + crash_penalty → v4_success_landing_bonus
- 各轮得分：-0.067 → 5.0 → 0.0 / 0.0 → 30.0
- **方向切换**：连续惩罚→稀疏奖励：1次切换 (v1→v2)

### 接近/进度
- 组件轨迹：v1~v4 均为 progress_shaping
- 各轮得分：1.582 → 5.583 → 1.488 → 5.597
- 方向切换：无（始终为连续势能奖励）

### 稳定性约束
- 组件轨迹：v1~v4 存在 angle_penalty + action_cost (v1 额外含 landing_velocity_penalty 已归入着陆事件)
- 各轮得分：-1.141/-1.804 → -0.205/-0.071 → -2.460/-1.472 → -0.087/-0.097
- 方向切换：无（始终为连续小惩罚）