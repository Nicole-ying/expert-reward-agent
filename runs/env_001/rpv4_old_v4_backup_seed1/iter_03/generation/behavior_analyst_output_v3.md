# 训练数据报告

## 1. 行为事实
- 得分：score=63.720709，range=[-208.175480, 265.833071]
- 终止方式：terminated 17/20，truncated 3/20
- 回合长度：len=668.050000
- Checkpoint 轨迹：官方得分从 cp10%=-73.8 → cp20%=-168.1 → cp30%=-119.8 → cp40%=-97.7 → cp50%=-92.9 → cp60%=-2.9 → cp70%=70.8 → cp80%=-44.1 → cp90%=-17.7 → cp100%=48.7（波动，先降后升再到正分）

## 2. 骨架演进表
| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | goal_proximity_ep_sum | goal_proximity_active | upright_penalty_ep_sum | upright_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -76.561074 | 975.150000 | terminated=1/20, truncated=19/20 | 0.000000 | 0.0% | 0.635053 | 100.0% | -3.714416 | 100.0% | -12.561269 | 100.0% |
| 2 | 63.720709 | 668.050000 | terminated=17/20, truncated=3/20 | 1075.630017 | 5.2% | 1.835102 | 98.8% | -7.300626 | 100.0% | -9.516978 | 99.0% |

## 3. 数字标记

### 主导
- **contact_bonus**：magnitude_share=98.2%（>80%），主导总奖励量级

### 死组件
- 无组件 active_rate < 2%（contact_bonus 为 5.2%，未触发阈值）

### 淹没
- 无组件同时满足 active_rate > 90% 且 magnitude_share < 3%（goal_proximity 为 98.8%/0.3%，upright_penalty 为 100%/0.7%，velocity_penalty 为 99.0%/0.9%，均未同时触发双条件）

### 全覆盖惩罚
- 无组件同时满足 active_rate > 95% 且 signed_share 为负

### 量级对比
- 每步均值（ep_sum ÷ len）：
  - contact_bonus：1075.630017 ÷ 668.05 = **1.610**（最大）
  - goal_proximity：1.835102 ÷ 668.05 = **0.00275**（第二大，绝对值）
  - velocity_penalty：-9.516978 ÷ 668.05 = **-0.01425**（第三大，绝对值）
  - upright_penalty：-7.300626 ÷ 668.05 = **-0.01093**（最小，绝对值）
- 最大的两个：contact_bonus（1.610），velocity_penalty（0.01425，绝对值）
- 最小的两个：goal_proximity（0.00275，绝对值），upright_penalty（0.01093，绝对值）

### 修改变化（iter 2 vs iter 1）
- contact_bonus：ep_sum Δ=+1075.630017，active_rate Δ=+5.2%
- goal_proximity：ep_sum Δ=+1.200049，active_rate Δ=-1.2%
- upright_penalty：ep_sum Δ=-3.586210，active_rate Δ=0.0%
- velocity_penalty：ep_sum Δ=+3.044291（惩罚减轻），active_rate Δ=-1.0%

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | v1_goal_proximity → v2_goal_proximity | -76.56 → 63.72 | 无切换（均为delta距离改善，权重从1.0→2.0） |
| 着陆/接触信号 | v1_contact_bonus(死) → v2_contact_bonus | -76.56(0.00) → 63.72(1075.63) | 稀疏bonus形态未变：0次切换 |
| 稳定性/约束信号 | v1_velocity_penalty → v2_velocity_penalty | -76.56(-12.56) → 63.72(-9.52) | 全高度→高度门控：1次切换 |
| 稳定性/约束信号 | v1_upright_penalty → v2_upright_penalty | -76.56(-3.71) → 63.72(-7.30) | 无切换（均为连续二次型惩罚） |