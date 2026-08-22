# 训练数据报告 — 迭代 N=2

## 1. 行为事实
- 得分：score=-37.03，range=[-119.36, 188.51]
- 终止方式：terminated 4/20（成功着陆），truncated 16/20（时间上限）
- 回合长度：len=962.55
- Checkpoint 轨迹：官方得分 cp10%=-217.1 → cp50%=-58.6 → cp80%=-5.4 → cp100%=-29.4（整体上升，自 cp80% 后维持在正值附近）

## 2. 骨架演进表
| iter | score | len | term | landing_ep_sum | landing_active | speed_penalty_ep_sum | speed_penalty_active | angular_vel_penalty_ep_sum | angular_vel_penalty_active | progress_ep_sum | progress_active |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|---:|---:|
| 1 | -203.53 | 185.80 | 20/0/0 | 0.00 | 0.0% | -0.67 | 15.5% | -0.43 | 100.0% | -0.20 | 100.0% |
| 2 | -37.03 | 962.55 | 4/0/16 | 697.44 | 99.6% | -42.13 | 100.0% | -2.46 | 100.0% | 0.66 | 100.0% |

## 3. 数字标记
- **主导**：landing magnitude_share=93.8% > 80%
- **死组件**：无
- **淹没**：angular_vel_penalty active_rate=100.0%、magnitude_share=0.3% < 3%；progress active_rate=100.0%、magnitude_share=0.2% < 3%
- **全覆盖惩罚**：speed_penalty active_rate=100.0% > 95%、signed_share=-5.7% 为负，但 magnitude_share=5.7% ≥ 3%，不符合淹没条件，标记为高活跃抑制信号
- **量级对比**：
  - 每步均值最大：landing=0.725（697.44÷962.55）
  - 每步均值第二大：speed_penalty=-0.044（42.13÷962.55，绝对值）
  - 每步均值最小：progress=0.001（0.66÷962.55，绝对值）
  - 每步均值第二小：angular_vel_penalty=-0.003（2.46÷962.55，绝对值）
- **修改变化（iter_1 → iter_2）**：
  - landing：ep_sum Δ=+697.44，active_rate Δ=+99.6pp
  - speed_penalty：ep_sum Δ=-41.46，active_rate Δ=+84.5pp
  - angular_vel_penalty：ep_sum Δ=-2.03，active_rate Δ=0
  - progress：ep_sum Δ=+0.86，active_rate Δ=0

## 4. 语义演化

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆/接触信号 | iter_1:landing → iter_2:landing | -203.53 → -37.03 | 稀疏bonus（product-of-factors）→ 连续奖励（sum-of-factors）：1次切换 |
| 接近/进度信号 | iter_1:progress → iter_2:progress | -203.53 → -37.03 | 无切换（均为delta距离） |
| 稳定性/约束信号 | iter_1:speed_penalty → iter_2:speed_penalty<br>iter_1:angular_vel_penalty → iter_2:angular_vel_penalty | -203.53 → -37.03 | speed_penalty：低活跃（15.5%）→ 全覆盖（100.0%），保持门控设计<br>angular_vel_penalty：全覆盖→全覆盖，无设计切换 |