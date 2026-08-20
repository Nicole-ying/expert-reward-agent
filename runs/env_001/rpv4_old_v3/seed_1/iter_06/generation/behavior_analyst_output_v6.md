# 训练数据报告

## 1. 行为事实
- **得分**：score=-119.003231，range=[-144.398763, -101.141368]
- **终止方式**：terminated 20/20（全部为着陆终止），truncated 0/20（无时间截断）
- **回合长度**：len=68.400000
- **Checkpoint 轨迹**：官方评估得分 40% 步数=0.00，80% 步数=0.00，持平为0.00，无上升或下降趋势。

## 2. 骨架演进表
所有 ep_sum 为 episode_sum_mean，active 为 active_rate（%）。组件不存在时记为 0 或 `-`（尽量填写数值 0）。

| iter | score | len | term | goal_approach_ep_sum | goal_approach_active | soft_landing_ep_sum | soft_landing_active | crash_avoidance_ep_sum | crash_avoidance_active | landing_success_ep_sum | landing_success_active | contact_reward_ep_sum | contact_reward_active |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -214.16 | 182.50 | 20/20 terminated | 0.152 | 100% | 0.0 | 0% | 0.0 | 0% | 0.0 | 0% | 0.0 | 0% |
| 2 | -109.58 | 68.45 | 20/20 terminated | 1.120 | 100% | -19.091 | 100% | 0.0 | 0% | 0.0 | 0% | 0.0 | 0% |
| 3 | -115.21 | 68.35 | 20/20 terminated | 1.119 | 100% | -20.744 | 100% | 0.0 | 0% | 25.0 | 0.4% | 0.0 | 0% |
| 4 | -111.06 | 68.40 | 20/20 terminated | 1.120 | 100% | -19.297 | 100% | 0.0 | 0% | 0.0 | 0% | 1.15 | 1.7% |
| 5 | -119.00 | 68.40 | 20/20 terminated | 1.117 | 100% | -20.641 | 100% | 0.0 | 0% | 25.0 | 0.4% | 0.0 | 0% |

## 3. 数字标记
- **主导**：无组件 magnitude_share > 80%（最高为 landing_success 53.4%）。
- **死组件**：landing_success（active_rate=0.4% < 2%）。
- **淹没**：goal_approach（active_rate=100% > 90%，但 magnitude_share=2.5% < 3%）。
- **全覆盖惩罚**：soft_landing（active_rate=100% > 95%，signed_share=-44.1% < 0）。
- **量级对比**（每步均值 ep_sum / len）：
  - 最大两个：landing_success (+0.3655)，goal_approach (+0.0163)
  - 最小两个：soft_landing (-0.3018)，goal_approach (+0.0163)  
  *注：goal_approach 同时为第二大和第二小，因仅有三个组件。*
- **修改变化（iter4 → iter5）**：
  - goal_approach：Δ ep_sum = -0.0035，Δ active = 0 pp
  - soft_landing：Δ ep_sum = -1.3436，Δ active = 0 pp
  - landing_success（iter4 不存在）：Δ ep_sum = +25.0，Δ active = +0.4 pp
  - contact_reward（iter5 移除）：Δ ep_sum = -1.15，Δ active = -1.7 pp

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|--------|---------|
| 接近/进度信号 | v1_goal_approach → v2_goal_approach → v3_goal_approach → v4_goal_approach → v5_goal_approach | -214.16 → -109.58 → -115.21 → -111.06 → -119.00 | 无（始终为连续potential-based shaping） |
| 稳定性/约束信号 | v1_soft_landing(死，门控二次) → v2_soft_landing(全高度线性) → … → v5_soft_landing | -214.16 → -109.58 → -115.21 → -111.06 → -119.00 | 门控→全高度：1次切换 |
| 着陆/接触信号 | v1_crash_avoidance(死) → v2_crash_avoidance(死) → v4_contact_reward(连续奖励) | -214.16 → -109.58 → -111.06 | 稀疏惩罚→连续奖励：1次切换 |
| 完成/成功代理 | v3_landing_success → v5_landing_success | -115.21 → -119.00 | 无（均为稀疏大额bonus，条件阈值放宽但未发生对立切换） |