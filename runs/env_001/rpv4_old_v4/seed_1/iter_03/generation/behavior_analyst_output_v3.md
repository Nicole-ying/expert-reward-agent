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