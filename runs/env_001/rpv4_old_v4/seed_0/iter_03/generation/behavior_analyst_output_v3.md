# 训练数据报告（iter_02）

---

## 1. 行为事实

- **得分**：score=-36.380930，range=[-66.996077, -5.388459]
- **终止方式**：terminated=0/20（成功着陸 0 集），truncated=20/20（全为达到时间上限 1000 步后被截断）
- **回合长度**：len=1000.000000（达到步数上限）
- **Checkpoint 轨迹（官方得分）**：cp10%=-974.2 → cp20%=-566.6 → cp30%=-328.5 → cp40%=-114.1 → cp50%=-76.9 → cp60%=-146.3 → cp70%=-100.0 → cp80%=-58.5 → cp90%=-140.0 → cp100%=-34.4（整体从 -974.2 上升至 -34.4，中间有波动，cp50%→cp60%下降 69.4，cp80%→cp90%下降 81.5，cp90%→cp100%上升 105.6）

---

## 2. 骨架演进表

| iter | score | len | term | landing_proximity_reward_ep_sum | landing_proximity_reward_active | orientation_penalty_ep_sum | orientation_penalty_active | progress_reward_ep_sum | progress_reward_active | vel_damping_ep_sum | vel_damping_active |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -159.490304 | 303.35 | term=20, trunc=0 | — | — | -0.033500 | 100.0% | 0.104125 | 100.0% | 0.000000 | 0.0% |
| 2 | -36.380930 | 1000.00 | term=0, trunc=20 | 2704.041458 | 100.0% | -0.302478 | 100.0% | 1.036857 | 100.0% | -26.778066 | 95.3% |

*注：iter_01 的骨架为 `contact_reward + orientation_penalty + progress_reward + vel_damping`，其中 contact_reward 在 iter_02 被 landing_proximity_reward 替换，故不在本表列出。*

---

## 3. 数字标记

- **主导**：`landing_proximity_reward` 的 magnitude_share = 99.0%（> 80%），该组件在量级上完全主导 reward 构成
- **死组件**：无组件 active_rate < 2%
- **淹没**：无组件满足 active_rate > 90% 且 magnitude_share < 3%（`progress_reward` active_rate=100% 但 magnitude_share=0.0%，接近淹没边界；`orientation_penalty` active_rate=100% 但 magnitude_share=0.0%，接近淹没边界）
- **全覆盖惩罚**：无组件 active_rate > 95% 且 signed_share 为负（`vel_damping` active_rate=95.3% 但 signed_share=-1.0%，不符合全覆盖惩罚条件）
- **量级对比（每步均值 ep_sum ÷ len）**：
  - 最大：`landing_proximity_reward` = 2704.04/1000 = 2.704
  - 次大：`vel_damping` = -26.78/1000 = -0.027
  - 次小：`progress_reward` = 1.04/1000 = 0.001
  - 最小：`orientation_penalty` = -0.30/1000 = -0.0003
- **修改变化（iter_01 → iter_02）**：
  - `landing_proximity_reward`（新组件，替换 contact_reward）：ep_sum 从 0.000 → 2704.041，active_rate 从 0.0% → 100.0%
  - `orientation_penalty`：ep_sum 从 -0.034 → -0.302（Δ=-0.269），active_rate 保持 100.0%（Δ=0.0%）
  - `progress_reward`：ep_sum 从 0.104 → 1.037（Δ=+0.933），active_rate 保持 100.0%（Δ=0.0%）
  - `vel_damping`：ep_sum 从 0.000 → -26.778（Δ=-26.778），active_rate 从 0.0% → 95.3%（Δ=+95.3%）

---

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/着陆综合信号 | v1:contact_reward(死)→v2:landing_proximity_reward(主导) | -159.5→-36.4 | 接触条件→连续高斯乘积，1次切换 |
| 稳定性/约束信号 | v1:orientation_penalty→v2:orientation_penalty | -159.5→-36.4 | 无切换（同为二次惩罚） |
| 接近/进度信号 | v1:progress_reward→v2:progress_reward | -159.5→-36.4 | 无切换（同为改善量delta） |
| 近地速度阻尼 | v1:vel_damping(死)→v2:vel_damping(活跃) | -159.5→-36.4 | 同为门控惩罚，无切换 |