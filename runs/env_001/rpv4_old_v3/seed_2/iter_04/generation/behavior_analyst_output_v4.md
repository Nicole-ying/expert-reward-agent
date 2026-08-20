# 训练数据报告 — iter_03（本轮）

## 1. 行为事实
- **得分**：score=-203.45，range=[-270.27, -158.78]
- **终止方式**：全部因接触地面或越界而提前终止（terminated=20/20）；无时间上限截断（truncated=0）。
- **回合长度**：len=154.15
- **Checkpoint 轨迹**：40% 步数时得分 0.00，80% 步数时得分 0.00（均无提升）。
- **奖励组成**：
  - 正向贡献仅有 `progress_shaping` `+1.49`（sign=+17.9%，mag=52.6%）。
  - 惩罚端主要由 `angle_penalty`（-2.46）和 `action_cost`（-1.47）构成。
  - `crash_penalty` 与 `success_landing_bonus` 均完全未触发（active_rate=0.0%，ep_sum=0.0）。

## 2. 骨架演进表

| iter | score | len | term | progress_shaping_ep_sum | progress_active | success_landing_bonus_ep_sum | success_active | landing_velocity_penalty_ep_sum | lvp_active | crash_penalty_ep_sum | crash_active | angle_penalty_ep_sum | angle_active | action_cost_ep_sum | action_active |
|------|--------|------|------|--------------------------|-----------------|------------------------------|----------------|---------------------------------|-------------|----------------------|--------------|----------------------|--------------|--------------------|---------------|
| 1 | -155.98 | 154.40 | 20/0 | 1.58 | 100.0% | — | — | -0.07 | 5.3% | — | — | -1.14 | 100.0% | -1.80 | 58.4% |
| 2 | -119.51 | 68.30 | 20/0 | 5.58 | 100.0% | 5.00 | 0.1% | — | — | — | — | -0.20 | 100.0% | -0.07 | 5.2% |
| 3 | -203.45 | 154.15 | 20/0 | 1.49 | 100.0% | 0.00 | 0.0% | — | — | 0.00 | 0.0% | -2.46 | 100.0% | -1.47 | 47.7% |

*ter: terminated/truncated。active = active_rate。*

## 3. 数字标记

- **全覆盖惩罚**：`angle_penalty` active_rate=100.0% 且 signed_share=-29.7%，属于持续激活的纯惩罚项。
- **死组件**：`crash_penalty` active_rate=0.0%，`success_landing_bonus` active_rate=0.0%。两者在 iter_03 中完全未生效。
- **量级对比**（每步均值 ep_sum÷len）：
  - 最大正贡献：`progress_shaping` = +0.00965
  - 最大负贡献：`angle_penalty` = -0.01596
  - 最小两个：`crash_penalty` = 0.0，`success_landing_bonus` = 0.0
- **修改变化（iter_02 → iter_03 Δ）**：
  | 组件 | Δ ep_sum | Δ active_rate |
  |------|----------|---------------|
  | progress_shaping | -4.10 | 0.0pp |
  | success_landing_bonus | -5.00 | -0.1pp |
  | angle_penalty | -2.26 | 0.0pp |
  | action_cost | -1.40 | +42.5pp |
  | crash_penalty（新增） | +0.00 | +0.0% |

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮 ep_sum_mean | 方向切换 |
|---------|----------|-----------------|---------|
| 接近/进度信号 | iter_1: progress_shaping → iter_2: progress_shaping → iter_3: progress_shaping | 1.58 → 5.58 → 1.49 | 0 次（始终为连续 Δ 势能） |
| 着陆/接触信号 | iter_1: landing_velocity_penalty（死） → iter_2: success_landing_bonus（0.1%） → iter_3: success_landing_bonus（死） + crash_penalty（死） | -0.07 → 5.00 → 0.00+0.00 | 1 次：连续惩罚（v1）→ 稀疏 bonus（v2→v3） |
| 稳定性/约束信号 | iter_1: angle_penalty → iter_2: angle_penalty → iter_3: angle_penalty | -1.14 → -0.20 → -2.46 | 0 次（始终为二次型姿控） |
| 完成/成功代理 | 无 | — | — |

**方向切换说明**：着陆/接触信号在 v1 使用连续速度惩罚（激活率仅 5.3%），v2 切换为稀疏大 bonus（激活率 0.1%），v3 新增对称稀疏 crash 惩罚但激活率为 0%，同一功能角色的信号从“连续惩罚”切换到“稀疏 bonus+稀疏惩罚”，发生 1 次方向切换。