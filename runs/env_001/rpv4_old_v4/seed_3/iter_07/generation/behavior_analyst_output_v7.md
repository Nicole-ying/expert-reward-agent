# 训练数据报告

## 1. 行为事实

- 得分：score=-15.73，range=[-41.62, 19.63]
- 终止方式：terminated=0/20（成功落地 0 次），truncated=20/20（全部达到 1000 步时间上限）
- 回合长度：len=1000.00（全部截断）
- reward_errors：0
- Checkpoint 轨迹：cp10%=-61.3 → cp40%=-9.2 → cp80%=156.2（峰值）→ cp90%=89.4 → cp100%=-15.0（最终得分 100 步窗口内剧烈回撤）
- contact_bonus checkpoint 轨迹：cp10%-cp70% 均为 0.000，cp80%=1168.992（骤升），cp90%=136.610（骤降），cp100%=0.000（归零）
- descent_incentive 本轮 ep_sum_mean=3.366，active_rate=73.8%（未全覆盖）
- 早期终止（<150 步且 score<-50）：0/20

## 2. 骨架演进表

| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | orientation_penalty_ep_sum | orientation_penalty_active | descent_incentive_ep_sum | descent_incentive_active | contact_bonus_ep_sum | contact_bonus_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -32.35 | 1000.00 | 0/20/20 | 3.758 | 100.0% | -0.486 | 100.0% | -3.959 | 100.0% | — | — | 0.000 | 0.0% |
| 2 | 107.13 | 775.60 | 11/9/20 | 3.963 | 99.1% | -5.313 | 99.2% | -3.090 | 100.0% | — | — | 154.465 | 2.8% |
| 3 | -26.17 | 1000.00 | 0/20/20 | 7.685 | 100.0% | -4.758 | 100.0% | -4.038 | 100.0% | — | — | 0.000 | 0.0% |
| 4 | 183.52 | 710.25 | 19/1/20 | 7.951 | 98.1% | -5.736 | 98.4% | -3.386 | 100.0% | 0.779 | 72.4% | 114.083 | 4.9% |
| 5 | -15.01 | 1000.00 | 0/20/20 | 7.881 | 100.0% | -6.826 | 100.0% | -3.615 | 100.0% | 0.624 | 66.1% | 0.000 | 0.0% |
| 6 | -15.73 | 1000.00 | 0/20/20 | 7.861 | 100.0% | -5.725 | 100.0% | -3.917 | 100.0% | 3.366 | 73.8% | 0.000 | 0.0% |

## 3. 数字标记

**死组件**：
- contact_bonus：active_rate=0.0%，ep_sum_mean=0.000（连续 3 轮在非成功运行中均为 0%）

**量级对比**（ep_sum ÷ len 每步均值）：
- 最大正分量：goal_proximity=0.00786，descent_incentive=0.00337
- 最大负分量：velocity_penalty=-0.00572，orientation_penalty=-0.00392
- 最小分量：contact_bonus=0.00000

**修改变化（Δ iter_05→iter_06）**：
| component | Δ ep_sum_mean | Δ active_rate |
|---|---:|---:|
| goal_proximity | -0.021 | 0.0pp |
| velocity_penalty | +1.102 | 0.0pp |
| orientation_penalty | -0.302 | 0.0pp |
| descent_incentive | +2.742 | +7.7pp |
| contact_bonus | 0.000 | 0.0pp |

**无主导**：goal_proximity magnitude_share=39.1%，未超过 80%。
**无淹没**：所有组件 active_rate 均在 0%-100% 之间，未触发淹没条件。
**无全覆盖惩罚**：无组件同时满足 active_rate>95% 且 signed_share 为负。

## 4. 语义演化（跨轮组件功能追踪）

### 接近/进度信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近信号 | v1_goal_proximity → v2_goal_proximity → v3_goal_proximity → v4_goal_proximity → v5_goal_proximity → v6_goal_proximity | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 | 改善量：0次切换（始终为平方距离delta） |

goal_proximity 在所有 6 轮中保持"改善量 delta"形态未变。ep_sum 在迭代 3 因权重翻倍（w_x 1→2, w_y 2→4）从 ~3.8 跳升至 ~7.7，此后稳定在 7.8-8.0 区间，是每轮最稳定的组件。

### 稳定性/约束信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 速度约束 | v1_velocity_penalty → v2_velocity_penalty → v3+v4+v5+v6_velocity_penalty | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 | 门控形式：1次切换（v1曼哈顿距离门控→v2+指数高度门控） |
| 姿态约束 | v1_orientation_penalty → v2+v3+v4+v5+v6_orientation_penalty | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 | 0次切换（始终为|angle|+angvel²） |

velocity_penalty 在 v1→v2 完成从曼哈顿距离门控到指数高度门控的 1 次切换后，后续 5 轮未再变更设计。ep_sum 在成功轮次（v2=-5.31, v4=-5.74）和失败轮次（v5=-6.83, v6=-5.72）之间波动，失败轮次中惩罚更强。orientation_penalty 6 轮 0 次设计变更。

### 着陆/接触信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 着陆奖励 | v1_contact_bonus → v2_contact_bonus → v3_contact_bonus → v4_contact_bonus → v5_contact_bonus → v6_contact_bonus | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 | 稀疏bonus：权重反复：w=10→10→4→4→5→4，共3次切换 |
| 下降引导 | — → — → — → v4_descent_incentive → v5_descent_incentive → v6_descent_incentive | — → — → — → 183.5 → -15.0 → -15.7 | 门控→全高度：1次切换（v4/v5 y<0.8→v6 y<1.5） |

contact_bonus 在 6 轮中经历 3 次权重切换（10→10→4→4→5→4），仅在 iter_02 和 iter_04 两轮成功激活（active_rate 2.8% 和 4.9%），其余 4 轮均为死组件（active_rate 0%）。checkpoint 数据显示 iter_06 中 cp80% 曾达到 1168.992（骤升），但在 cp90% 暴跌至 136.610，cp100% 归零——agent 在训练中期发现着陆后最终放弃了该行为。

descent_incentive 从 v4 引入至今经历 1 次设计切换（门控高度从 0.8 扩大到 1.5，权重从 2.0 提升到 6.0）。v6 中 ep_sum 从 v5 的 0.624 增长到 3.366（+438%），active_rate 从 66.1% 提升到 73.8%，但并未转化为着陆成功（terminated 保持 0/20）。

### 完成/成功代理
无独立完成代理组件。contact_bonus 兼作稀疏成功信号。