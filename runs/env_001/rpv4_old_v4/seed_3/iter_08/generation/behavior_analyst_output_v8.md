# 训练数据报告

## 1. 行为事实

- **得分**：score=227.64，range=[-11.91, 281.99]
- **终止方式**：terminated=19/20（95% 成功着陆），truncated=1/20（5% 时间上限）
- **回合长度**：len=374.60（成功率提升，回合提前终止）
- **Checkpoint 官方得分轨迹**：
  cp10%: -32.2 → cp20%: -150.0 → cp30%: -4.4 → cp40%: -2.7 → cp50%: -9.8 → cp60%: 246.1 → cp70%: 157.2 → cp80%: 223.1 → cp90%: 248.3 → cp100%: 236.3
  - 训练在 cp60% 处分数骤升（从 -9.8 到 246.1）
  - cp70% 有所回落（157.2），随后恢复并稳定在 220-250 区间
  - 最终评估得分 227.64，与 cp100% 的 236.3 接近

---

## 2. 骨架演进表

| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | descent_incentive_ep_sum | descent_incentive_active | goal_proximity_ep_sum | goal_proximity_active | orientation_penalty_ep_sum | orientation_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -32.35 | 1000.00 | 0T/20Tr | 0.000 | 0.0% | — | — | 3.758 | 100.0% | -3.959 | 100.0% | -0.486 | 100.0% |
| 2 | 107.13 | 775.60 | 11T/9Tr | 154.465 | 2.8% | — | — | 3.963 | 99.1% | -3.090 | 100.0% | -5.313 | 99.2% |
| 3 | -26.17 | 1000.00 | 0T/20Tr | 0.000 | 0.0% | — | — | 7.685 | 100.0% | -4.038 | 100.0% | -4.758 | 100.0% |
| 4 | 183.52 | 710.25 | 19T/1Tr | 114.083 | 4.9% | 0.779 | 72.4% | 7.951 | 98.1% | -3.386 | 100.0% | -5.736 | 98.4% |
| 5 | -15.01 | 1000.00 | 0T/20Tr | 0.000 | 0.0% | 0.624 | 66.1% | 7.881 | 100.0% | -3.615 | 100.0% | -6.826 | 100.0% |
| 6 | -15.73 | 1000.00 | 0T/20Tr | 0.000 | 0.0% | 3.366 | 73.8% | 7.861 | 100.0% | -3.917 | 100.0% | -5.725 | 100.0% |
| 7 | 227.64 | 374.60 | 19T/1Tr | 36.600 | 20.3% | 11.552 | 87.9% | 7.917 | 96.4% | -2.401 | 100.0% | -6.845 | 97.0% |

---

## 3. 数字标记

### 主导
- **contact_bonus** magnitude_share=55.8%，超过 80% 阈值？**否**（55.8% < 80%），无组件构成主导

### 死组件
- 无组件 active_rate < 2%

### 淹没
- **orientation_penalty**：active_rate=100.0%，magnitude_share=3.7%（< 3%？否，3.7% ≥ 3%，不满足淹没条件，但极低）
- **无其他组件**同时满足 active_rate > 90% 且 magnitude_share < 3%

### 全覆盖惩罚
- **orientation_penalty**：active_rate=100.0%（> 95%），signed_share=-3.7%（为负）→ **全覆盖惩罚**：该组件在几乎每一步都被激活且始终产生负奖励

### 量级对比（每步均值 = ep_sum ÷ len）

| 组件 | ep_sum_mean | len | 每步均值 |
|---|---|---|---|
| contact_bonus | 36.600 | 374.60 | **0.0977** ← 最大正向 |
| descent_incentive | 11.552 | 374.60 | 0.0308 |
| goal_proximity | 7.917 | 374.60 | 0.0211 |
| orientation_penalty | -2.401 | 374.60 | **-0.00641** ← 最小（绝对值） |
| velocity_penalty | -6.845 | 374.60 | -0.0183 |

- 每步均值最大两个正向组件：**contact_bonus (0.0977)**、**descent_incentive (0.0308)**
- 每步均值最小两个（绝对值最小）：**orientation_penalty (-0.00641)**、**velocity_penalty (-0.0183)**

### 修改变化（iter 7 vs iter 6，Δ = iter7 − iter6）

注意：iter 6 的 len 为 1000，iter 7 的 len 为 374.60，ep_sum 变化受回合长度缩短影响。

| 组件 | Δ ep_sum | Δ active_rate |
|---|---|---|
| contact_bonus | +36.600 | +20.3%（从 0% → 20.3%） |
| descent_incentive | +8.186 | +14.1%（从 73.8% → 87.9%） |
| goal_proximity | +0.056 | -3.6%（从 100% → 96.4%） |
| orientation_penalty | +1.516（绝对值减小） | 0%（维持 100%） |
| velocity_penalty | -1.120（绝对值增大） | -3.0%（从 100% → 97.0%） |

---

## 4. 语义演化（跨轮组件功能追踪）

### 语义功能分组

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---|---|---|---|
| **着陆/接触信号** | iter1: contact_bonus(0.0,死) → iter2: contact_bonus(154.5,主导) → iter3: contact_bonus(0.0,死) → iter4: contact_bonus(114.1,主导) → iter5: contact_bonus(0.0,死) → iter6: contact_bonus(0.0,死) → iter7: contact_bonus(36.6,主导) | -32.3 → 107.1 → -26.2 → 183.5 → -15.0 → -15.7 → 227.6 | 稀疏bonus↔死组件：**6次切换**（iter1死→iter2活→iter3死→iter4活→iter5死→iter6死→iter7活） |
| **接近/进度信号** | iter1-3: goal_proximity → iter4-7: goal_proximity + descent_incentive | 得分见上 | 门控线性→潜力式：**1次切换**（iter6 desc为线性门控，iter7 desc改为潜力式 shaping） |
| **稳定性/约束信号** | iter1-7: velocity_penalty + orientation_penalty | 得分见上 | 门控形式：iter1 Manhattan距离门控 → iter2-7 指数高度门控，**1次切换** |

### 方向切换详情

- **着陆/接触信号**：在"稀疏bonus"和"死组件"之间交替了 6 次。该功能在成功轮（iter2、iter4、iter7）中表现为主导奖励，在失败轮（iter1、iter3、iter5、iter6）中完全死亡。无"连续奖励"替代方案——该功能始终使用稀疏条件性 bonus 数学形态。

- **接近/进度信号**：iter4-6 使用线性门控 descent_incentive（`near_ground * descending`），iter7 切换为潜力式 shaping（`φ(y_nxt)-φ(y_cur)`）。在 iter1-3 中无 descent_incentive 组件，仅由 goal_proximity 承担接近引导。**切换次数：1 次**（线性→潜力式，发生在 iter6→iter7）。

- **稳定性/约束信号**：velocity_penalty 从 iter1 的 Manhattan 距离门控（`1/(1+dist)`）切换为 iter2 的指数高度门控（`exp(-y/h)`），此后维持不变。orientation_penalty 自 iter1 起未改变数学形态。**切换次数：1 次**（迭代速度门控形式，发生在 iter1→iter2）。