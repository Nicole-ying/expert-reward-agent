# 训练数据报告

## 1. 行为事实
- 得分：score=64.238903，range=[-138.260553, 191.986938]
- 终止方式：terminated 10/20（成功），truncated 10/20（时间上限）
- 回合长度：len=924.050000
- Checkpoint 轨迹：官方得分 cp40%=0.00 → cp80%=0.00（持平）
- 早期终止：0/20 episodes（<150 steps 且 score<-50）

## 2. 骨架演进表

| iter | score | len | term | angle_ep_sum | angle_active | angvel_ep_sum | angvel_active | contact_ep_sum | contact_active | pos_ep_sum | pos_active | vel_ep_sum | vel_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.735 | 341.750 | 15/5 | -0.445 | 100.0% | -1.237 | 92.7% | 555.994 | 31.9% | -88.133 | 100.0% | -5.300 | 100.0% |
| 2 | 64.239 | 924.050 | 10/10 | -0.546 | 100.0% | -0.373 | 99.2% | 305.309 | 12.0% | 3.554 | 99.9% | -1.496 | 99.8% |

## 3. 数字标记

**主导**
- contact 的 magnitude_share=97.8%（>80%），主导总得分

**死组件**
- 无（所有组件 active_rate ≥ 12.0%）

**淹没**
- pos: active_rate=99.9%（>90%），magnitude_share=1.4%（<3%）
- angle: active_rate=100.0%（>90%），magnitude_share=0.2%（<3%）
- angvel: active_rate=99.2%（>90%），magnitude_share=0.1%（<3%）
- vel: active_rate=99.8%（>90%），magnitude_share=0.5%（<3%）

**全覆盖惩罚**
- 无（所有 active_rate>95% 的组件 signed_share 均为负值，不符合全覆盖惩罚定义）

**量级对比（每步均值：ep_sum / len）**
- 最大两个：contact=0.330，pos=0.004
- 最小两个：angvel=-0.0004，angle=-0.0006

**修改变化（Δ iter_01→iter_02）**
- contact: ep_sum -250.685，active_rate -19.9%
- pos: ep_sum +91.687，active_rate -0.1%
- vel: ep_sum +3.805，active_rate -0.2%
- angle: ep_sum -0.101，active_rate 0.0%
- angvel: ep_sum +0.864，active_rate +6.5%

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | pos (iter_01) → pos (iter_02) | 26.735 → 64.239 | 连续罚分(dist²罚)→delta改善奖励：1次切换 |
| 着陆/接触信号 | contact (iter_01) → contact (iter_02) | 26.735 → 64.239 | 稀疏bonus↔连续奖励：无切换（保持both_legs接触形态） |
| 稳定性/约束信号 | vel+angle+angvel (iter_01) → vel+angle+angvel (iter_02) | 26.735 → 64.239 | 门控→门控：无切换（保持dist-gated连续惩罚形态） |

**方向切换说明**
- 接近信号：iter_01 使用 state-value 连续罚分（-w × dist²），iter_02 切换为 delta 改善型奖励（+w × (prev_dist - next_dist)），计 1 次切换。语义角色仍为接近引导，但数学形态从"状态值"变为"改善量"。
- 着陆信号：两轮保持相同数学形态（both_legs 接触 × 距离折减），ep_sum 从 556 下降至 305，但未发生方向切换。
- 稳定性信号：两轮保持门控连续惩罚形态（squared × gate），速度、角度、角速度三位一体，未发生方向切换。