# 训练数据报告

## 1. 行为事实
- 得分：score=-40.04, range=[-133.28, 15.32]
- 终止方式：terminated 20/20（成功/失败），truncated 0/20（时间上限）
- 回合长度：len=154.05
- Checkpoint 轨迹：官方得分从 cp_40%: 0.00 → cp_80%: 0.00（持平）

## 2. 骨架演进表
| iter | score | len | term | crash_prevention_x_ep_sum | crash_prevention_x_active | crash_prevention_y_ep_sum | crash_prevention_y_active | goal_proximity_ep_sum | goal_proximity_active | living_penalty_ep_sum | living_penalty_active | soft_landing_penalty_ep_sum | soft_landing_penalty_active |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -40.04 | 154.05 | 20/0 | 0.000 | 0.0% | 0.000 | 0.0% | 2.139 | 100.0% | -0.308 | 100.0% | -0.053 | 1.5% |

## 3. 数字标记
- **主导组件**：goal_proximity (magnitude_share = 86.4%)
- **死组件**：
  - soft_landing_penalty (active_rate = 1.5%)
  - crash_prevention_x (active_rate = 0.0%)
  - crash_prevention_y (active_rate = 0.0%)
- **全覆盖惩罚**：living_penalty (active_rate = 100.0%, signed_share = -11.6%)
- **量级对比**（每步均值，ep_sum / len）：
  - 最大两个：
    1. goal_proximity: 0.0139
    2. living_penalty: -0.0020 (绝对值 0.0020)
  - 最小两个：
    1. crash_prevention_x: 0.0000
    2. crash_prevention_y: 0.0000
- **修改变化**：无上一轮，Δ 不适用