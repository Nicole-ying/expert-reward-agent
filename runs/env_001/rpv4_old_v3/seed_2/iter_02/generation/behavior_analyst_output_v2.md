# 训练数据报告

## 1. 行为事实
- 得分：score=-155.975043，range=[-180.244147, -114.227696]
- 终止方式：terminated 20/20（环境判定终止），truncated 0/20（无超时）——注：early_terminal 10/20（步数<150且score<-50），说明一半episode提前结束
- 回合长度：len=154.400000
- 奖励计算错误：reward_errors=0
- Checkpoint 轨迹：
  - cp_40%: score_mean=0.00, score_std=0.00
  - cp_80%: score_mean=0.00, score_std=0.00
  - 轨迹形态：持平（0.00→0.00），cp1~cp3、cp5~cp7、cp9~cp10 缺失

## 2. 骨架演进表
| iter | score | len | term | progress_shaping_ep_sum | progress_shaping_active | action_cost_ep_sum | action_cost_active | angle_penalty_ep_sum | angle_penalty_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -155.98 | 154.40 | T=20/20, TR=0/20 | 1.582203 | 100.0% | -1.804000 | 58.4% | -1.140523 | 100.0% | -0.066830 | 5.3% |

## 3. 数字标记

### 主导
- **无**：最大 magnitude_share 为 progress_shaping=59.8%，未超过 80% 阈值

### 死组件
- **无**：最低 active_rate 为 landing_velocity_penalty=5.3%，高于 2% 阈值

### 淹没
- **无**：无组件同时满足 active_rate > 90% 且 magnitude_share < 3%

### 全覆盖惩罚
- **无**：landing_velocity_penalty 的 signed_share 为负（-0.9%），但 active_rate=5.3%，未超过 95%

### 量级对比（每步均值 = ep_sum ÷ len）
| 组件 | 每步均值 |
|---|---|
| progress_shaping | +0.010248 |
| action_cost | -0.011685 |
| angle_penalty | -0.007387 |
| landing_velocity_penalty | -0.000433 |

- 量级最大两个：progress_shaping（+0.010248）、action_cost（-0.011685）
- 量级最小两个：angle_penalty（-0.007387）、landing_velocity_penalty（-0.000433）

### 修改变化（Δ）
- 无历史轮次，无法计算 Δ

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | progress_shaping | -155.98 | N/A（仅一轮） |
| 着陆/接触信号 | landing_velocity_penalty | -155.98 | N/A（仅一轮） |
| 稳定性/约束信号 | angle_penalty, action_cost | -155.98 | N/A（仅一轮） |

**说明**：
- 接近/进度信号（progress_shaping）：基于位置势能的 shaping 奖励，连续奖励形态，无历史切换。
- 着陆/接触信号（landing_velocity_penalty）：基于垂直速度的 hinge 惩罚，proximity_gate 门控形态（y<0.5 激活），无历史切换。
- 稳定性/约束信号（angle_penalty + action_cost）：angle_penalty 为二次型姿态惩罚（始终激活），action_cost 为离散动作代价（按动作触发），均无历史切换。

**方向切换次数汇总**：0 次（仅一轮迭代，无可对比历史）