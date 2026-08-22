# 训练数据报告

## 1. 行为事实
- 得分：score=-159.49，range=[-274.81, -104.97]
- 终止方式：terminated=20/20（全部成功触地），truncated=0/20
- 回合长度：len=303.35
- Checkpoint 轨迹（官方得分）：
  cp10%: -758.2 → cp20%: -754.5 → cp30%: -391.6 → cp40%: -336.1 → cp50%: -206.0 → cp60%: -166.4 → cp70%: -179.8 → cp80%: -161.7 → cp90%: -145.8 → cp100%: -155.5
  （从 cp10% 到 cp90% 总体上升，cp100% 轻微回落）

## 2. 骨架演进表
| iter | score | len | term | progress_reward_ep_sum | progress_reward_active | orientation_penalty_ep_sum | orientation_penalty_active | contact_reward_ep_sum | contact_reward_active | vel_damping_ep_sum | vel_damping_active |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | -159.49 | 303.35 | terminated=20/20 | 0.104 | 100% | -0.0335 | 100% | 0.000 | 0% | 0.000 | 0% |

## 3. 数字标记
- **主导**：progress_reward magnitude_share=93.2% > 80%
- **死组件**：
  - contact_reward active_rate=0% (<2%), ep_sum=0.0
  - vel_damping active_rate=0% (<2%), ep_sum=0.0
- **淹没**：无（所有 active 组件 magnitude_share 均不低于 3%）
- **全覆盖惩罚**：无（orientation_penalty active_rate=100% 但 signed_share=-6.8%，并非 magnitude_share 大且 signed_share 为负；此处 signed_share 负但 magnitude_share 只有 6.8%，不满足幅度低，实际 active_rate 100% 但 magnitude_share 6.8% >3% 不构成淹没，且 signed_share 负但不构成“全覆盖惩罚”定义，我们仅标记定义内的，无该项）
- **量级对比**：
  - 每步均值：progress_reward=0.104125/303.35≈0.000343，orientation_penalty=-0.0335/303.35≈-0.000110，contact_reward=0，vel_damping=0
  - 最大两个组件：progress_reward (0.000343), orientation_penalty (0.000110)
  - 最小两个组件：contact_reward (0), vel_damping (0)
- **修改变化**：无上一轮，不列出Δ。

## 4. 语义演化（跨轮组件功能追踪）

本报告仅含一个轮次（iter_01），无可跨轮对比。按现有配置识别语义功能：

- 接近/进度信号：progress_reward（基于距离改善的稠密进度信号）
- 稳定性/约束信号：orientation_penalty（角度和角速度的二次惩罚）
- 着陆/接触信号：contact_reward（双足触地且姿态/速度接近零的连续条件奖励）
- 稳定性/约束信号（近垫速度阻尼）：vel_damping（近垫时对速度绝对值的铰链式惩罚）

因仅有一轮，无跨轮切换记录。

```
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | progress_reward | -159.49（含整体得分） | — |
| 稳定性/约束（角度） | orientation_penalty | -159.49 | — |
| 着陆/接触信号 | contact_reward | -159.49 | — |
| 稳定性/约束（近垫速度） | vel_damping | -159.49 | — |
```

方向切换全部为0，因为仅有一轮。