# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度（水平速度）
    horizontal_velocity = obs[2]
    # 稳定约束：躯干倾斜角度（保持直立）
    hull_angle = obs[0]

    # 权重设定
    w_forward = 1.0
    w_upright = 0.5

    forward_reward = w_forward * horizontal_velocity
    upright_penalty = -w_upright * (hull_angle ** 2)

    total_reward = forward_reward + upright_penalty

    components = {
        "forward_velocity": forward_reward,
        "upright_penalty": upright_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 任务画像
- **task_family / dynamics_subtype**: locomotion_continuous_control / planar_bipedal_gait
- **目标**: 在起伏地形上尽可能远、快地向前行走，并节约能量；摔倒或到达终点均结束回合。

## 奖励职责分解
根据 environment_card，reward_v1 需覆盖以下职责：
1. **主学习信号**：持续前进，即最大化水平方向的速度（前进距离的代理）。
2. **健康/安全约束**：防止躯体过度倾斜导致摔倒，用躯干角度进行轻量惩罚。

由于 info 为空且无显式成功/失败标志，无法直接加入 terminal success/failure 奖励。附属职责（节能、步态协调、地形感知）统一留到后续迭代。

## 职责-信号映射
- **前进**: `horizontal_velocity` (obs[2])，每步可测的连续正值，直接驱动向前移动。
- **直立约束**: `hull_angle` (obs[0])，躯干相对直立方向的角度，偏离零值代表失稳风险，用二次惩罚轻约束。

## 公式算子选择
- **主学习信号**: `dense_state_signal` (positive) – `w * signal`，直接取水平速度，提供稠密梯度。
- **稳定约束**: `quadratic_penalty` – `-w * error**2`，对躯干倾斜角进行二次惩罚，在角度较小时惩罚温和，在角度较大时迅速增大，避免倾斜摔倒。

## 排除的职责及原因
- `terminal_success_reward` / `terminal_failure_penalty`：info 为空，无 success/failure 标志，无法使用。
- `energy_efficiency`：动作扭矩平方和（能耗）留在后续迭代，v1 聚焦于学会稳定前进。
- `gait_coordination`：触地/离地步态相位未在当前信号中直接提供，且 v1 无需引入。
- `terrain_awareness`：lidar 距离可辅助地形适应，但主信号已由水平速度覆盖，暂时不做地形引导。
- `curriculum_weighting`：训练进度未启用，且 v1 尚无阶段性冲突需求。

## 后续迭代预留
- 能耗约束（`-w * sum(action**2)`）
- 基于腿接触的步态切换奖励或交替触地鼓励
- 地形感知组件（利用 lidar 进行安全高度/越障辅助）
- 动态权重/课程调度（当速度与稳定性出现明显折中时调整）

## 训练后需关注的 failure modes
- **小步抖动前进**：仅通过小幅高频动作维持极小速度，直立惩罚很低，但整体前进缓慢。
- **冲刺后摔倒**：短时间获得很高水平速度奖励，但躯干倾斜角迅速增大导致回合终止，总回报不高。
- **完全静止**：为避免任何倾斜惩罚，智能体可能选择完全不动（水平速度 0），但没有任何前进奖励。
- **角速度振荡**：虽未直接惩罚，但躯干角度惩罚会间接抑制大幅角速度；未来可加入轻量 angular_velocity 惩罚消除振荡。

当前 v1 设计用两个组件平衡前进与稳定，便于初期学习方向建立，后续可逐步引入效率和步态精细化。