# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主进展：水平速度正向奖励
    horizontal_speed = obs[2]
    progress_reward = 1.0 * horizontal_speed

    # 稳定约束：躯干倾斜角 hinge 惩罚（超出阈值才惩罚）
    hull_angle = obs[0]
    angle_threshold = 0.3  # 容许小幅倾斜，靠近摔倒前开始惩罚
    hinge = max(0.0, abs(hull_angle) - angle_threshold)
    stability_penalty = -0.5 * hinge

    # 效率代价：动作二次惩罚（轻量）
    action_efficiency_penalty = -0.01 * sum(a * a for a in action)

    total_reward = progress_reward + stability_penalty + action_efficiency_penalty
    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'action_efficiency_penalty': action_efficiency_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **selected_route_id**: locomotion_continuous_control  
- **dynamics_subtype**: planar_bipedal_gait（平面双足步态，地形高度变化）  
- 核心需求：双足机器人在粗糙地形上稳定前进，避免摔倒。

## selected reward roles（按优先级）
1. **progress（主学习信号）** – 告诉 agent “向前走得快就得分”  
2. **stability（安全约束）** – 防止身体过度倾斜，提前施加惩罚  
3. **efficiency（动作代价）** – 抑制不必要的关节力矩，降低能耗（次要目标）

这三个角色覆盖了 environment_card 中提取的职责：
- 前进（obligatory）
- 防止倾倒（obligatory safety/health）
- 关节能耗（optional，但动作维度≥2 且任务含节能需求，加入极轻量惩罚）

## role‑to‑signal mapping 与所选 formula operator

| 角色 | 信号 | 算子 | 数学形式 | 原因 |
|------|------|------|----------|------|
| progress | `obs[2]` horizontal_speed | `dense_state_signal` (线性) | `w_progress * horizontal_speed` | 速度是每步可测的连续进展信号，线性形式直接且稠密，无需凸化（环境未报告速度停滞在低值） |
| stability | `obs[0]` hull_angle | `dense_state_signal` (hinge) | `-w * max(0, |hull_angle| - threshold)` | 只在超出安全范围时惩罚，保留正常小幅度倾斜的探索空间，给予连续梯度（避免全时惩罚） |
| efficiency | `action[0:3]` 四维力矩 | `quadratic_penalty` | `-w * Σ action_i²` | 抑制高频大幅动作，减少能耗，系数极小以免压制探索 |

## excluded roles 及原因
- **terminal_success_reward** – explicit_success_flag_available = false，且观测中无法可靠推断到达终点时刻（无位置信息）  
- **terminal_failure_penalty** – explicit_failure_flag_available = false；已采用 hinge 提供连续梯度，硬惩罚非必需（避免稀疏信号）  
- **preview_conditioned_reward** – LiDAR 信号需复杂预处理，v1 阶段不引入  
- **joint_condition_proxy** – 无需构造软任务完成近似，前进速度已充当进展度量  
- **soft_health_gate** – 当前选择 hinge 替代，若后续观察到“先冲后死”模式（terminated 率高且主奖励仍为正）可切换为 gate  
- **action_smoothness_penalty** – 无可用的 previous action 历史，不能计算平滑性惩罚

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- environment_card 明确标记 `explicit_success_flag_available=false`, `explicit_failure_flag_available=false`，且 info 字典为空  
- 摔倒惩罚通过 continuous hinge 实现：agent 在倾斜超标时立即获得负信号，避免完全依赖稀疏事件  
- 到达终点无法可靠识别，不引入不可靠的 terminal bonus（通过速度奖励间接覆盖）

## 哪些职责留到后续迭代
- 利用 LiDAR 进行前瞻避障（preview_conditioned_reward）  
- 动态调整权重的 curriculum（若训练初期不会前进或过早摔倒）  
- 更精细的能耗建模（ torque‑rate 惩罚）  
- 关节位置/步态对称性约束（若出现跛行或畸形步态）  
- 软健康门控（如果 hinge 不足以防止“高速→失衡→摔倒”模式）

## 训练后应观察的 failure modes
1. **静止站立（stand‑still）** – agent 可能发现不动作时 stability_penalty 为零，但 progress_reward 为零，整体奖励不高；需验证 progress 项是否足以驱使它迈步  
2. **高频抖动但低速度** – 大量无用动作消耗 energy penalty，但速度未提升；可能需要微调 efficiency 权重  
3. **倾斜前进（tilt‑run）** – hull_angle 长时间接近阈值，hinge 已生效但 agent 仍依靠倾斜获得额外速度；需检查 stability 权重是否足够压制这种行为  
4. **先加速后摔倒** – 如果 hinge 惩罚出现得太晚（只在阈值后施加），可能导致 agent 高速冲击但最终摔倒；后续可考虑改用 soft_health_gate 乘到 progress_reward 上，提前衰减主奖励  
5. **触地欺骗（contact hacking）** – 频繁蹦跳以获得瞬时速度但姿态失控； hinge 惩罚对短暂超标影响有限，若出现此模式需增加姿态角速度惩罚或加腿接触一致性约束