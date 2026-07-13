# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 匿名环境观测假设（经典2D着陆器）：
    # obs[0]: x (水平偏移)    obs[1]: y (高度)
    # obs[2]: vx             obs[3]: vy
    # obs[4]: angle          obs[5]: ang_vel
    # obs[6]: left_contact   obs[7]: right_contact
    # 数值均为连续量，接触信号为0/1（或接近0/1的浮点数）

    # 1. 主学习信号：距离改善 × 姿态门控
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next          # 正值表示靠近目标

    angle_abs = abs(next_obs[4])                     # 机体角度（弧度）
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)        # soft health gate
    shaped_progress = progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚
    ang_vel         = next_obs[5]
    ang_vel_penalty = -0.1 * (ang_vel ** 2)

    # 3. 任务完成近似：双腿着陆奖励
    left  = next_obs[6]
    right = next_obs[7]
    both_contact = left * right                      # 仅当两腿同时接触时非零
    landing_bonus = 10.0 * both_contact

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 选定任务类型与动力学子类
- **task_family**：`navigation_goal_reaching`（源自 `environment_card.md`）
- **dynamics_subtype**：2D 连续控制着陆器，具备水平/垂直位移、姿态角度、角速度和着陆腿接触检测

## 选定的奖励职责
| 职责角色 | 组件名称 | 对应公式算子 | 使用的信号（obs索引） |
| --- | --- | --- | --- |
| **主学习信号** | `shaped_progress` | `improvement_delta` + `soft_health_gate` | `obs[0,1]`（距离变化）、`next_obs[4]`（姿态角度） |
| **稳定/安全约束** | `angular_vel_penalty` | `quadratic_penalty` | `next_obs[5]`（角速度） |
| **任务完成近似** | `landing_bonus` | `joint_condition_proxy`（二值乘积） | `next_obs[6,7]`（双腿接触标志） |

## 职责‑信号映射与算子选择理由
- **主信号**直接奖励“到目标垫距离的减少”，同时通过**姿态软门控**确保这种接近在机体垂直时最为有效（`soft_health_gate`）。  
- 门控使用 `gate = 1/(1 + k*|angle|)`，在角度接近 0 时几乎为 1，角度较大时强烈抑制，避免 agent 倾斜冲刺刷分。  
- **角速度惩罚**防止剧烈旋转，采用轻量二次惩罚，权重 0.1，不会导致 agent 不敢动。  
- **着陆奖励**仅在双腿同时接触时触发，乘积形式简洁，无需条件分支，且不依赖环境提供的显式成功/失败标志。

## 未使用的职责及原因
- `terminal_success_reward`：环境卡片未声明 `explicit_success_flag_available`，我们保守假设为 `false`，因此不依赖 `info` 中的成功标志。  
- `terminal_failure_penalty`：同理，未提供显式失败标志，不使用。  
- `动作/能耗代价`：v1 阶段暂不引入，避免压制探索。待 agent 掌握稳定着陆路径后再考虑燃料优化。  
- `垂直速度硬约束`：若直接惩罚 `vy²`，会同时惩罚必要的下降速度，导致 agent 悬停。留到后续迭代结合高度门控设计。

## 观测空间说明
当前 `environment_card.md` 的 `observation_space` 字段为空。本函数基于 2D 着陆任务的通常观测结构假设 8 维连续/混合观测：`[x, y, vx, vy, angle, ang_vel, left_contact, right_contact]`。若真实环境的顺序或维度与此不符，需要调整索引映射。

## 后续迭代可加入的职责
- 精细垂直速度门控（近地时限制下降速率）
- 能耗/推力惩罚
- 动态课程权重（早期重探索，后期重精确着陆）
- 更严格的多条件着陆完成判定（低速度 + 小角度 + 双腿接触）

## 训练后应重点观察的失效模式
1. **高速倾斜着陆**：姿态 gate 未充分遏制大角度冲刺，着陆时速度高但两条腿勉强接触而获得 bonus。  
2. **悬停不前**：如果 progress 项不够强或 gate 过严，agent 可能选择在空中小幅晃动以维持距离不变同时避免触地。  
3. **接触刷分**：agent 可能学会一条腿轻触平台同时另一条腿反复离触，利用乘积产生间歇奖励。后续可能需要加入持续接触或稳定条件。  
4. **着陆后不稳定**：一旦获得 landing bonus，agent 可能继续移动导致翻倒，因为后续步无惩罚。未来可考虑添加成功后的小幅存活奖励或终止 episode。
