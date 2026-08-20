第 2 轮迭代训练已结束，得分 26.735066（目标: 200）。

用 `read_memory()` 快速确认全局状态，用 `get_component_history` 逐组件检查是否有死组件（active<2%）或全覆盖惩罚（active>95% 且负贡献）。
找到问题组件后，用 `read_past_reflection` 确认你之前是否已经改过它、结果如何。

实验记录目录：
```
rpv4_old_v3/seed_4/
  iter_01/ ... iter_01/     ← 历史迭代
    generation/response_records/agent_reflection.md   ← 每轮反思决策
    generation/behavior_analyst_output_vN.md          ← Subagent 数据报告
    training/eval_result.json                          ← 训练评估结果
  memory/reward_memory.md                       ← 历史记忆表
  best/best_reward.py                            ← 历史最佳奖励函数
```

# 本轮训练反馈 (score=26.735066)
# Training Feedback

## Final-policy outcome
score=26.735066, len=341.750000, terminated=15/20, truncated=5/20, reward_errors=0
score_range=[-78.024087, 258.305735]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact | 555.994366 | 85.4% | 85.4% | 31.9% |
| pos | -88.133305 | -13.5% | 13.5% | 100.0% |
| vel | -5.300010 | -0.8% | 0.8% | 100.0% |
| angvel | -1.237484 | -0.2% | 0.2% | 92.7% |
| angle | -0.444999 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估
| pct | score_mean | score_std |
|---|---|---|
| 40% | 0.00 | 0.00 |
| 80% | 0.00 | 0.00 |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs indices:
    # 0: x_position (relative to pad center)
    # 1: y_position (relative to pad height)
    # 2: x_velocity
    # 3: y_velocity
    # 4: body_angle
    # 5: angular_velocity
    # 6: left_support_contact (0.0 or 1.0)
    # 7: right_support_contact (0.0 or 1.0)

    nx = next_obs[0]
    ny = next_obs[1]
    nvx = next_obs[2]
    nvy = next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft = next_obs[6]
    nright = next_obs[7]

    # Squared distance to pad center
    dist_sq = nx**2 + ny**2
    dist = (dist_sq + 1e-8) ** 0.5

    # -- 1) approach progress: penalize squared distance (dense signal) --
    w_pos = 1.0
    pos_reward = -w_pos * dist_sq

    # -- 2) soft landing: speed & attitude penalty gated by distance --
    k_gate = 10.0
    gate = 1.0 / (1.0 + k_gate * dist)       # near pad -> gate ~1, far away -> gate ~0

    w_vel = 0.5
    vel_penalty = -w_vel * (nvx**2 + nvy**2) * gate

    w_angle = 0.5
    angle_penalty = -w_angle * (nangle**2) * gate

    w_angvel = 0.5
    angvel_penalty = -w_angvel * (nangvel**2) * gate

    # -- 3) safe contact: reward both legs on the pad near center --
    contact_proxy = nleft * nright               # both legs touching -> 1.0, else 0.0
    contact_threshold = 0.3
    contact_factor = max(0.0, 1.0 - dist / contact_threshold)
    w_contact = 10.0
    contact_reward = w_contact * contact_proxy * contact_factor

    total_reward = pos_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        'pos': pos_reward,
        'vel': vel_penalty,
        'angle': angle_penalty,
        'angvel': angvel_penalty,
        'contact': contact_reward
    }

    return total_reward, components
```

# Subagent 数据报告
# 训练数据报告 — iter_01

## 1. 行为事实
- 得分：score=26.74，range=[-78.02, 258.31]
- 终止方式：terminated 15/20（成功着垫/低高度终止），truncated 5/20（时间上限截断）
- 回合长度：len=341.75
- Checkpoint 轨迹：cp40%=0.00 → cp80%=0.00（全程持平在 0 分基线）
- 奖励误差：reward_errors=0（全部 20 个评估 episode 无计算异常）

## 2. 骨架演进表
| iter | score | len | term | contact_ep_sum | contact_active | pos_ep_sum | pos_active | vel_ep_sum | vel_active | angle_ep_sum | angle_active | angvel_ep_sum | angvel_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.74 | 341.75 | 15/5 | 555.99 | 31.9% | -88.13 | 100.0% | -5.30 | 100.0% | -0.44 | 100.0% | -1.24 | 92.7% |

## 3. 数字标记
- **主导**：contact 组件的 magnitude_share=85.4%，超过 80% 阈值。该组件在非零 episode 中占据绝对量级主导地位。
- **淹没**：angvel 组件 active_rate=92.7%（>90%），但 magnitude_share=0.2%（<3%），处于淹没状态——该组件几乎总是激活，但对最终 reward 的量级贡献可忽略。
- **全覆盖惩罚**：无组件同时满足 active_rate>95% 且 signed_share 为负。
- **死组件**：无组件 active_rate<2%。最低为 contact 的 31.9%，仍远高于死组件阈值。
- **量级对比**（每步均值 = ep_sum / len）：
  - 最大两个：contact=1.626，pos=0.258（绝对值）
  - 最小两个：angle=0.0013，angvel=0.0036（绝对值）
  - 注：所有组件的每步均值为 contact=+1.626, pos=-0.258, vel=-0.016, angle=-0.0013, angvel=-0.0036
- **修改变化**：本轮为首轮（iter_01），无历史数据，Δ 均为 N/A。

## 4. 语义演化（跨轮组件功能追踪）
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 接近/进度信号 | pos（连续惩罚，距离平方） | 26.74 | 0 次（首轮） |
| 着陆/接触信号 | contact（稀疏乘积 gated-by-distance bonus） | 26.74 | 0 次（首轮） |
| 稳定性/约束信号 | vel, angle, angvel（门控惩罚组，gate=1/(1+10·dist)） | 26.74 | 0 次（首轮） |

**说明**：当前仅有一轮数据（iter_01），尚未出现跨轮组件名称变更或功能设计切换。稳定性信号组内三个组件共享相同的门控机制设计（距离门控 + 平方惩罚），功能同构。

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
这是一个 **2D 飞行器轨迹优化任务**。  
主体在视口上方中央位置出发，受到初始随机力作用。  
**首要目标**：到达画面中央的着陆垫，并稳定停靠（减速、保持姿态、安全接触）。  
**次要目标**：同时尽可能 **快速到达** 且 **节省引擎燃料**。  
**不应混淆的目标**：不是单纯的无限制飞行或持续移动；也不是简单的平衡或抓取任务；核心在于“安全抵达+稳定着陆”。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: `x_position` — 水平相对坐标（相对于目标垫中心），可用于奖励： true
- obs[1]: `y_position` — 垂直相对坐标（相对于着陆垫高度），可用于奖励： true
- obs[2]: `x_velocity` — 水平线速度，可用于奖励： true
- obs[3]: `y_velocity` — 垂直线速度，可用于奖励： true
- obs[4]: `body_angle` — 机体倾斜角度，可用于奖励： true
- obs[5]: `angular_velocity` — 角速度，可用于奖励： true
- obs[6]: `left_support_contact` — 左支撑腿接触标志（1.0 表示接触），可用于奖励： true
- obs[7]: `right_support_contact` — 右支撑腿接触标志，可用于奖励： true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine`（不做任何推进/姿态调整）
- action 1: `left_orientation_engine`（启动左姿态引擎，产生逆时针旋转力矩）
- action 2: `main_engine`（启动主引擎，产生与机体朝向一致的推力）
- action 3: `right_orientation_engine`（启动右姿态引擎，产生顺时针旋转力矩）

> 说明：所有引擎动作都会消耗燃料，而燃料总量在观测中不可见。主体通过姿态调整+主引擎实现2D空间的移动。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination**：  
  `body_not_awake_or_settled` — 主体静止/稳定，若同时满足 **当前坐标已落在目标垫上**（x,y接近0且接触着陆），视为成功着陆。但环境**未显式标记**成功。
- **failure‑like termination**：  
  `crash_or_body_contact` — 任意身体部分（如支撑腿之外的部位）与地面/障碍物接触，可能表示坠落/翻倒；  
  `horizontal_position_outside_viewport` — 水平位置超出视口边界，表示飞出任务区。
- **ambiguous termination**：  
  仅靠 `body_not_awake_or_settled` 而不检查位置会无法区分“卡在垫外停止”或“悬空静止”的情况，需要结合观测判断。
- **truncation**： 无 episode 长度截断（未在 step 代码中体现）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: 空（info 始终为 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（包括任何 `success`、`failure`、`termination_reason`）

## 7. 可用于奖励函数的信号
- **position**：`obs[0]` (x), `obs[1]` (y) 及 `next_obs[0:2]`
- **velocity**：`obs[2]` (vx), `obs[3]` (vy) 及对应 next_obs
- **orientation / angular**：`obs[4]` (角度), `obs[5]` (角速度) 及对应 next_obs
- **contact**：`obs[6]` (左接触), `obs[7]` (右接触) 及对应 next_obs  
  注意：接触标志为 0.0/1.0，可组合判断单脚/双脚接触。
- **action/engine**：`action` 可取 0–3，可映射为燃料消耗（主引擎或姿态引擎使用）
- **other**：无额外环境信息