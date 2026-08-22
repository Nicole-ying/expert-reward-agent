第 3 轮迭代训练已结束，得分 64.238903（目标: 200）。

用 `read_memory()` 快速确认全局状态，用 `get_component_history` 逐组件检查是否有死组件（active<2%）或全覆盖惩罚（active>95% 且负贡献）。
找到问题组件后，用 `read_past_reflection` 确认你之前是否已经改过它、结果如何。

实验记录目录：
```
rpv4_old_v3/seed_4/
  iter_01/ ... iter_02/     ← 历史迭代
    generation/response_records/agent_reflection.md   ← 每轮反思决策
    generation/behavior_analyst_output_vN.md          ← Subagent 数据报告
    training/eval_result.json                          ← 训练评估结果
  memory/reward_memory.md                       ← 历史记忆表
  best/best_reward.py                            ← 历史最佳奖励函数
```

# 本轮训练反馈 (score=64.238903)
# Training Feedback

## Final-policy outcome
score=64.238903, len=924.050000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-138.260553, 191.986938]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact | 305.309309 | 97.8% | 97.8% | 12.0% |
| pos | 3.554315 | 1.1% | 1.4% | 99.9% |
| vel | -1.495697 | -0.5% | 0.5% | 99.8% |
| angle | -0.546160 | -0.2% | 0.2% | 100.0% |
| angvel | -0.372968 | -0.1% | 0.1% | 99.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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

    ox = obs[0]
    oy = obs[1]
    nx = next_obs[0]
    ny = next_obs[1]
    nvx = next_obs[2]
    nvy = next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft = next_obs[6]
    nright = next_obs[7]

    # Distances
    prev_dist = (ox**2 + oy**2 + 1e-8) ** 0.5
    next_dist = (nx**2 + ny**2 + 1e-8) ** 0.5

    # -- 1) approach progress: reward reduction in distance to pad --
    w_pos = 3.0
    pos_reward = w_pos * (prev_dist - next_dist)   # >0 when approaching, <0 when retreating

    # -- 2) soft landing: speed & attitude penalty gated by distance --
    k_gate = 10.0
    gate = 1.0 / (1.0 + k_gate * next_dist)         # near pad -> gate ~1, far away -> gate ~0

    w_vel = 0.5
    vel_penalty = -w_vel * (nvx**2 + nvy**2) * gate

    w_angle = 0.5
    angle_penalty = -w_angle * (nangle**2) * gate

    w_angvel = 0.5
    angvel_penalty = -w_angvel * (nangvel**2) * gate

    # -- 3) safe contact: reward both legs on the pad near center --
    contact_proxy = nleft * nright                    # both legs touching -> 1.0, else 0.0
    contact_threshold = 0.3
    contact_factor = max(0.0, 1.0 - next_dist / contact_threshold)
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