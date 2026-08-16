# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主要目标：控制一个带有主引擎和两个姿态引擎的 2D 车体，从视口顶部中心尽快降落到中央目标垫上并稳定停靠。稳定停靠需要满足位置接近垫中心、速度接近零、姿态保持水平且两个支撑腿安全接触垫面。

次要目标：在确保成功着陆的前提下，尽可能减少引擎使用（节约燃料），同时尽量缩短飞行时间。

不应混淆的目标：该任务不是单纯的速度追踪或跟踪轨迹，也不是平衡存活问题；引擎节油和时间最优是辅助，核心是安全到达目标区域。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 任务描述明确以“到达并稳定在中央目标垫”为最终判定依据，动作和状态设计都服务于向目标区域接近并控制姿态；辅助目标（节油、快速）不改变核心目的，不存在同样权重的多目标冲突。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (推断)
- **obs[0]**: x_position – 相对于目标垫中心的水平坐标，reward_usable: true
- **obs[1]**: y_position – 相对于垫着陆面的垂直坐标（高度），reward_usable: true
- **obs[2]**: x_velocity – 水平线速度，reward_usable: true
- **obs[3]**: y_velocity – 垂直线速度，reward_usable: true
- **obs[4]**: body_angle – 车体倾角，reward_usable: true
- **obs[5]**: angular_velocity – 角速度，reward_usable: true
- **obs[6]**: left_support_contact – 左支撑腿接触标志（连续 1.0 或 0.0），reward_usable: true
- **obs[7]**: right_support_contact – 右支撑腿接触标志，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- **action 0**: no_engine – 不点火，自由运动  
- **action 1**: left_orientation_engine – 左姿态引擎点火（产生逆时针力矩）  
- **action 2**: main_engine – 主引擎点火（产生向上推力，同时可能改变姿态）  
- **action 3**: right_orientation_engine – 右姿态引擎点火（产生顺时针力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination**: `body_not_awake_or_settled` 中的 “settled” 情况（身体静止且可能已着陆在垫上），通常意味着成功。但不能仅靠该标记，因为也可能包含觉醒失败等。
- **failure‑like termination**: `crash_or_body_contact`（与危险物体碰撞或身体部分接触非目标表面）、`horizontal_position_outside_viewport`（飞出水平边界）
- **ambiguous termination**: `body_not_awake_or_settled` 在没有额外信息时可能是成功也可能是长时间静止在空中，需要结合位置和接触信号判断。
- **truncation**: 未提及（环境中无）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields:（info 返回 `{}`，没有任何可用字段）
- forbidden_or_uncertain_info_fields: 没有 info 字段，不能依赖任何隐藏终止原因；成功/失败需通过状态和终止组合推断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs` – 当前观测
- `action` – 当前动作
- `next_obs` – 下一时刻观测
- `info` – 明确为空字典，可忽略
- `training_progress` – 仅在 prompt 明确允许时使用（此处未允许，故不使用）

禁止使用：
- `original_reward` – 官方奖励已被隐去，严禁直接或间接利用
- 未声明的 info 字段
- 未声明的 obs 切片（只能使用下标 0‑7）

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]` (x), `next_obs[1]` (y) – 接近目标水平的水平和垂直距离
- **velocity**: `next_obs[2]` (vx), `next_obs[3]` (vy) – 抑制高速，鼓励着陆时低速
- **orientation**: `next_obs[4]` (angle), `next_obs[5]` (angular velocity) – 姿态稳定
- **contact**: `next_obs[6]`, `next_obs[7]` – 接触状态，可用于奖励正确着垫
- **action/engine**: `action` – 可用于惩罚不必要的引擎使用，且动作类型影响姿态和推力
- **other**: 可结合上述信号构建复合指标（如距离惩罚、摇晃惩罚）

## 8. 不确定或不可用的信号
- 无可用 info 字段，不能获取“成功标志”、“碰撞细节”、“是否在垫上”等直接标签。
- 无直接时间或剩余燃料信息。
- 终止条件的具体触发逻辑不透明；仅能从环境组合判断轨迹结果。
- 没有任何关于目标区域的大小或期望着陆精度的硬编码数值。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_with_attitude_control
  actuator_type: thrusters (main_engine + two orientation_engines)
  contact_structure: two_support_legs (left and right)
primary_objectives:
  - arrive close to the target pad center (x≈0, y≈0)
  - achieve near-zero linear and angular velocity
  - maintain upright orientation (angle≈0)
  - ensure safe contact with both legs on the pad
secondary_objectives:
  - minimize fuel consumption (thrust usage)
  - minimize approach time (though no direct timer, can be encouraged via step‑based cost)
main_failure_risks:
  - crashing into ground or other obstacles
  - flying outside horizontal viewport
  - hanging in the air without settling
  - landing hard with high speed causing immediate termination
  - landing only partially, with one leg off the pad or tilted
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: goal_proximity**  
  purpose: 奖励向目标垫中心 (0,0) 靠近，惩罚偏离。  
  why_required: 这是导航核心，不提供该信号代理无法知道去往何处。  
  usable_signals: [next_obs[0], next_obs[1]]  
  risks: 可能使代理过早放弃高度管理，需与其他角色结合。

- **role_id: soft_landing_contact**  
  purpose: 奖励低速、姿态平稳、双支撑接触着陆面。  
  why_required: 任务要求“safe contact”且终止条件可能与碰撞相关，需引导软着陆。  
  usable_signals: [next_obs[2], next_obs[3], next_obs[6], next_obs[7], next_obs[4]]  
  risks: 若仅依赖接触标志可能鼓励任意撞击，必须与速度和姿态条件耦合。

- **role_id: attitude_stabilization**  
  purpose: 保持车体水平，抑制角速度和大幅倾角。  
  why_required: 姿态失控容易造成碰撞或单腿着陆，且目标要求“stable orientation”。  
  usable_signals: [next_obs[4], next_obs[5]]  
  risks: 过于严厉的惩罚可能导致害怕转动而不用姿态引擎，需适度。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_efficiency**  
  condition_to_use: 全程启用，但接近阶段可适当降低权重，或只在接触前施加。  
  usable_signals: [action]  
  risks: 强度过大可能使代理完全不使用任何引擎而悬停在空中不动，导致无法着陆。

- **role_id: time_efficiency**  
  condition_to_use: 可通过对每一步施加固定小惩罚来实现“尽快”。但若任务无时间限制显式信息，此角色为可选。  
  usable_signals: []（依赖全局步骤，可在每步给予常数负奖励）  
  risks: 时间惩罚可能导致代理急于下降而忽略姿态或接触质量，需较低强度。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: survival_balance**  
  reason: 该任务不是平衡存活问题，没有持续倒立摆或防止倒塌的目标，姿态稳定已包含在 attitude_stabilization 中。  
  forbidden_or_missing_signals: 无需独立生存角色。

- **role_id: sparse_exploration**  
  reason: 虽然初始可能需要探索，但目标区域已知且状态连续，不需要专门的探索奖励。  
  forbidden_or_missing_signals: 没有提供好奇度或计数信号。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | next_obs[0], next_obs[1] | – | dense_state_signal, bounded_signal (e.g. tanh distance), quadratic_penalty (distance^2) | 可组合 x 和 y 距离加权 |
| soft_landing_contact | next_obs[2], next_obs[3], next_obs[6], next_obs[7], next_obs[4] | exact contact surface info, leg force | conditional_signal (if both legs contact and |v|<ε and |angle|<ε then bonus) | 必须同时检查低速度和小倾角，防止假接触 |
| attitude_stabilization | next_obs[4], next_obs[5] | – | quadratic_penalty (angle^2, angular_vel^2) | 可分别对倾角和角速度进行惩罚 |
| fuel_efficiency | action | thrust magnitude | action_count_penalty (‑1 for each engine fire except no-op) | 动作 0 不点火，1,2,3 均点火，可给予负奖励 |
| time_efficiency | – | clock or step counter | constant_step_penalty | 每步固定小负值，强度需弱于主要奖励 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停在空中不下降 | 高度 y 长期较大，速度接近 0，引擎很少点火 | 加强 goal_proximity 中垂直维度奖励，或适度增大时间惩罚，或引入高度下降激励 |
| 高速撞击垫面导致立即终止 | 终止时 |v_y| 很大，且两腿 contact 为 1 | 强化软着陆条件：仅在低速 + 水平姿态下给予正向接触奖励，高速时给予惩罚 |
| 单腿着陆或侧翻 | 接触时只有一个 contact 为 1，或终止时倾角大 | 在接触奖励中加入必须双腿同时接触且保持水平角度的条件，提升 attitude 惩罚 |
| 超出水平边界 | 水平位置 |x| 超过视口极限 | 增加对 x 偏差的剧烈惩罚，或提前截断时给予巨大负奖励 |
| 引擎滥用并导致姿态剧烈震荡 | 动作频繁切换 1→3，角速度大幅波动 | 加强姿态稳定惩罚，并适当增大燃料惩罚权重，或加入角速度变化率惩罚 |
| 过早减速导致缓慢下降，浪费时间 | y 位置减小极慢，速度接近很小的常数 | 动态调整时间惩罚或引入基于高度的指标（如期望下降速度） |
