# 匿名环境理解卡片

## 1. 任务目标
主体是一个二维飞行器，从视口顶部中心附近以随机初始速度开始运动。核心目标：以最小的引擎推力、最短的时间，到达画面中央的目标停靠平台，并在目标位置保持稳定姿态与安全接触。智能体必须学会精确导航、减速软着陆、维持水平姿态并最终使身体稳定（settled）。混淆目标：不能将“尽量少用引擎”误解为完全不使用引擎；不能把“尽快到达”与“粗暴着陆”等价；不能只优化到达而忽略着陆稳定性。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达指定目标位置（中央平台）并稳定停靠，附属目标（燃料效率、快速性）不改变主目标定位。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- 字段详解：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|----------------|
| 0 | x_position | 本体相对目标平台的水平坐标 | true |
| 1 | y_position | 本体相对平台高度的垂直坐标（正向可能为上方） | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 本体倾角（弧度） | true |
| 5 | angular_velocity | 角速度（弧度/单位时间） | true |
| 6 | left_support_contact | 左支撑腿/触地点接触标志（1.0接触，0.0离地） | true |
| 7 | right_support_contact | 右支撑腿/触地点接触标志 | true |

所有观测均可用于奖励设计，尤其是位置、速度、角度、接触标志。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：

| 动作值 | 名称 | 含义 |
|--------|------|------|
| 0 | no_engine | 无推力，依靠当前动量滑行 |
| 1 | left_orientation_engine | 启动左姿态引擎（产生逆时针或顺时针力矩） |
| 2 | main_engine | 启动主引擎（产生垂直向上推力，可能伴随小力矩） |
| 3 | right_orientation_engine | 启动右姿态引擎（产生相反方向力矩） |

动作直接影响本体推力和姿态力矩，间接影响位置、速度、角度、角速度。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` – 当本体与平台稳定接触且速度/角速度极小，进入物理引擎的“休眠”状态。结合任务目标，若该事件发生在目标平台附近且接触成功，即为成功着陆。但该条件本身仅依赖物理状态，不检查位置是否在目标附近，因此单靠此终止可能包含非目标位置的过早休眠。

- **failure-like termination**:  
  `crash_or_body_contact` – 任何非支撑腿的身体部分接触地面或发生猛烈碰撞，通常表示硬着陆、倾覆或偏离安全姿态。  
  `horizontal_position_outside_viewport` – 水平漂移出画面边界，显然失败。

- **ambiguous termination**: 无。

- **truncation**: step 源码中未显示任何时间步截断，返回 `terminated` 且 `truncated=False`。因此没有时间上限导致的截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: `{}`（step 返回空字典）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，因为未提供任何信息。

终止条件仅能从 `terminated` 标志判断 episode 结束，无法直接读取 success/failure 标签。任何基于成功/失败的奖励必须从观测状态中自行推断（例如结合位置、接触和休眠）。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs` : 当前观测（8维），全部可用。
- `action` : 当前执行的动作（0,1,2,3）。
- `next_obs` : 过渡后的观测，全部可用。
- `info` : 当前环境返回空字典，但规范上不应依赖任何字段。
- `training_progress` : 只有当 prompt 明确声明可以使用时才引入，此处未声明，默认不可依赖。

**禁止使用**：
- `original_reward` （被显式 masked）。
- 任何未在本卡片中声明的 `info` 字段。
- 任何未阐明的 `obs` 切片含义之外的信号。

## 7. 可用于奖励函数的信号
- **位置信号**：`x_position` (obs[0])，`y_position` (obs[1]) – 用于计算到目标的距离、高度误差。
- **速度信号**：`x_velocity` (obs[2])，`y_velocity` (obs[3]) – 用于鼓励减速、控制着陆冲击。
- **姿态信号**：`body_angle` (obs[4])，`angular_velocity` (obs[5]) – 用于维持水平姿态、稳定角速度。
- **接触信号**：`left_support_contact` (obs[6])，`right_support_contact` (obs[7]) – 用于检测着陆成功、双腿接地、避免单腿/侧倾。
- **动作/引擎信号**：`action` – 用于惩罚引擎使用、鼓励节约。
- **其他组合**：依据连续两帧的 `obs` 与 `next_obs` 可计算位置、速度的变化，评估进展或风险。

## 8. 不确定或不可用的信号
- **官方奖励**：已被 masked，不可用。
- **成功/失败标签**：环境不提供，不可用。
- **身体是否 awake**：仅在终止判定中使用，无法在 step 中间获取，不可用于奖励。
- **平台绝对坐标**：相对坐标已包含在 obs 中，无需额外信息。
- **时间步/剩余步数**：未提供，不可用。
- **加速度或引擎力**：不可直接观测，只能通过速度变化间接推断。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body
  actuator_type: main_engine + two_orientation_engines
  contact_structure: two_support_legs (binary contacts)
primary_objectives:
  - reach_target_pad: 最小化终点位置偏差（x,y → 0）
  - safe_landing: 低速、双腿同时接触、水平姿态、稳定
secondary_objectives:
  - fuel_efficiency: 最小化引擎使用次数/推力（动作惩罚）
  - time_efficiency: 尽快完成（可隐含，但无显式时间步惩罚则依赖快速到达）
main_failure_risks:
  - hard_crash: 高速接触或身体碰撞导致失败
  - drifting_out: 水平漂移出界
  - premature_sleep: 未到达目标即休眠（settled）
  - orientation_instability: 角度过大或不稳定旋转
  - single_leg_landing: 一条腿接地导致倾斜或无法稳定
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: target_proximity**  
  purpose: 引导主体向目标平台移动  
  why_required: 任务核心是到达指定位置，必须存在距离奖励或密集指引  
  usable_signals: [x_position, y_position] 或组合距离  
  risks: 若距离奖励过强，可能忽略着陆安全；需配合减速项

- **role_id: soft_landing_velocity_control**  
  purpose: 在接近目标时降低速度，确保安全接触  
  why_required: 安全着陆需要低速，否则硬碰撞失败  
  usable_signals: [x_velocity, y_velocity, x_position, y_position] (可结合距离做条件放大)  
  risks: 过早减速会使学习缓慢；需在距离近时才激活

- **role_id: orientation_stability**  
  purpose: 保持本体水平，避免倾覆  
  why_required: 任务要求“保持稳定姿态”；角度过大会导致接触失败  
  usable_signals: [body_angle, angular_velocity]  
  risks: 若惩罚过大，智能体可能回避必要的小幅度姿态调整

- **role_id: dual_leg_contact**  
  purpose: 鼓励双腿同时接触平台，完成稳定站立  
  why_required: 安全接触定义隐含双腿触地；单腿悬空无法稳定休眠  
  usable_signals: [left_support_contact, right_support_contact]  
  risks: 过度奖励双腿接触可能导致智能体过早追求接触而不控制下降速度

### 10.2 条件职责 conditional_roles
- **role_id: fuel_consumption_penalty**  
  condition_to_use: 始终允许加入，但权重需平衡，以免因节约推力导致无法到达。通常在训练后期或临界奖励缩放时激活。  
  usable_signals: [action] – 对非零动作施加代价  
  risks: 过强的动作惩罚会使智能体选择“静止滑行”而不主动控制，延长到达时间，甚至无法克服初始速度。

- **role_id: time_pressure**  
  condition_to_use: 当任务描述强调“尽可能快”但未提供时间戳上限时，隐含的快速到达通常由期望 discount 或稀疏终止奖励自然体现，无需显式时间惩罚。若需要显式加速，可引入每步小负奖励，但可能干扰稳定性训练。建议早期不启用。  
  usable_signals: [step_count] 不可用，只能通过 training_progress 近似，而 training_progress 不被默认允许。  
  risks: 若强制引入，可能导致仓促着陆、增加 crash 风险。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: explicit_success_bonus**  
  reason: 成功标签不存在，缺乏可靠信号。如果试图从 obs 判断“成功”，容易引入偏差（如将任何非目标位置的休眠也视为成功）。  
  forbidden_or_missing_signals: [success_flag, info字段]  

- **role_id: explicit_failure_penalty**  
  reason: 失败标签不存在，且终止原因无法在 step 中读取。想依据 crash/out 给予惩罚，需要在终止帧之前从状态中预测，风险高，不推荐。  
  forbidden_or_missing_signals: [failure_flag, info字段]  

- **role_id: angular_momentum_minimization**  
  reason: 已有 orientation_stability 角色处理角度与角速度，额外单独惩罚角速度平方可能重复，但中等程度可以包含在姿态稳定角色中。暂不列为独立角色。  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| target_proximity | x_position, y_position | 无 | euclidean_distance, bounded_signal, quadratic_penalty | 直接计算到 (0,0) 的欧氏距离，或分轴线性/平方惩罚 |
| soft_landing_velocity_control | x_velocity, y_velocity, x_position, y_position | 无 | gated_penalty (若距离<阈值则惩罚速度大小), product_with_proximity | 只在接近目标时激活速度惩罚，避免早期抑制移动 |
| orientation_stability | body_angle, angular_velocity | 无 | absolute_penalty, quadratic_penalty | 惩罚倾角绝对值 + 角速度绝对值/平方 |
| dual_leg_contact | left_support_contact, right_support_contact | 无 | conditional_reward (两者同时为1时给奖), binary_and | 鼓励双点触地，可结合速度低时给予加成 |
| fuel_consumption_penalty | action | 无 | discrete_action_cost (非零动作惩罚) | 对 action != 0 施加固定代价；可随训练进度动态缩放 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 硬着陆/碰撞 crash | 高速垂直接触、body_contact 频繁触发终止 | 增强 soft_landing_velocity_control 的权重；在接近目标时加大速度惩罚梯度 |
| 悬停/不下降徘徊 |