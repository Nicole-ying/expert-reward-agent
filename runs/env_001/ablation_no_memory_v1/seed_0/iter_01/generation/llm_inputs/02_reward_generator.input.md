# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从视口顶部中心区域出发（初始带有一定随机速度），尽快抵达中央的着陆垫并稳定着陆。  
要求：以尽可能快的速度接近目标，减少移动速度，保持稳定姿态，最后安全地接触到着陆垫。  
此外，在整个过程中尽量少使用主引擎推力，实现节能。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box（连续值）
- shape: (8,)
- dtype: float32
- 每一维含义：
  - obs[0]: x_position — 飞行器相对于着陆垫中心的水平坐标（负左正右）
  - obs[1]: y_position — 飞行器相对于着陆垫表面高度的垂直坐标（正值在上方）
  - obs[2]: x_velocity — 水平线速度
  - obs[3]: y_velocity — 竖直线速度
  - obs[4]: body_angle — 机体倾斜角度
  - obs[5]: angular_velocity — 机体角速度
  - obs[6]: left_support_contact — 左支撑/接触标志（1.0 接触，0.0 不接触）
  - obs[7]: right_support_contact — 右支撑/接触标志（1.0 接触，0.0 不接触）

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 4
- 动作含义：
  - action 0: no_engine — 不启动任何引擎，只受重力/惯性影响
  - action 1: left_orientation_engine — 启动一个姿态引擎，使机体逆时针（或某一方向）旋转
  - action 2: main_engine — 启动主引擎，产生与机体朝向相关的推力（通常向上）
  - action 3: right_orientation_engine — 启动另一个姿态引擎，使机体相反方向旋转

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  没有显式的成功标记。但在任务描述中，“到达着陆垫并稳定接触”是目标。  
  当机体在垫上方、两侧接触标志均为1、且速度/角速度极小、位置稳定（即 `body_not_awake_or_settled`）时，环境会因 `body_not_awake_or_settled` 而终止。这种情况很可能对应成功着陆（需结合实际行为判断）。
- failure-like termination:
  - `crash_or_body_contact`：可能指机体与除着陆垫以外的地面或墙壁发生碰撞（如侧翻、头部碰撞等），这应属于失败。
  - `horizontal_position_outside_viewport`：飞行器水平位置超出视野边界，显然失败。
- ambiguous termination:  
  - `body_not_awake_or_settled` 这个终止条件本身不区分是成功着陆还是单纯因机体“睡着”（如动作太弱、陷入死区）。如果出现在没有着陆接触、远离目标点时，可能不代表成功；若出现在垫上且接触时，代表成功。因此单独看终止信号不能直接作为奖励依据。
- truncation: 未提供（本次环境没有设定最大步数截断，或已融入终止条件中）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: `{}` （step 返回的 info 为空字典，可用字段无）
- forbidden_or_uncertain_info_fields: 任何自定义字段（如 `success`、`landed`、`crash` 等）均不可用，因为环境未提供

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`：当前步的 8 维观察（每维含义见上方）
- `action`：当前步采取的动作编号（0~3）
- `next_obs`：下一步的 8 维观察
- `info`：空字典，无可依赖字段
- `training_progress`：若 prompt 明确允许，否则禁止使用

禁止使用：
- `original_reward`（官方奖励被屏蔽，不可访问、不可重建）
- 任何未在 `observation_space` 中声明的变量
- 任何未在 `info` 中出现的字段
- 对 obs 中未明确声明的维度切片

## 7. 可用于奖励函数的信号
- **位置（相对目标）**：`next_obs[0]` (x_position) 和 `next_obs[1]` (y_position) 可用于评判距着陆垫中心的距离
- **速度**：`next_obs[2]` (x_velocity) 和 `next_obs[3]` (y_velocity) 可用于评判机体在着陆时的平稳性
- **姿态**：`next_obs[4]` (body_angle) 和 `next_obs[5]` (angular_velocity) 可用于评判姿态是否保持水平/稳定
- **接触**：`next_obs[6]` (left_support_contact) 和 `next_obs[7]` (right_support_contact) 可用于检测是否着落到垫上（双接触通常说明成功着陆）
- **动作/引擎**：`action` 可用于衡量引擎使用情况，特别是 action 2（主引擎）的使用频率

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：info 中无相关字段，只能通过终止条件组合和接触、位置状态来间接推断
- **绝对世界坐标**：只提供了相对于目标的坐标，无法知道全局顶点位置（但不影响任务）
- **真实环境奖励**：被屏蔽，无法使用
- **着陆垫的绝对坐标**：未给出，仅能从 x, y 相对值推理
- **剩余步数/时间信号**：没有提供，不能作为 progress 相关信号（除非 prompt 允许 training_progress）



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架是任务相关的设计原语、风险提示和参考起点，不构成封闭候选集合。可直接采用、组合、变形，或基于环境事实提出新的数学结构。

## 1. 任务路由摘要
- navigation_goal_reaching：任务目标是接近/到达指定位置。重点观察 goal_near_oscillation / high_reward_without_success / fast_crash_near_goal。

## 2. 相关奖励骨架摘要（按任务路由检索）

以下骨架由任务路由检索推荐。是否使用某个骨架取决于：
1. 该骨架所需信号是否在环境接口中实际可用；
2. 是否与该任务阶段匹配（v1 优先设计核心学习信号，效率/安全类后续迭代加入）。

### progress_delta_reward
- 角色: 密集学习引导
- 数学形态: d(obs, goal) - d(next_obs, goal)
- 需要信号: obs[0], obs[1], next_obs[0], next_obs[1]
- 使用说明: 奖励每一步更接近目标。适合目标位置已知的导航/到达任务。
- 风险: 目标附近震荡。
- 后续迭代: 可 clip；后续配合成功、时间或稳定信号。

### distance_reward
- 角色: 密集过程引导
- 数学形态: -d(obs, goal)
- 需要信号: obs[0], obs[1]
- 使用说明: 连续负距离信号，引导 agent 靠近目标。与 progress_delta_reward 同时大权重使用会产生重复信号。
- 风险: 接近目标但不完成；不关心速度和姿态。
- 后续迭代: 训练后检查 high_reward_without_success。

### potential_based_shaping
- 角色: 势能塑形
- 数学形态: gamma*Phi(next_obs)-Phi(obs)
- 需要信号: 可定义 Phi
- 使用说明: 基于势能差分的塑形信号。比 progress_delta 更抽象，当任务有明确的势能定义时可使用。
- 风险: Phi 错误会误导学习。
- 后续迭代: 如果需要更标准的 shaping，再替换或补充。

### stability_penalty
- 角色: 轻量稳定约束
- 数学形态: -lambda_v*|velocity| - lambda_a*|angle| - lambda_w*|angular_velocity|
- 需要信号: next_obs[2], next_obs[3], next_obs[4], next_obs[5]
- 使用说明: 抑制高速、大角度或高角速度。适合需要稳定运动或姿态控制的任务。
- 风险: 过强会保守或不敢动。
- 后续迭代: 若高速失稳，增大权重。

### soft_landing_proxy
- 角色: 任务完成近似信号
- 数学形态: small_bonus if near_target and low_speed and stable_angle and both_contact else 0
- 需要信号: position, velocity, angle, contact flags
- 使用说明: 多条件组合的完成近似。不能直接把 contact 当 success。
- 风险: 如果条件太宽，会变成 contact reward hacking。
- 后续迭代: 如果 high_reward_without_success，收紧条件或移除。

### terminal_success_reward
- 角色: 任务目标奖励
- 数学形态: R_success * I[success]
- 需要信号: 显式 success flag
- 使用说明: 当环境提供显式 success flag 时可用。若 explicit_success_flag_available=false，不可使用。
- 风险: 会诱导 LLM 发明 info['success']。
- 后续迭代: 当 wrapper 明确暴露 success 后再加。

### terminal_failure_penalty
- 角色: 失败惩罚
- 数学形态: -R_failure * I[failure]
- 需要信号: 显式 failure flag 或 termination_reason
- 使用说明: 当环境提供显式 failure flag 时可用。若 explicit_failure_flag_available=false，不可使用。
- 风险: 误判终止原因。
- 后续迭代: 当能区分失败终止后再加。

### time_penalty
- 角色: 效率约束
- 数学形态: -lambda_time
- 需要信号: 每步调用
- 使用说明: 惩罚每步耗时。先完成任务再加入，不建议 v1 使用。
- 风险: 可能导致冒险或快速失败。
- 后续迭代: 若能接近但拖太久，再小权重加入。

### energy_penalty
- 角色: 动作/能耗约束
- 数学形态: -lambda_action * engine_use(action)
- 需要信号: action
- 使用说明: 惩罚动作幅度/能耗。先完成任务再加入，v1 太早加入可能不敢动。
- 风险: agent_afraid_to_move。
- 后续迭代: 能完成任务并稳定后再优化能耗。

### gated_reward
- 角色: 安全门控
- 数学形态: if unsafe then penalty else task_reward
- 需要信号: 明确 unsafe 条件
- 使用说明: 按条件切换奖励模式。v1 不建议使用复杂门控。
- 风险: 门控过严导致学不到。
- 后续迭代: 若安全被进度奖励抵消，再加入。

## 3. reward_v1 生成要求
- 直接生成 reward_v1.py，不再生成 reward_design_plan.json。
- 使用 role-based component budget：每个组件必须有明确角色，不能为了显得完整而堆叠。
- 将上述骨架作为思考面而非允许列表；最终设计由任务目标、可用信号和预期策略行为决定，不要求组件对应已有 skeleton_id。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。