# environment_card.md

# Env_001 环境理解卡片

## 1. 任务目标  
智能体控制一个二维飞行器/着陆器，起始于画面顶部中央附近，并受到一个随机的初始作用力。  
核心目标：**在最短时间内，以最小的引擎推力，稳定、安全地降落在中央目标平台上**。  
理想行为：接近目标 → 减速 → 保持直立姿态 → 用两条支撑腿平稳触地 → 保持静止。

## 2. 任务类型选择  
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达并停留在指定的目标平台（空间导航目标），而最小化时间和推力是效率约束，不是相互冲突的多目标优化；并且成功与否完全由“是否安全到达目标区域并静止”决定，属于典型的导航目标到达任务。

## 3. 观察空间 observation_space  
- type: Box  
- shape: (8,)  
- dtype: float32（所有字段均存储为浮点数，接触标志为 0.0 或 1.0）  

每个索引含义：
- obs[0]: x_position — 水平坐标，相对于目标平台的水平位移  
- obs[1]: y_position — 垂直坐标，相对于平台高度（目标高度为 0）  
- obs[2]: x_velocity — 水平线速度  
- obs[3]: y_velocity — 垂直线速度  
- obs[4]: body_angle — 机体倾角  
- obs[5]: angular_velocity — 角速度  
- obs[6]: left_support_contact — 左侧支撑腿是否接触目标平台（1.0 接触，0.0 未接触）  
- obs[7]: right_support_contact — 右侧支撑腿是否接触目标平台（1.0 接触，0.0 未接触）  

## 4. 动作空间 action_space  
- type: Discrete  
- n: 4  
- 动作含义：  
  - action 0: no_engine — 不点火，无任何推力  
  - action 1: left_orientation_engine — 点燃左侧定向引擎，产生角加速度（使机体逆时针旋转）  
  - action 2: main_engine — 点燃主引擎，沿机头方向产生线推力（通常在机体竖直朝上时提供向上推力）  
  - action 3: right_orientation_engine — 点燃右侧定向引擎，产生反向角加速度（使机体顺时针旋转）

## 5. step 与终止条件分析  

### 5.1 终止模式  
- **失败类终止**：  
  1. `crash_or_body_contact` — 机体本体（非支撑腿）与地面发生碰撞，典型如坠毁、侧翻触地。  
  2. `horizontal_position_outside_viewport` — 水平位置漂移出允许范围（飞出画面边界）。  

- **成功类终止**（不直接给出，需通过观测推断）：  
  可能的成功状态：`y_position ≈ 0`、`x_velocity ≈ 0`、`y_velocity ≈ 0`、`|body_angle| ≈ 0`、且 `left_support_contact == 1.0` 和 `right_support_contact == 1.0`，之后触发的终止条件一般为 `body_not_awake_or_settled`。

- **歧义终止**：  
  `body_not_awake_or_settled` — 当机体进入休眠（不再运动）或完全静止时触发。这一条件可能由 **成功着陆后稳定** 引起，也可能由 **失败后卡住/翻倒静止** 引起，无法直接区分。

- **截断**（truncation）：  
  无截断标志，所有 episode 均以 `terminated=True` 结束，无 `truncated` 信号。

### 5.2 success/failure 信号可用性  
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 在 step 返回中为 {}，无任何预定义键）  
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用  

## 6. reward 函数接口契约  
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：  
- obs：长度为 8 的当前观察（可用所有维度）  
- action：0~3 的离散动作（可用于惩罚引擎使用）  
- next_obs：长度为 8 的下一时刻观察（可用于捕捉变化量）  
- info 中明确允许的字段：**无**（当前 info 为空）  
- training_progress：**仅当题目明确允许时使用，本环境描述未提及，禁止使用**

**禁止使用**：  
- original_reward  
- official_reward  
- 任何未在允许信息中列出的 info 字段  
- 任何未在允许信息中列出的 obs 切片（本环境 obs 全部可用，但不可使用隐含的未文档化维度）  

## 7. 可用于奖励函数的信号  
以下信号可从 obs / next_obs / action 中可靠获得：  
- **位置**：x_position (obs[0]), y_position (obs[1]) 以及它们的变化量  
- **速度**：x_velocity (obs[2]), y_velocity (obs[3])  
- **朝向**：body_angle (obs[4])，通常期望接近 0  
- **角速度**：angular_velocity (obs[5])  
- **接触状态**：left_support_contact (obs[6]), right_support_contact (obs[7])  
- **动作/引擎**：是否使用主引擎（action==2）、是否使用侧向引擎（action==1 或 3）

## 8. 不确定或不可用的信号  
- **original_reward**：已被完全屏蔽，不可作为信号源  
- **显式成功/失败标志**：不存在  
- **终止原因标签**：无法从 info 或终止返回值直接获知具体原因，只能通过最终状态推测  
- **引擎推力大小**：动作是离散的，但无法直接获取作用力或冲量数值  
- **外部风力/扰动**：描述提到有随机初始力，但无实时观测，无法用于奖励  
- **机体质量、惯性等物理参数**：未提供，不可用



# expert_reward_context.md

# 专家奖励知识上下文（RAG 压缩版）

这份内容不是完整知识库原文，而是给 Reward Generator 直接使用的压缩决策摘要。
以下骨架由任务路由检索生成，不预设特定组合。具体选择由环境接口中可用信号决定。

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
- 从上述任务路由推荐的骨架中选择，优先选择所需信号在环境接口中可用的骨架。
- 如果 success/failure 显式信号不存在，不要使用 terminal_success_reward / terminal_failure_penalty。
- 效率类骨架（energy_penalty、time_penalty）和复杂门控（gated_reward）默认后续迭代再加入。
- 每个组件的设计要考虑可利用风险：agent 可能找到哪些捷径？条件信号是否容易被 exploit？
- 返回格式建议为 return float(total_reward), components；components 必须是 dict。