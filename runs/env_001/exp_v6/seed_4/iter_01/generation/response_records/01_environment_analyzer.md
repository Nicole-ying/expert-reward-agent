# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
该环境是一个 2D 飞行器轨迹优化任务。  
飞行器初始位置位于视野顶部中央附近，受到随机初始作用力。  
任务目标是 **尽快使飞行器到达并稳定停留在中心的目标垫片上**，同时尽量 **节省引擎推力**。  
智能体需要学会：接近目标、减小速度、保持姿态稳定、实现安全软着陆。

## 2. 任务类型选择
selected_route_id: multi_objective_task  
confidence: high  
reason: 任务明确包含多个优化目标——尽快到达目标、降低燃料消耗、保持姿态稳定、安全接触。速度与能耗之间存在权衡，符合多目标任务特征。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- 各维度含义：
    - obs[0] (x_position): 相对于目标垫中心的水平坐标
    - obs[1] (y_position): 相对于垫片高度的垂直坐标（高于垫片为正）
    - obs[2] (x_velocity): 水平线速度
    - obs[3] (y_velocity): 垂直线速度
    - obs[4] (body_angle): 机体朝向角度
    - obs[5] (angular_velocity): 角速度
    - obs[6] (left_support_contact): 左支撑腿接触标志（接触为 1.0，否则 0.0）
    - obs[7] (right_support_contact): 右支撑腿接触标志（接触为 1.0，否则 0.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- 动作含义：
    - action 0: no_engine —— 不点火，无任何推力
    - action 1: left_orientation_engine —— 点燃左侧姿态控制引擎
    - action 2: main_engine —— 点燃主引擎（向下推力）
    - action 3: right_orientation_engine —— 点燃右侧姿态控制引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  当 `body_not_awake_or_settled` 为真（即机体进入休眠或稳定状态）且飞行器位于目标垫片附近时，可视为成功降落并稳定。
- **failure-like termination**:  
  ① `crash_or_body_contact`（机体主要部分猛烈触地或发生碰撞，而非仅支撑腿软着陆）  
  ② `horizontal_position_outside_viewport`（水平越界）
- **ambiguous termination**:  
  `body_not_awake_or_settled` 若发生在远离目标的位置，则可能表示任务未能成功到达目标，此时性质模糊。
- **truncation**: 暂无基于步数或其他条件的时间截断说明，但后续可能引入。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回的 info 为空字典 {}）
- forbidden_or_uncertain_info_fields: 所有未在 step 代码中显式出现的 info 键值均不能使用，尤其不能假设存在 `info["success"]`、`info["failure"]` 或 `info["termination_reason"]`。

终止原因只能从 `terminated` 的真假以及 `next_obs` 的状态推断，无法直接获取区分成功/失败的布尔标志。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs` (整个 8 维向量)
- `action` (整型动作 ID)
- `next_obs` (整个 8 维向量)
- `info` 中明确出现的字段（当前为空，因此实际上无法使用 info）
- `training_progress` 只有在 prompt 中明确允许时才可使用（本环境未明确允许，故应避免依赖）

**禁止使用**：
- `original_reward`（已被屏蔽，不得参与任何计算）
- 任何未在 step 源中出现的 `info` 字段
- 任何未声明的 obs 切片含义（应基于上表维度解读）

## 7. 可用于奖励函数的信号
- **位置信号**：`x_position`, `y_position`（反映与目标点的水平与垂直距离）
- **速度信号**：`x_velocity`, `y_velocity`（可用于惩罚高速，鼓励软着陆）
- **姿态信号**：`body_angle`, `angular_velocity`（鼓励保持竖直姿态）
- **接触信号**：`left_support_contact`, `right_support_contact`（软着陆辅助判断）
- **动作/引擎信号**：`action`（用于惩罚主引擎和姿态引擎的使用，促进节能）

## 8. 不确定或不可用的信号
- **成功/失败标签**：info 中无任何标签，不能直接分辨成功或失败。
- **燃料消耗量**：未在观察空间中直接提供，只能通过动作频率间接推断。
- **目标垫片绝对高度/宽度**：仅在观察中给出相对位置，无目标尺寸等辅助信息。
- **全局时间/步数**：未作为观察或 info 暴露，若未明确允许 `training_progress` 则不可用。
