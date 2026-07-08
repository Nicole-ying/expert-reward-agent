# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。智能体从一个靠近视口顶部中心的位置出发，初始受到随机力扰动。  
核心目标：**尽快且稳定地降落在画面中央的目标平台上**（降低水平与垂直速度、保持姿态平稳、安全接触）。  
同时希望发动机推力使用越少越好（能耗经济性为附属优化项）。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（连续部分为浮点数，接触标志用 0.0/1.0 表示）
- obs[0] (x_position): 水平坐标，相对于目标平台中心的偏移量（朝右为正）
- obs[1] (y_position): 垂直坐标，相对于平台高度的偏移量（朝上为正）
- obs[2] (x_velocity): 水平线速度
- obs[3] (y_velocity): 垂直线速度
- obs[4] (body_angle): 机体朝向角度（单位：弧度）
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左侧支撑接触标志，0.0 或 1.0
- obs[7] (right_support_contact): 右侧支撑接触标志，0.0 或 1.0

## 4. 动作空间 action_space
- type: Discrete(4)
- 各动作含义：
  - action 0: no_engine —— 不启用任何引擎（滑行/自由运动）
  - action 1: left_orientation_engine —— 点燃左侧姿态调整引擎（产生旋转力矩）
  - action 2: main_engine —— 点燃主引擎（主要推力，垂直于机体下方）
  - action 3: right_orientation_engine —— 点燃右侧（与左侧相反）姿态调整引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（机体不再活跃或已稳定，可能表示成功着陆并静止）
- failure-like termination: `crash_or_body_contact`（撞击或机体其他部位接触障碍）、`horizontal_position_outside_viewport`（水平出界）
- ambiguous termination: 无（所有终止条件均按上述归类，但需注意 `body_not_awake_or_settled` 未直接判断是否在目标位置，实际成功与否依赖于奖励设计）
- truncation: 未显式设置，理论上可能由时间限制截断（本环境未说明，可忽略）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回空字典 `{}`）
- forbidden_or_uncertain_info_fields: 任何 info 字段均不可用（因 info 为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观察，shape=(8,)）
- action（当前动作，0-3）
- next_obs（下一时刻观察）
- info 中明确允许的字段（本环境 info 为空，故无法可靠使用任何字段）
- training_progress（仅在 prompt 明确允许时使用，此处不涉及）

禁止使用：
- original_reward（已被遮盖的官方奖励）
- official_reward（同义，不可用）
- 未声明的 info 字段（实际上不存在）
- 未声明的 obs 切片（仅可使用上述已说明的 8 维）

## 7. 可用于奖励函数的信号
- position: `next_obs[0]`（水平偏移）、`next_obs[1]`（垂直偏移）
- velocity: `next_obs[2]`（水平速度）、`next_obs[3]`（垂直速度）
- orientation: `next_obs[4]`（机体角度）与 `next_obs[5]`（角速度）
- contact: `next_obs[6]` 和 `next_obs[7]`（左右支撑接触标志）
- action/engine: 可通过动作选择判断是否使用主引擎、姿态引擎，进而隐含能耗信号

## 8. 不确定或不可用的信号
- info 字段全部不可用（字典为空）
- 原始奖励（original_reward）已遮盖，不可用
- 确切的任务成功/失败标记（如 `success`、`failure`）不存在
- “是否真正在目标平台上”无法直接从 `body_not_awake_or_settled` 判断，需通过位置阈值的自定义逻辑推断