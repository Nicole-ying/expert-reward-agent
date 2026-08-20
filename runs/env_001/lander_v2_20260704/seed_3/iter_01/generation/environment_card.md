# Env_001 环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从起点（视口顶部中心附近，有随机初始受力）移动并平稳降落到画面中央的目标平台。  
要求尽可能**快**地到达目标，同时**尽量少使用引擎推力**，并在接触时保持**姿态稳定**和**安全接触**。  
核心目标：到达目标平台并稳定停驻。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float64（默认连续值）
- obs[0]：x_position —— 相对于目标平台的水平坐标  
- obs[1]：y_position —— 相对于平台高度的垂直坐标  
- obs[2]：x_velocity —— 水平线速度  
- obs[3]：y_velocity —— 垂直线速度  
- obs[4]：body_angle —— 机体倾角  
- obs[5]：angular_velocity —— 角速度  
- obs[6]：left_support_contact —— 左脚接触标志（1.0/0.0）  
- obs[7]：right_support_contact —— 右脚接触标志（1.0/0.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0：no_engine —— 无推力
- action 1：left_orientation_engine —— 启动左侧姿态引擎（产生角速度/微小推力）
- action 2：main_engine —— 启动主引擎（产生主要上升/减速推力）
- action 3：right_orientation_engine —— 启动右侧姿态引擎（与左边方向相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled` —— 机体停止运动并稳定（即成功着陆并停驻）
- failure-like termination: 
  - `crash_or_body_contact` —— 任何非法的碰撞或身体接触（如撞到非平台区域、过猛撞击）
  - `horizontal_position_outside_viewport` —— 水平位置超出视口边界
- ambiguous termination: 无
- truncation: 未出现在 step 中（`truncated` 返回 `False`），说明无时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（`info` 为空字典 `{}`）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均禁止使用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（上一时刻的 8 维观察）
- action（执行的动作 id）
- next_obs（下一时刻的 8 维观察，包含新位置、速度、角度、接触标志等）
- info 中明确允许的字段（当前无）

禁止使用：
- original_reward —— 官方原始奖励已遮盖，严禁使用
- official_reward —— 同上
- training_progress —— 未在任务说明中允许，禁止使用
- 任何未声明的 info 字段

## 7. 可用于奖励函数的信号
- position: next_obs[0] (水平位置误差), next_obs[1] (垂直位置/接近平台高度)  
- velocity: next_obs[2] (水平速度), next_obs[3] (垂直速度) —— 可用于惩罚硬着陆  
- orientation: next_obs[4] (倾角) —— 可用于要求平稳姿态  
- contact: next_obs[6], next_obs[7] (左右接触标志) —— 可奖励双脚稳定着地  
- action/engine: action id —— 可惩罚使用引擎推力以促进节能

## 8. 不确定或不可用的信号
- 显式的成功/失败标志（info 中不存在）  
- 任何关于“碰撞”或“出界”的实时信号（仅通过终止条件反馈，无法在内部步获取）  
- 任务说明中提到的“尽快”和“尽量少用推力”无额外量化信号，仅可通过自身设计（如通过时间累积或动作惩罚）间接体现，但禁止依赖 unknown info