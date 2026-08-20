# 匿名环境理解卡片

## 1. 任务目标
智能体控制一个2D飞行器，从视口顶部中心附近以随机初始力出发。  
核心目标是尽快飞行并**稳定降落在视口中央的目标垫上**。  
辅助要求：尽量减少主发动机推力消耗，保持姿态稳定，并安全接触目标垫。  
智能体需要学习接近目标、减速、调整姿态并让两个支撑脚接触目标垫，最终稳定停在垫子上。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32
- obs[0]: x_position – 水平位置（相对于目标垫中心）
- obs[1]: y_position – 垂直位置（相对于垫子高度）
- obs[2]: x_velocity – 水平线性速度
- obs[3]: y_velocity – 垂直线性速度
- obs[4]: body_angle – 主体旋转角度
- obs[5]: angular_velocity – 角速度
- obs[6]: left_support_contact – 左脚接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact – 右脚接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine – 不启动任何发动机，仅靠惯性运动
- action 1: left_orientation_engine – 启动左侧姿态发动机（产生旋转力矩）
- action 2: main_engine – 启动主发动机（产生向下的推力，即让飞行器向上减速或悬停）
- action 3: right_orientation_engine – 启动右侧姿态发动机（产生反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  飞行器与目标垫接触且稳定下来（body_not_awake_or_settled）且没有发生 crash 或越界。  
  具体表现为：在某一 episode 中，终止条件 `crash_or_body_contact` 为 false、`horizontal_position_outside_viewport` 为 false，而且 `body_not_awake_or_settled` 为 true。  
  （即飞行器在接触目标垫后速度/角速度变得极小，进入休眠状态。）
- failure-like termination:  
  - `crash_or_body_contact` 触发，例如飞行器主体或支撑脚撞击到垫子以外的地面或其他物体。  
  - `horizontal_position_outside_viewport` 触发，飞行器飞出水平边界。
- ambiguous termination:  
  `body_not_awake_or_settled` 可能在没有接触目标垫的情况下触发（比如飞行器飘到某处后静止），这种情况可能是失败，但难以仅靠观察空间区分究竟是哪种终止。
- truncation: 无（返回值 truncated 固定为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 始终为空字典，无任何额外信号）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（info 为空），不能依赖任何来自 info 的成功/失败标志。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs（当前观察）
- action（执行的动作）
- next_obs（下一观察）
- training_progress （仅在 prompt 明确允许时使用，当前未明确允许，建议不用）

禁止使用：
- original_reward（官方奖励已屏蔽，不得计算、拷贝或近似）
- info（为空，无可用字段）
- 任何未在上述观察空间中声明的 obs 切片或外部信号

## 7. 可用于奖励函数的信号
以下信号可直接从 obs / next_obs 中提取，供设计奖励函数使用：
- position: next_obs[0]（水平坐标），next_obs[1]（垂直坐标）  
- velocity: next_obs[2]（水平速度），next_obs[3]（垂直速度）  
- orientation: next_obs[4]（角度），next_obs[5]（角速度）  
- contact: next_obs[6]（左脚接触），next_obs[7]（右脚接触）  
- action: action 本身可用于限制发动机使用（0/2 主发动机是否使用，1/3 姿态发动机使用）

## 8. 不确定或不可用的信号
- 无法获取绝对成功/失败标签（info 为空，termination 原因未暴露具体类型）。  
- 无法区分 `body_not_awake_or_settled` 是由成功着陆引发的，还是其他静止导致的失败终止。  
- 无法获得“与目标垫接触”的直接标志（左右支撑接触可能还接触了非目标地面），因此需要组合位置、速度、接触、角速度等信号间接推断是否成功着陆。  
- 无法获得燃料消耗量、剩余时间、主体碰撞细节等任何 info 字段。  
- original_reward 被完全屏蔽，不可用。