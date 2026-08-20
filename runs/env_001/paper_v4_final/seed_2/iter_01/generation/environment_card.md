# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 飞行器（着陆器）轨迹优化任务。主体从视口顶部中心附近出发，初始受到随机扰动。核心目标是 **尽快、平稳地降落到中央目标平台上，并保持机身竖直稳定**。次要目标是 **尽量节省主引擎燃料**，即少用主推力。  
Agent 需要学会：向目标平台逼近、适时减速、保持小角度、最终实现低冲击的安全着陆。  
不要把“平稳着陆”与单纯的“位置到达”混淆，着陆质量（速度、姿态、接触）与燃料效率不可忽略，但到达目标是第一优先级。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 任务的主问题是要到达指定的目标位置（平台中央），并附加着陆姿态与速度约束。附属的能源优化不是主目标，因此不是 multi_objective_task。该环境没有持续的存活/平衡压力，也不是纯粹的 locomotion 或 manipulation；核心是到达和稳定在目标点，所以属于 goal reaching 族。

## 3. 观察空间 observation_space
- **type**: `Box`
- **shape**: `[8]`
- **dtype**: `float32` (推断)
- **obs[0]**: `x_position` —— 相对于目标平台中心的水平坐标（正右方向），reward_usable: true
- **obs[1]**: `y_position` —— 相对于目标平台着陆面高度的垂直坐标（疑似上为正，平台面为 0），reward_usable: true
- **obs[2]**: `x_velocity` —— 水平线速度，reward_usable: true
- **obs[3]**: `y_velocity` —— 垂直线速度，reward_usable: true
- **obs[4]**: `body_angle` —— 机身倾角（很可能以弧度表示，0 为竖直），reward_usable: true
- **obs[5]**: `angular_velocity` —— 角速度，reward_usable: true
- **obs[6]**: `left_support_contact` —— 左支撑脚接地标志（1.0 表示接触），reward_usable: true
- **obs[7]**: `right_support_contact` —— 右支撑脚接地标志，reward_usable: true

## 4. 动作空间 action_space
- **type**: `Discrete`
- **n**: 4
- **动作清单**：
  - **action 0**: `no_engine` —— 所有引擎关闭
  - **action 1**: `left_orientation_engine` —— 启动左姿态引擎（产生角力矩，主要用于调整机头方向）
  - **action 2**: `main_engine` —— 启动主引擎（向下喷气，产生向上的推力，同时可能带来微小角力矩）
  - **action 3**: `right_orientation_engine` —— 启动右姿态引擎（与左姿态引擎相反方向）

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：机体与地面或平台发生强烈接触（可能包含坠毁或非常粗糙的着陆），触发终止。
- **horizontal_position_outside_viewport**：水平位置超出有效视野/世界边界，任务失败终止。
- **body_not_awake_or_settled**：机体进入稳定/不活跃状态（如着陆后静止），终止发生。此极可能是成功着陆后的正常终止。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 无（环境 step 返回的 `info` 为空字典 `{}`）  
- **forbidden_or_uncertain_info_fields**: 所有可能的终止原因、成败标记、elapsed steps 等均不可直接使用  
- 尽管如此，成功着陆的迹象可通过 **next_obs** 间接推断：
  - 位置接近目标原点 `(0,0)`，速度接近 0，角度接近 0，且左右支撑接地标志同时为 1。  
  - 该推断路径记为 **derived_possible**，可在奖励设计中使用，但不可作为绝对成功判决。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training