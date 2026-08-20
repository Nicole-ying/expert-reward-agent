# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从视口顶部中央附近出发，尽快飞到画面中心的着陆垫上并稳定停靠。过程中需要尽量节省引擎推力，同时避免发生碰撞、水平出界或提前休眠。最终期望的位置是着陆垫上方、接近静止、姿态稳定且左右支撑脚均与垫子接触。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据常规推断，其中 contact 标志用 0.0/1.0 表示）
- obs[0] (x_position): 飞行器相对着陆垫中心的水平坐标（可能是负、零或正）
- obs[1] (y_position): 飞行器相对着陆垫高度的垂直坐标（正值在上方，负值在下方）
- obs[2] (x_velocity): 水平线速度
- obs[3] (y_velocity): 垂直线速度
- obs[4] (body_angle): 机体朝向角度（弧度或度？具体范围由物理实现决定，但在奖励中应处理为接近 0 表示竖直姿态）
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左支撑脚是否接触着陆垫（0.0 无接触，1.0 接触）
- obs[7] (right_support_contact): 右支撑脚是否接触着陆垫（0.0 无接触，1.0 接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine —— 什么都不做（不点火，惯性和风力影响下自由飘动）
- action 1: left_orientation_engine —— 点火某个方向的姿态控制引擎（用于向左旋转机体）
- action 2: main_engine —— 点火主引擎（提供向上的推力，用于减速或上升）
- action 3: right_orientation_engine —— 点火相反方向的姿态控制引擎（用于向右旋转机体）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 飞行器在着陆垫上方稳定着陆，速度很小，角度接近 0，且左右支撑脚均接触（即 body_not_awake_or_settled 触发，同时位置、速度、角度、接触均满足期望状态）。该情形隐含“成功到达目标并安全停靠”。
- failure-like termination: 
  - crash_or_body_contact（飞行器主体与地面或边界发生破坏性碰撞）
  - horizontal_position_outside_viewport（飞行器水平飞出画面）
- ambiguous termination: body_not_awake_or_settled 可能既包含成功着陆（满足条件），也可能包含飞行器在非目标位置提前休眠（例如坠落在远处后不动、或能量耗尽停止）。因此该终止条件本身不能直接等同于成功，必须结合位置、接触等信息判断。
- truncation: 源码中为 `False`，未显示任何截断机制，可假设无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典，无任何显式标记）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空，无有效字段可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段（因为 info 为空，不能假设存在 success、failure 或其他标记）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用：**
- obs：上一时刻的完整 8 维观察（可在函数内部使用，但通常奖励基于下一状态 next_obs 更多）
- action：刚刚执行的动作（0~3）
- next_obs：执行动作后的完整 8 维观察，是奖励计算的主要依据
- info：当前 step 返回的 info，这里为空字典 `{}`，因此不能依赖其中的任何字段
- training_progress：未在 prompt 中声明需要使用，仅作为保留参数，默认不允许用于奖励计算，除非 prompt 明确允许

**禁止使用：**
- original_reward（已明令禁止）
- 任何未被允许的 info 字段（info 为空，无字段可用）
- 任何未在观察空间中声明的变量（如官方隐藏的内部状态）

## 7. 可用于奖励函数的信号
以下信号均可从 next_obs（或结合 obs 计算增量）中提取，且属于任务相关的可靠特征：

- position: x_position, y_position（obs[0], obs[1]）—— 可衡量距着陆目标的远近、是否处于垫子正上方
- velocity: x_velocity, y_velocity（obs[2], obs[3]）—— 速度大小影响安全着陆和平顺性
- orientation: body_angle（obs[4]）—— 姿态是否竖直（接近 0）
- angular_velocity: obs[5] —— 旋转是否稳定
- contact: left_support_contact, right_support_contact（obs[6], obs[7]）—— 是否双脚接触，判断着陆状态
- action/engine: 可依据动作是否使用主引擎或姿态引擎来鼓励节能（如惩罚主引擎使用）

## 8. 不确定或不可用的信号
- 没有任何来自 info 的成功/失败/终止原因等信号
- 无从判断系统内部是否已判定“任务成功”（只能通过 next_obs 的数值组合自行推断）
- 可能存在的引擎燃料余量、风力大小等内部物理量完全未暴露，不可使用
- 原始奖励 original_reward 被明确禁止，不可使用



# expert_reward_context.md

# Expert Reward Context Disabled

This run is the w/o Expert RAG ablation. Design from environment facts only.
