# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器/着陆器从视口上方出发，尽快降落到画面中央的水平目标垫上并稳定停靠。主目标是精确到达并停稳在目标垫中心（位置误差趋于零，速度接近零，两支撑脚着垫）。次要目标是尽量减少引擎使用（节能），快速完成任务。注意不要与此类任务可能混淆的纯飞行姿态控制、单纯前进速度优化或仅存活不要求停稳的任务混淆。

## 2. 任务类型选择
- **selected_route_id:** `navigation_goal_reaching`
- **confidence:** high  
- **reason:** 任务叙述明确要求“reach and settle at a central target pad”，观测中包含相对于目标垫的水平/垂直坐标，终端条件中有到达后稳定停靠的事件（body_not_awake_or_settled）。附属有节能、快速要求，但不构成权重相当的多个冲突目标，核心仍是目标到达与停稳，故属于 navigation_goal_reaching。

## 3. 观察空间 observation_space
- **type:** Box  
- **shape:** (8,)  
- **dtype:** 通常为 float64（环境默认），可视为连续浮点数。  

各索引含义：  
- `obs[0]`：`x_position`，飞行器相对目标垫中心的水平距离（向右为正），reward_usable: true  
- `obs[1]`：`y_position`，飞行器相对目标垫高度的垂直距离（向上为正，0 表示与垫面等高），reward_usable: true  
- `obs[2]`：`x_velocity`，水平线速度，reward_usable: true  
- `obs[3]`：`y_velocity`，垂直线速度，reward_usable: true  
- `obs[4]`：`body_angle`，机身倾角（弧度，0 为水平），reward_usable: true  
- `obs[5]`：`angular_velocity`，角速度，reward_usable: true  
- `obs[6]`：`left_support_contact`，左侧支撑脚触地标志（0.0 或 1.0），reward_usable: true  
- `obs[7]`：`right_support_contact`，右侧支撑脚触地标志（0.0 或 1.0），reward_usable: true  

## 4. 动作空间 action_space
- **type:** Discrete  
- **n:** 4  
- **动作说明：**  
  - `action 0`：“no_engine” — 所有引擎关闭，无推力。  
  - `action 1`：“left_orientation_engine” — 点燃左侧姿态引擎，产生偏航/旋转力矩。  
  - `action 2`：“main_engine” — 点燃主引擎，产生主体推力（通常向上或沿机身轴线）。  
  - `action 3`：“right_orientation_engine” — 点燃右侧姿态引擎，产生反方向旋转力矩。  

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled`，三种触发情景：
- **crash_or_body_contact**：飞行器主体（非支撑脚）与地面或环境障碍碰撞，通常表示失败。  
- **horizontal_position_outside_viewport**：飞行器水平超出视口范围，失败。  
- **body_not_awake_or_settled**：物理体进入休眠状态或因稳定停靠而“settled”。根据任务目标，在目标垫上稳定停靠后应触发此条件，属于成功结果；但也可能因坠毁后体僵硬休眠触发，因此需要结合其他观测才能确定是成功还是失败。  

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** false  
- **explicit_failure_flag_available:** false  
- **allowed_info_fields:** `info` 当前为空字典 `{}`，无法直接获得任何结果标志。  
- **forbidden_or_uncertain_info_fields:** 任何未声明的字段（如 `success`、`failure`、`termination_reason` 等）均不可信。  

成功/失败只能通过 **derived_possible** 方式从观测序列中推断：  
- 成功终端（目标垫稳定停靠）：`episode` 结束时，`x_position`≈0, `y_position`≈0, `|x_velocity|` 和 `|y_velocity|` 很小，`left_support_contact`==1, `right_support_contact`==1，且未发生 `horizontal_out` 现象。  
- 坠毁终端：`episode` 结束时，倾角 `|body_angle|` 很大，或 `y_position` 异常低（地面以下），或只有一只脚接触物且位置远离目标垫。  
- 出界终端：`episode` 结束时，`x_position` 绝对值超出合理范围（范围需通过环境运行中观测到的边界估计，如 |x| > 1.5，或从 rollouts 中统计）。  

## 6. reward 函数接口契约
函数签名：  
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
- **允许使用：** `obs`、`action`、`next_obs`、`info`（目前为空）。  
- **禁止使用：** `original_reward`、任何官方奖励信号；未声明的 `info` 字段；任何未在观测空间中声明的切片；`training_progress` 除非 prompt 明确允许。  
- **额外约束：** 不能使用环境步长、真实时间刻度等隐含变量；奖励计算必须仅依赖当前步的 `obs` 和 `next_obs` 以及 `action`。  

## 7. 可用于奖励函数的信号
- **位置相关：**  
  - `x_position`, `y_position`（可直接计算到目标垫中心的欧氏距离 `dist = sqrt(x² + y²)`）  
  - 可衍生：`dist_to_target`，上一时刻距离与当前距离之差（delta progress）：`progress = dist(obs) - dist(next_obs)`，正值表示靠近。  
- **速度相关：**  
  - `x_velocity`, `y_velocity` 可用于惩罚接近时的剩余动能，或构建稳定条件。  
- **姿态相关：**  
  - `body_angle` 可用于 hinge penalty（防止倾斜过大）；`angular_velocity` 用于抑制快速旋转。  
- **接触信号：**  
  - `left_support_contact`, `right_support_contact` 可判断双脚是否着垫，是成功停靠的必要条件。  
- **动作相关：**  
  - `action` 值可用于计算动作成本（action ≠ 0 时轻微惩罚）。  
- **衍生信号（derived_possible，需与环境边界参数拟合）：**  
  - **终端成功事件：** 当 `terminated` 且 `dist_to_target` 小于阈值 (如 0.1)，速度幅值低于阈值，且 `left_support_contact` 和 `right_support_contact` 均为 1。  
  - **坠毁事件：** 当 `terminated` 且不满足成功条件，同时 `|body_angle|` 过大或 `y_position` 偏离过大。  
  - **出界事件：** 当 `terminated` 且 `x_position` 超出可靠运行范围。  

## 8. 不确定或不可用的信号
- 无任何显式成功/失败标志。  
- 无任何环境提供的进度度量（如剩余时间、能量剩余）。  
- 无明确的地面高度或视口边界坐标，只能通过大量 rollouts 推断出有效的 `x_position` 边界或安全 `y_position` 下限。  
- 无直接的“身体接触地面”标志，只有两只小腿的接触信息，坠毁时身体直接触地无法直接观测。  
- 无明确的燃料消耗量，只能通过 action 频率间接评估引擎使用；但任务要求“尽量少用引擎”，因此动作成本是合理副目标。  

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with two legs/supports
  actuator_type: