# Response Record

# 匿名环境理解卡片

## 1. 任务目标
主目标：控制一个 2D 平面内的刚体飞行器（带有主引擎和两侧姿态引擎），从初始偏高位置出发，尽可能快地下降并稳定、安全地停在中央目标垫上。停靠时要求速度极低、姿态竖直、双腿同时接触垫面（soft settling）。  
次要目标：在达成主目标的前提下，最小化引擎推力使用（燃料效率），并避免坠毁、飞出视口或粗鲁撞击。  
不应混淆的目标：单纯省燃料而拒绝降落、长时间悬停但不靠近垫子、快速但剧烈碰撞都属于失败行为。

## 2. 任务类型选择
**selected_route_id:** `navigation_goal_reaching`  
**confidence:** high  
**reason:** 核心任务是“到达指定目标位置并稳定停靠”——存在明确的目标垫坐标，必须移动至该位置并满足接触/稳定条件。燃料效率仅为次要优化，不构成多目标冲突。该路由最贴合任务本质。

## 3. 观察空间 observation_space
- **type:** Box  
- **shape:** [8]  
- **dtype:** 假定为 float32，连续值；最后两维为 float 但实际是 0/1  
- **各维含义（下述均为可用于奖励计算的信号）：**  

| index | name                         | meaning                                                                   | reward_usable |
|-------|------------------------------|---------------------------------------------------------------------------|---------------|
| 0     | x_position                   | 刚体中心相对于目标垫中心的水平距离（正方向未知，可正可负）               | true          |
| 1     | y_position                   | 刚体中心相对于目标垫高度的垂直距离（正值表示在垫上方）                   | true          |
| 2     | x_velocity                   | 水平线速度                                                                 | true          |
| 3     | y_velocity                   | 垂直线速度（正值向上）                                                     | true          |
| 4     | body_angle                   | 刚体姿态角（弧度），0 可能表示竖直向上                                     | true          |
| 5     | angular_velocity             | 角速度                                                                     | true          |
| 6     | left_support_contact         | 左支撑腿是否接触目标垫/地面（1：接触，0：未接触）                         | true          |
| 7     | right_support_contact        | 右支撑腿是否接触目标垫/地面（1：接触，0：未接触）                         | true          |

## 4. 动作空间 action_space
- **type:** Discrete  
- **n:** 4  
- **动作说明：**  

| action_id | name                        | meaning                           |
|-----------|-----------------------------|-----------------------------------|
| 0         | no_engine                   | 不点火任何引擎（滑行）            |
| 1         | left_orientation_engine     | 点燃左侧姿态引擎（产生一个力矩和可能的侧向推力） |
| 2         | main_engine                 | 点燃主引擎（产生向上的推力，可能带少量横向分量） |
| 3         | right_orientation_engine    | 点燃右侧姿态引擎（与左引擎对称）  |

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination:** `body_not_awake_or_settled` —— 当刚体速度/角速度低到被视为“睡着”或稳定停靠时触发，这极有可能是成功着陆的信号（双腿接触、速度近零）。  
- **failure‑like termination:**  
   - `crash_or_body_contact` —— 与目标垫以外的其他物体（如地面、边界）发生危险碰撞或翻滚。  
   - `horizontal_position_outside_viewport` —— 飞出水平视口，代表完全失控。  
- **ambiguous termination:** `body_not_awake_or_settled` 也可能因为某些物理休眠设定错误触发，但通常视为成功。  
- **truncation:** 当前环境未提供明确的截断条件（可能具有最大步数限制，未在 spec 中列出，属于环境内部实现）。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available:** false（`info` 字典为空 {}，无 success 标记）  
- **explicit_failure_flag_available:** false（同上）  
- **allowed_info_fields:** 无（`info` 为空，不可用任何字段）  
- **forbidden_or_uncertain_info_fields:** 所有 `info` 字段均禁止使用；`original_reward` 必须被屏蔽，禁止读取。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
**允许使用：**
- `obs`：  当前步完整的 8 维观察（含位置、速度、角度、接触标志）
- `action`： 当前步执行的动作（0‑3 离散值）
- `next_obs`： 执行动作后下一时刻的完整 8 维观察
- `info`：  仅限明确声明为可用的字段，当前环境中 `info` 为空，因此不可用
- `training_progress`： 本提示未明确允许使用，**禁止**使用

**严格禁止：**
- `original_reward`（官方奖励）
- 未在 spec 中列出的任何 `info` 字段
- 假设存在 `info["success"]`、`info["failure"]` 等信号
- 读取代环境的内部状态或通过任意方式获取终止原因

## 7. 可用于奖励函数的信号
- **位置相关：**
  - `obs[0]`（x_position） / `next_obs[0]`
  - `obs[1]`（y_position） / `next_obs[1]`
- **速度相关：**
  - `obs[2]`（x_velocity） / `next_obs[2]`
  - `obs[3]`（y_velocity） / `next_obs[3]`
- **姿态相关：**
  - `obs[4]`（body_angle） / `next_obs[4]`
  - `obs[5]`（angular_velocity） / `next_obs[5]`
- **接触/着陆信号：**
  - `obs[6]`（left_support_contact） / `next_obs[6]`
  - `obs[7]`（right_support_contact） / `next_obs[7]`
- **动作/推力：**
  - `action` （离散值）可用来推断哪个引擎被点燃，从而施加推力惩罚
- **其他派生信号：**
  - 从原始信号可以衍生出 `dist_to_target = sqrt(x^2 + y^2)`、`speed = sqrt(vx^2 + vy^2)`、`both_legs_contact = left_contact & right_contact` 等，均为合法使用。

## 8. 不确定或不可用的信号
- 终止原因（crash / settle / out‑of‑viewport）—— 不可在 reward 中直接获取
- 引擎推力大小 —— 未直接给出，只能通过动作推断是否点火
- 燃料/剩余燃料 —— 未提供
- 到目标垫的距离角度 —— 只能从位置间接计算
- 任何来自 `info` 的字段 —— 当前为空，不可用
- `original_reward` —— 绝对禁止使用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete  # 引擎为离散开关
morphology:
  body_type: rigid_body_with_two_legs  # 类似着陆器
  actuator_type: main_engine + two_lateral_orientation_thrusters
  contact_structure: two_leg_contacts (left/right)
primary_objectives:
  - 到达目标垫正上方（x≈0, y≈0）
  - 以极低速度完成两条腿同时与垫面稳定接触（soft settling）
secondary_objectives:
  - 最小化引擎使用量（燃料经济性）
  - 在下降过程中保持姿态稳定（接近竖直）
  - 尽快完成着陆（时间效率）
main_failure_risks:
  - 与地面或其他构件发生剧烈碰撞（crash）
  - 水平位置飞出视口（完全失控）
  - 反复弹跳或无法稳定接触，导致进入不理想的状态并触发睡眠判定
  - 为了省燃料而悬停不降，任务不终止
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id:** `goal_proximity`  
  **purpose:** 鼓励 agent 不断接近目标垫中心（x=0, y=0），是完成导航的核心驱动力。  
  **why_required:** 没有该信号，agent 无法知道应该前往何处，任务无法完成。  
  **usable_signals:** [obs[0], obs[1], next_obs[0], next_obs[1]]  
  **risks:** 纯距离奖励可能导致 agent 高速撞向目标，因此必须配合软着陆限制。

- **role_id:** `soft_landing_contact`  
  **purpose:** 促使 agent 在到达目标时速度极低、姿态竖直，且双腿同时接触垫面，实现安全稳定停靠。  
  **why_required:** 任务明确要求 “settle at a central target pad”，且减少坠毁风险。  
  **usable_signals:** [obs[2..7], next_obs[2..7]]（速度、角度、角速度、双腿接触）  
  **risks:** 若权重不当，可能使 agent 过于谨慎而在高处长时间微调，降低效率；需与进度奖励平衡。

- **role_id:** `fuel_penalty`  
  **purpose:** 惩罚引擎使用，促进燃料效率。  
  **why_required:** 任务描述明确要求 “using as little engine thrust as possible”。  
  **usable_signals:** [action]（通过动作推断引擎是否点燃）  
  **risks:** 过强惩罚可能导致 agent 完全不用引擎，无法调整轨迹；需在训练初期较弱或仅在接近目标时启用。

### 10.2 条件职责 conditional_roles
- **role_id:** `orientation_stability`  
  **condition_to_use:** 当 agent 已非常靠近目标（例如 dist < 某个阈值）时，额外惩罚角度偏差和角速度，确保竖直姿态。全局施加可能导致前期抑制必要的姿态调整，因此推荐有条件使用。  
  **usable_signals:** [obs[4], obs[5], next_obs[4], next_obs[5]]  
  **risks:** 过早施加会限制 agent 探索必要的倾斜机动，可能无法成功导航。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id:** `success_bonus` （基于显式成功标志）  
  **reason:** 环境不提供任何 explicit success flag；info 为空，无法可靠获得成功事件。  
  **forbidden_or_missing_signals:** 禁止使用任何 `info["success"]` 或类似字段。

- **role_id:** `terminal_state_reward` （基于终止原因）  
  **reason:** 终止原因在 reward 函数内不可获取，且有的终止（crash）是失败，有的是成功，错误分类会严重误导学习。  
  **forbidden_or_missing_signals:** 没有可用的终止原因标识。

- **role_id:** `shaped_energy_limit` （对剩余燃料的约束）  
  **reason:** 没有燃料相关信号，无法实现。

## 11. role_to_signal_mapping
| role_id                | usable signals                                        | missing signals                      | candidate formula operators           | notes |
|------------------------|-------------------------------------------------------|--------------------------------------|----------------------------------------|-------|
| goal_proximity         | obs[0], obs[1]                                        | none                                 | dense_state_signal (negative L2 distance), bounded_signal | 核心驱动，全训练阶段有效 |
| soft_landing_contact   | obs[2], obs[3], obs[4], obs[5], obs[6], obs[7]        | explicit success flag               | quadratic_penalty on speed/angle, contact_bonus (both legs) | 速度、角度用连续惩罚，接触用离散奖励；速度项建议使用 next_obs 以保证安全 |
| fuel_penalty           | action                                                | fuel consumption (直接值)           | sparse_penalty (per engine fire)      | 可针对 main_engine 施加更高惩罚，姿态引擎惩罚较轻 |
| orientation_stability  | obs[4], obs[5]                                        | none                                 | quadratic_penalty, bounded_signal     | 推荐在接近目标（如高度 < 1.0）时激活 |

## 12. 初始训练后应观察的 failure modes
| failure_mode                              | evidence_to_check                                       | possible_intervention |
|-------------------------------------------|---------------------------------------------------------|-----------------------|
| 始终悬停在高空，不下落                     | y_position 长期大于某值，速度接近零，引擎几乎不点燃  | 增强 goal_proximity 对高度的权重，或给较低高度以额外奖励 |
| 高速撞击目标垫，导致 crash                | 靠近目标时速度极大，其后 episode 终止且 reward 骤降   | 强化 soft_landing_contact 的速度惩罚，或启动阶段弱化速度惩罚 |
| 反复弹跳无法稳定（残存速度或角速度）       | 双腿交替接触，接触后 y_velocity 仍震荡                 | 增加稳定接触后的持续奖励（如连续多步双腿接触且速度低给 bonus），惩罚未接触时的速度 |
| 飞出水平边界                              | x_position 绝对值快速增大 → episode 终止               | 添加对横向位置偏离的二次惩罚，或对 near‑boundary 的线性惩罚 |
| 姿态失控翻滚                              | angle 和 angular_velocity 持续大幅波动                  | 对 angular_velocity 施加强惩罚，或在下降阶段激活 orientation_stability |
| 只满足单腿接触，另一腿悬空但被判定 settled | 接触标志某一维度为 0，但 episode 终止（可能是误判成功） | 在奖励中要求双腿同时接触；若此行为出现且 episode 被视为成功，则必须修改终止条件或观察环境，奖励无法单独解决 |
