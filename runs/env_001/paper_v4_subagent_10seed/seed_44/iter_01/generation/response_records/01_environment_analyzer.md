# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境为一个二维轨迹优化任务：代理人（飞行器）初始位于视口顶部中央附近，携带随机初始力。核心任务是**尽快到达并稳定停靠在画面中央的目标着陆垫上**，同时**尽可能减少发动机推力使用**。代理人需要学会接近目标、减速、保持姿态稳定并实现安全接触。

主要目标：在尽量短的时间内，让身体相对着陆垫的位置（x, y）趋近于零，同时速度降为零，姿态保持竖直（body_angle≈0），并使左右两个支撑脚同时接触。  
次要目标：最小化燃料消耗（动作中使用发动机的次数）。  
不应混淆为仅需靠近即可得分的悬浮任务——单纯悬浮不应获得持续奖励，且最终必须实现有接触的稳定停靠。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 核心目标是到达指定的目标位置（x=0, y=0 附近），并满足姿态和接触等附加约束。接近目标的过程可以自然通过位置差来衡量进度；燃料消耗和接触姿态是附属要求，不对任务族构成独立权重相当的冲突目标。  

dynamics_subtype: **goal_approach_and_soft_contact**  
解释：在导航到达的基础上，动力学强调减速、稳定接触（双脚着垫、低速、竖直姿态），需要软着陆性行为。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64（推断为 float，实际代码中可能 float32/float64，对奖励无影响）  
- 各维度含义（均基于 next_obs 视角，无历史滑动窗口）：

| 索引 | 名称                     | 含义                                                                 | reward_usable |
|------|--------------------------|----------------------------------------------------------------------|---------------|
| 0    | x_position               | 身体水平坐标，相对于着陆垫中心的偏移                                   | true          |
| 1    | y_position               | 身体垂直坐标，相对于着陆垫高度的偏移                                   | true          |
| 2    | x_velocity               | 水平线速度                                                            | true          |
| 3    | y_velocity               | 垂直线速度                                                            | true          |
| 4    | body_angle               | 身体朝向角（弧度）                                                    | true          |
| 5    | angular_velocity         | 角速度                                                                | true          |
| 6    | left_support_contact     | 左支撑脚接触标志（1.0 接触，0.0 未接触）                             | true          |
| 7    | right_support_contact    | 右支撑脚接触标志（1.0 接触，0.0 未接触）                             | true          |

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作表：

| 动作 id | 名称                        | 含义                                                         |
|--------|-----------------------------|------------------------------------------------------------|
| 0      | no_engine                   | 不启动任何引擎，自由漂移                                    |
| 1      | left_orientation_engine     | 点燃左侧姿态引擎（产生逆时针？力矩，用于调整姿态）           |
| 2      | main_engine                 | 点燃主引擎（产生垂直向上的推力？或向下的推力？根据相对坐标系，可能提供垂直方向推力抵消重力/加速） |
| 3      | right_orientation_engine    | 点燃右侧姿态引擎（产生与左侧相反的力矩）                     |

注：虽然动作空间为离散，但动力学为连续（位置、速度、角度）。主引擎和姿态引擎的具体推力方向由底层物理决定，奖励函数只需知道动作 ID 即可识别是否使用了推力（id ≠ 0 时为有燃料消耗的动作）。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  - `body_not_awake_or_settled` 中的 *settled* 状态：当身体处于静止、双支撑脚接触、且位置/姿态满足一定标准时，被判为已停稳，终止回合。此即任务成功的信号。  
  - 从观测推断：若终止发生时，左右接触标志均为 1，位置 (0,0) 附近，角度≈0，速度≈0，则极大概率为成功。
- **failure-like termination**:  
  - `crash_or_body_contact`：身体与不可碰撞部位（如地面或非着陆垫物体）发生接触，或除支撑脚外的部位触地。  
  - `horizontal_position_outside_viewport`：水平位置超出视野边界。  
  - `body_not_awake_or_settled` 中的 *body_not_awake*：身体失去“意识”（可能因高速撞击、翻滚导致），但并非稳定停泊，属于失败。
- **ambiguous termination**:  
  - 仅有终止信号，没有显式 success/failure 标志时，需根据最终观测状态判断成败。  
  - `body_not_awake_or_settled` 内部可能包含成功（settled）和失败（not awake），完全依赖于观测解读。
- **truncation**: 本描述中未见最大步长截断（MASKED_STEP_SOURCE 中 `truncated=False` 始终返回），因此所有终止均为 terminated=True。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **false**  
- allowed_info_fields: 根据 step 源码，info 字典为空，**无任何可用字段**。  
- forbidden_or_uncertain_info_fields: 任何 info 字段均不可用；不得假设存在 `success`、`failure`、`termination_reason` 等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs`：上一步观测（shape 8）
- `action`：上一步动作（int, 0~3）
- `next_obs`：当前步观测（shape 8）
- `info`：**但 info 为空字典，无可用字段**，等同于不可用

**禁止使用**：
- `original_reward`（官方奖励被遮蔽，必须忽略）
- 任何未在上述允许列表中的信号（如 info
