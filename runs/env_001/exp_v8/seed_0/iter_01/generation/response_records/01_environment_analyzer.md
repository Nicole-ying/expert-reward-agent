# Response Record

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
