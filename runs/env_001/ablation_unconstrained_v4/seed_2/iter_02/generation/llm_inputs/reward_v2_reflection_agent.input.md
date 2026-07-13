# ⚠️ 上一版代码验证失败
错误信息：Reward v2 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs\env_001\ablation_unconstrained_v4\seed_2\iter_02\generation\validations\reward_v2.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
我们先基于训练反馈分析问题，再检索相关经验。

**分析**：
1. **agent发生了什么？**  
   - 全部20个episode都以early terminal结束（<150步且score<-50），表明agent无法完成着陆，全部失败（可能因撞击或越界）。  
   - 平均长度68.5步，说明失败发生得很快。  
   - reward组件中`goal_proximity`（由`dist_old - dist_new`线性进展奖励主导）总和为-73.93，即每个episode平均累积负值，意味着agent在大部分时间**距离目标平台越来越远**（可能在水平方向漂移出界）。  
   - `contact_establishment`正奖励总和仅3.4，active_rate 3.2%，说明接触很少发生，agent未能接近地面并让支撑腿接地。  
   - 速度/姿态约束只在距离<2.0时才激活（`gate_speed`），而agent可能在此之前就已高速、大角度翻滚导致失败，这些约束形同虚设。  
   - 总体，当前奖励未能提供有效的水平对准引导和全局姿态/速度安全约束，导致agent在早期就因失控而失败。

2. **最值得干预的组件**：  
   - **根本缺失**：水平位置控制。环境失败条件之一是`horizontal_position_outside_viewport`，而当前奖励对x坐标偏离没有任何直接惩罚，导致agent容易飘出边界。应增加**全局水平对准惩罚**，强制agent保持在中心附近。  
   - **约束时机错误**：速度与姿态约束设置了近距离门控（gate），而agent在远距离已经崩溃。需要将关键安全约束（角度、角速度、下降速度）改为**无门控的hinge惩罚**，始终生效。  
   - **着陆激励不足**：接触奖励权重较低且仅在`closeness_land`高时才有效，agent难以获得。可以加入更明确的**双脚着地成功信号**（接触+姿态竖直+低速），作为密集化的成功代理，引导agent追求稳定着陆状态。

3. **上一轮动作效果**：上一轮从无到有定义了四个组件（进展、速度约束、姿态约束、接触），但结果极差，得分-111.84。说明当前结构需要重大调整，而非微调参数。

接下来，检索相似失败模式的知识，看看经验修复方案。
搜索：`search_reward_design_knowledge("agent fails early due to horizontal drift and late engagement of constraints, linear progress reward negative")`

# Search objective
- target_score: 200.000000
- current_score: -111.843325
- gap_to_target: 311.843325
- target_achievement_ratio: -55.922%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -111.843325）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    dist_old = (x_old**2 + y_old**2)**0.5
    dist_new = (x_new**2 + y_new**2)**0.5
    
    w_progress = 1.0
    w_vel = 0.5
    w_att = 0.5
    w_contact = 1.0
    d_close = 2.0
    vx_max = 0.5
    vy_max = 0.5
    angle_max = 0.2
    angvel_max = 0.2
    d_land = 1.0
    
    # 1. 向目标移动的稠密进展奖励 (improvement_delta)
    progress_reward = w_progress * (dist_old - dist_new)
    
    # 2. 接近目标时速度约束 (hinge penalty with gate)
    gate_speed = max(0.0, 1.0 - dist_new / d_close)
    excess_vx = max(0.0, abs(vx_new) - vx_max)
    excess_vy = max(0.0, abs(vy_new) - vy_max)
    vel_penalty = -w_vel * gate_speed * (excess_vx**2 + excess_vy**2)
    
    # 3. 接近目标时姿态稳定性约束 (hinge penalty with gate)
    excess_angle = max(0.0, abs(angle_new) - angle_max)
    excess_angvel = max(0.0, abs(angvel_new) - angvel_max)
    att_penalty = -w_att * gate_speed * (excess_angle**2 + excess_angvel**2)
    
    # 4. 软着陆接触奖励 (continuous closeness × contact average)
    closeness_land = max(0.0, 1.0 - dist_new / d_land)
    avg_contact = (left_contact + right_contact) / 2.0
    contact_bonus = w_contact * avg_contact * closeness_land
    
    total = progress_reward + vel_penalty + att_penalty + contact_bonus
    components = {
        'progress_reward': progress_reward,
        'vel_penalty': vel_penalty,
        'att_penalty': att_penalty,
        'contact_bonus': contact_bonus
    }
    return float(total), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-111.843325, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-129.705727, -78.933292]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | -73.933146 | -84.0% | 84.0% | 100.0% |
| soft_landing_stabilization | -9.302658 | -10.6% | 10.6% | 99.8% |
| contact_establishment | 3.400000 | 3.9% | 3.9% | 3.2% |
| upright_attitude | -1.394754 | -1.6% | 1.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主任务：在二维平面中，控制一个带有两侧支撑腿的着陆器从视野上方中心附近受随机初速出发，**安全、稳定地降落到视野中央的目标平台上**，要求着陆器最终静止、姿态竖直且两侧支撑腿均与平台接触。  
次任务：在保证主目标的前提下，**尽可能快地完成着陆**，并且**尽量降低发动机推力消耗**（即少用燃料）。  
不应混淆的目标：燃料优化与快速着陆是辅助目标，它们不能凌驾于安全稳定着陆之上；本任务的最终评判是“是否成功停靠在平台上”这一离散事件，并非将能耗作为等价的主要目标。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: 根据上下文推断为 float32 (或混合了 bool 转为 float 的标志位，实际为连续量)
- **各维度含义**：
  - `obs[0]`: x_position (目标平台的相对水平坐标) —— reward_usable: true，用于控制水平对准。
  - `obs[1]`: y_position (相对于目标平台高度的垂直坐标) —— reward_usable: true，用于测量剩余高度/下降进度。
  - `obs[2]`: x_velocity (水平线速度) —— reward_usable: true，用于减速控制。
  - `obs[3]`: y_velocity (垂直线速度) —— reward_usable: true，用于控制坠落/着陆速度。
  - `obs[4]`: body_angle (机体偏转角) —— reward_usable: true，用于保持竖直姿态。
  - `obs[5]`: angular_velocity (角速度) —— reward_usable: true，姿态稳定惩罚。
  - `obs[6]`: left_support_contact (左支撑腿接触标志，1.0表示接触) —— reward_usable: true，接地判断。
  - `obs[7]`: right_support_contact (右支撑腿接触标志，1.0表示接触) —— reward_usable: true，接地判断。

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `action 0` (no_engine): 不启动任何引擎（零推力）。
  - `action 1` (left_orientation_engine): 启动左侧姿态引擎，产生旋转力矩。
  - `action 2` (main_engine): 启动主引擎，产生向上的升力。
  - `action 3` (right_orientation_engine): 启动右侧姿态引擎，产生相反的旋转力矩。
- **值域**：`[0, 1, 2, 3]`，每个动作只消耗一个离散选择的“燃料单位”。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` —— 机体静止且（可能）已与平台产生接触。根据任务描述“settle at a central target pad... make safe contact”，此条件很可能标志着成功着陆。但是否伴随 `left_support_contact` 和 `right_support_contact` 同时为真，需要观察，但终止代码中只用此单一条件判定终止，可能表示“已稳定着陆”。
- **failure-like termination**:  
  `crash_or_body_contact` —— 机体与外环境发生碰撞（非支撑腿安全着陆），通常是坠毁。  
  `horizontal_position_outside_viewport` —— 水平飘出视野边界，失败。
- **ambiguous termination**:  
  在没有显式成功标志的情况下，终止由这三种条件之一触发，其中 `body_not_awake_or_settled` 语义为“非清醒或已稳定”，我们暂时将其视为**成功候选**，但无法 100% 确认。需要后续根据经验收集证据。
- **truncation**:  
  源代码未展示任何步数限制或超时截断（`return ..., False, {}` 中没有 truncated）。如果存在，可能在上层包装器中，但当前分析不可见。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false （info 字典为空）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（info={} 且没有声明任何可用字段）
- **forbidden_or_uncertain_info_fields**: 所有 info 字段都不允许使用，因为没有可靠信息源。`original_reward` 被屏蔽，禁止访问。

## 7. 可用于奖励函数的信号
- **position**:
  - `next_obs[0]` (x_position)：到目标平台的水平距离。
  - `next_obs[1]` (y_position)：到目标平台高度的垂直距离。
- **velocity**:
  - `next_obs[2]` (x_velocity)
  - `next_obs[3]` (y_velocity)
- **orientation**:
  - `next_obs[4]` (body_angle)
  - `next_obs[5]` (angular_velocity)
- **contact**:
  - `next_obs[6]` (left_support_contact)
  - `next_obs[7]` (right_support_contact)
- **action/engine**:
  - `action`：离散动作编号，可用于惩罚引擎使用（燃料消耗），但不能直接读出推力大小。
- **other**:  
  无。