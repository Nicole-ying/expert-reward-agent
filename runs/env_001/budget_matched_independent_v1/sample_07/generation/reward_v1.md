# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ======== 观测解包 ========
    x_pos = obs[0]          # 水平偏移
    y_pos = obs[1]          # 垂直偏移
    x_vel = obs[2]          # 水平速度
    y_vel = obs[3]          # 垂直速度
    angle = obs[4]          # 姿态角
    ang_vel = obs[5]        # 角速度
    left_contact = obs[6]   # 左接触
    right_contact = obs[7]  # 右接触
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # ======== 组件A: 主学习信号 - 接近目标 (approach_to_target) ========
    # 使用欧氏距离的负线性形式，提供每步梯度
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_distance = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    
    # 使用 improvement_delta: 距离减少即正奖励
    approach_reward = distance - next_distance  # 正表示靠近目标
    # 缩放至合理范围，避免被其他组件主导
    A = 5.0 * approach_reward

    # ======== 组件B: soft_health_gate - 姿态与速度健康门控 ========
    # 在接近目标时，如果姿态或速度不合适，衰减主奖励
    # 使用倒数衰减形式，当姿态角或速度过大时门值降低
    
    # 姿态因子: 角度越接近0越好
    angle_factor = 1.0 / (1.0 + 10.0 * abs(angle))
    
    # 速度因子: 总速度越接近0越好，但只在接近目标时激活
    total_speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + 5.0 * total_speed)
    
    # 门控因子: 乘积形式，两个条件都需要满足
    # 使用几何平均避免乘积塌缩
    health_gate = (angle_factor * speed_factor) ** 0.5
    
    # 只在飞行器接近目标地面时应用门控 (y_pos 较小)
    # 使用平滑过渡，避免硬阈值
    height_proximity = 1.0 / (1.0 + 10.0 * abs(y_pos))
    
    # 组合门控: 当接近地面时，health_gate生效；远离时gate=1.0
    effective_gate = height_proximity * health_gate + (1.0 - height_proximity) * 1.0
    
    # 将门控应用到主奖励上
    # 但保持正向 progress 不受完全抑制，避免"不敢动"
    B = A * (effective_gate - 1.0)  # 附加项：当health良好时≈0，恶化时惩罚主奖励
    B = min(B, 0.0)  # 只衰减，不增加

    # ======== 组件C: 速度惩罚 - 轻量约束 ========
    # 对高速运动给予温和惩罚，防止翻滚和撞击
    # 使用二次惩罚，但保持权重较小
    speed_penalty = -0.1 * total_speed ** 2
    
    # 角速度惩罚: 防止过快旋转
    ang_vel_penalty = -0.05 * ang_vel ** 2
    
    C = speed_penalty + ang_vel_penalty

    # ======== 组件D: 姿态约束 - 铰链形式 ========
    # 只在角度较大时惩罚，允许小幅摆动
    angle_abs = abs(angle)
    angle_threshold = 0.3  # 约17度，在此范围内不惩罚
    D = -0.2 * max(0.0, angle_abs - angle_threshold)

    # ======== 总奖励组装 ========
    total_reward = A + B + C + D

    components = {
        'approach_reward': A,
        'health_gate_modulation': B,
        'speed_penalty': C,
        'angle_hinge_penalty': D
    }

    return float(total_reward), components
```

---

# reward_v1 设计说明

## 任务画像
- **task_family**: navigation_goal_reaching  
- **dynamics_subtype**: goal_approach_and_soft_contact  
- **核心目标**: 飞行器从扰动初始状态到达目标平台，稳定停驻（位置、速度、姿态均需收敛）  
- **无显式成功/失败标志**: expert_profile中确认无`info`字段可用

## 选定的奖励职责 (Selected Reward Roles)

### 1. **approach_to_target** (mandatory role)  
- **信号**: `obs[0]` (x_position), `obs[1]` (y_position)  
- **公式算子**: `improvement_delta` — 使用`old_distance - new_distance`  
- **实现**: 组件A，权重大(5.0)，提供核心驱动力  
- **为何这样选**: 每步都有梯度，直接反映任务进展；线性正奖励避免复杂化

### 2. **soft_health_gate** (conditional role)  
- **信号**: `obs[2-3]` (速度), `obs[4]` (角度), `obs[1]` (高度贴近度)  
- **公式算子**: `soft_health_gate` — 使用`1/(1+k*error)`构建各因子，乘积后弱化主奖励  
- **实现**: 组件B，仅衰减不增加  
- **为何这样选**: 防止agent“先冲后死”（快速到达但姿态/速度不合规触发crash）。gate在状态恶化时切断主奖励，比独立惩罚更高效

### 3. **速度约束** (conditional role)  
- **信号**: `obs[2-3]` (速度), `obs[5]` (角速度)  
- **公式算子**: `quadratic_penalty` — 轻量`-w * value**2`  
- **实现**: 组件C，权重极小(0.1/0.05)  
- **为何这样选**: 降低高速翻滚风险，但不压制探索

### 4. **姿态铰链约束** (conditional role)  
- **信号**: `obs[4]` (角度)  
- **公式算子**: `dense_state_signal (hinge)` — `-w * max(0, abs(angle)-threshold)`  
- **实现**: 组件D，只在超出0.3rad时惩罚  
- **为何这样选**: 允许小幅姿态调整（着陆过程必需），避免全时惩罚导致agent不敢转体

---

## 排除的职责 (Excluded Roles)

### ❌ **soft_landing** (mandatory role in environment_card)  
- **排除原因**: 该角色定义为“促使在接近目标时速度、姿态、接触达标”，但**无单一简单信号**可实现此目标。  
- **替代方案**: 将其拆分为 `soft_health_gate`（姿态+速度门控）+ `approach_to_target`（位置驱动）。接触信号自然出现在着陆末期，v1不显式奖励，避免伪造success proxy。

### ❌ **minimize_engine_usage** (secondary role)  
- **排除原因**: 效率优化留到v2；v1首要任务是学会着陆基本行为，动作惩罚可能让agent不愿使用引擎。

### ❌ **hard_terminal_constraints**  
- **排除原因**: 无显式failure flag，无法实现终端惩罚；crash检测依赖环境内部逻辑，我们无法获取。

---

## 为何没有使用 terminal_success_reward / terminal_failure_penalty?

- **explicit_success_flag_available = false** — 无`info["success"]`字段  
- **explicit_failure_flag_available = false** — 无`info["failure"]`或`termination_reason`  
- 终止信号（如`body_not_awake_or_settled`）为**环境内部判断**，不暴露给reward函数。  
- 因此v1完全基于连续信号塑造，不使用稀疏终端奖励。

---

## 职责留待后续迭代

| 职责 | 留待原因 |
|------|----------|
| minimize_engine_usage | 效率优化，v2加入动作代价或fuel proxy |
| soft_landing (完整版) | 需要更复杂的joint_condition_proxy，v2可加入接触条件乘积 |
| terminal_success_proxy | 当前无可靠success信号；v2可尝试基于多条件连续乘积构造soft proxy |
| curriculum_weighting | training_progress不可靠，v1无历史参考 |

---

## 训练后应观察的 failure modes

1. **hover_above_pad**: agent悬停在空中静止，获得中等distance reward但不降落。  
   - 原因: `approach_reward`已经很小，缺乏向下的驱动力。  
   - 检查: y_position是否接近零但速度为零，接触标志持续为0。

2. **fast_crash_landing**: agent快速冲向平台但速度/姿态失控，触发crash。  
   - 原因: `health_gate`衰减不够强或激活太晚。  
   - 检查: terminated率高，且`health_gate_modulation`在失败episode中接近0。

3. **single_contact_stand**: agent用单个支撑点站立，不稳定但未触发crash。  
   - 原因: v1未显式奖励双接触，agent可能满足于单点平衡。  
   - 检查: left_contact和right_contact交替为1，而非同时为1。

4. **oscillation_around_center**: agent在目标点附近来回摆荡，无法停驻。  
   - 原因: `approach_reward` delta形式可能在接近零时震荡（距离减小-增大循环）。  
   - 检查: x_position频繁变号，velocity方向反复切换。