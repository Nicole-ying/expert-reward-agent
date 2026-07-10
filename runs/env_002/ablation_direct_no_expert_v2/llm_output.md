# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward v1.
    
    Goal: Walk forward as far and as fast as possible while staying upright.
    Components:
    1. forward_reward   - reward for positive horizontal velocity
    2. upright_penalty  - penalty for body tilt (hull_angle)
    3. action_cost      - tiny penalty for joint torque usage (energy efficiency)
    """
    # Extract observation slices
    hull_angle = obs[0]               # body tilt angle (0 = upright)
    horizontal_velocity = obs[2]      # forward velocity (>0 = forward)

    # Main learning signal: reward forward progress
    forward_reward = max(horizontal_velocity, 0.0)    # no reward for moving backward

    # Stability constraint: penalise deviation from upright
    upright_penalty = abs(hull_angle)

    # Energy efficiency: small penalty on squared torques
    action_cost = sum(a * a for a in action)

    # Weights
    w_vel    = 1.0
    w_upright = 0.5
    w_action  = 0.01

    total_reward = w_vel * forward_reward - w_upright * upright_penalty - w_action * action_cost

    components = {
        "forward_reward": forward_reward,
        "upright_penalty": upright_penalty,
        "action_cost": action_cost
    }
    return float(total_reward), components
```

# 设计说明

### 任务类型判断
2D 双足行走（bipedal locomotion），目标是在不平坦地形上尽可能快、远地前进，同时保持平衡避免摔倒，并隐含降低能耗。属于 **locomotion** 任务。

### 组件选择与职责
- **主学习信号（forward_reward）**：使用 `horizontal_velocity`（obs[2]）的正值部分。每步都有梯度，直接驱动“向前走”这一核心目标。负速度（后退）不给奖励，避免捷径。
- **稳定约束（upright_penalty）**：使用 `hull_angle`（obs[0]）的绝对值惩罚身体倾斜。角度偏离 0 越大惩罚越重，引导策略保持直立，这是防止摔倒的最关键连续信号。
- **效率约束（action_cost）**：使用动作向量的平方和，权重极小（0.01）。仅轻微抑制不必要的大扭矩，避免能量浪费，但不主导主学习信号。

### 数学形式选择
- `max(horizontal_velocity, 0)`：线性奖励正向速度，简单且安全，不会奖励后退。
- `abs(hull_angle)`：线性惩罚，比二次惩罚更鲁棒，不会在小角度时梯度消失，也不会对大角度过度惩罚导致数值爆炸。
- `sum(a^2)`：经典的二次代价，抑制大动作幅值，平滑控制。

### 排除的职责
- **存活奖励**：未使用。终止条件（摔倒）本身会截断 episode，提供隐式负信号，单独再加存活奖励容易鼓励原地站立，与前进目标冲突。
- **关节速度/协调奖励**：v1 仅提供最基础的前进+稳定驱动，复杂的步态协调留给后续迭代。
- **terminal success/failure 奖励**：避免伪造停止信号，且主信号+约束已经足够。
- **动态权重或门控**：v1 保持静态简单。

### 预期 failure modes
- 智能体可能学出快速但踉跄的步态，若 `upright_penalty` 权重不足，可能为了速度牺牲稳定。  
- 若地形复杂，仅靠 hull angle 可能不足以避免所有摔倒，后期可加入足端接触信息约束。  
- action_cost 极小，不会显著抑制扭矩输出，仅提供微弱正则，基本不影响学习速度。