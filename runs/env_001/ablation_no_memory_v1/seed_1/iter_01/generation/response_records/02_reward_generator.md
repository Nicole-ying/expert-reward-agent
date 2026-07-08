# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D vehicle-like trajectory optimization:
    Reach and stabilize on the central target pad with minimal engine use.
    """
    # Unpack next observation
    px = next_obs[0]  # x position relative to target pad
    py = next_obs[1]  # y position relative to target pad
    vx = next_obs[2]  # horizontal velocity
    vy = next_obs[3]  # vertical velocity
    angle = next_obs[4]  # body angle
    ang_vel = next_obs[5]  # angular velocity

    # Distance to target pad center
    distance = (px**2 + py**2)**0.5
    # Speed magnitude
    speed = (vx**2 + vy**2)**0.5

    # 1. Main learning signal: continuous distance-based reward
    #    Encourages moving towards the target pad
    distance_reward = -distance

    # 2. Stability constraint: light penalty on high speed, large angle, high angular velocity
    #    Helps agent learn to slow down and keep stable attitude near target
    stability_penalty = -0.1 * speed - 0.05 * abs(angle) - 0.05 * abs(ang_vel)

    # 3. Soft approaching proxy: reward getting close and slow simultaneously
    #    Acts as a smoothed "landing" surrogate without contact signals.
    #    Gaussian-like weighting concentrates reward near zero distance and zero speed.
    sigma_dist = 0.2
    sigma_speed = 0.3
    nearness = 2.718281828 ** (-(distance**2) / (2.0 * sigma_dist**2))
    slowness = 2.718281828 ** (-(speed**2) / (2.0 * sigma_speed**2))
    soft_landing_reward = 1.0 * nearness * slowness

    # Combine components
    total_reward = distance_reward + stability_penalty + soft_landing_reward

    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'soft_landing_reward': soft_landing_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件与角色

1. **`distance_reward` – 主学习信号（密集引导）**  
   负的欧氏距离至目标垫中心 `(0, 0)`，每一步都提供梯度，直接推动智能体靠近目标。  
   作用：核心导航驱动力，保证在所有状态下都有向目标移动的压力。

2. **`stability_penalty` – 轻量稳定约束**  
   惩罚当前步的速度大小、身体倾斜角和角速度。三个项的系数均较小，避免压制主信号。  
   作用：引导智能体在接近目标时自然减速并保持竖直姿态，抑制高速冲过目标或剧烈旋转。

3. **`soft_landing_reward` – 任务完成近似信号（proxy）**  
   使用乘以形式的连续高斯权重（无二值条件）：  
   - `nearness`：距离越小，奖励越高（sigma=0.2 使奖励集中在目标附近）。  
   - `slowness`：速度越小，奖励越高（sigma=0.3 促使低速状态）。  
   两个因子相乘，只有在**既靠近又慢**时奖励才显著，然后逐步增强直至稳定在中心附近。  
   作用：在不具备显式成功标志的情况下，提供一个稠密的“完成度”估值，帮助智能体发现最佳末端状态（零距离、零速度）。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确声明 `explicit_success_flag_available: false` 且 `info` 为空字典。环境终止信号中“身体静止”既可来自安全着陆也可来自卡死，无法区分成功与失败。  
- 强行使用会诱导幻觉性的 `info['success']` 或不可靠的终止解析，违反信号可用性原则，因此 v1 完全不依赖终端奖励。

## 哪些组件留到后续迭代

- **能量/引擎惩罚 (`energy_penalty`)**：待智能体学会稳定着陆后再加入小权重燃料消耗惩罚，优化推力使用效率。当前阶段加入可能导致不敢移动。
- **门控式安全约束 (`gated_reward`)**：如剧烈碰撞、出界等不安全状态需明确检测条件，且与终止信号联动，留到后续确认安全边界后设计。
- **终态成功/失败奖励**：当 wrapper 能明确传递 `success` 或 `failure` 标记时，可追加稀疏终端事件，与现有密集信号配合。

## 训练后应观察的 failure mode

- **目标附近震荡**：若距离奖励的线性梯度和 soft_landing 的局部提升导致智能体在中心附近小幅来回，可能需要微调 `stability_penalty` 的系数或引入小的速度变化惩罚。
- **未完全减速上的高奖励**：如果智能体高速穿过中心且距离短暂归零时仍能获得 soft 奖励，检查 sigma 尺度是否过宽；必要时可加略微收紧 sigma。
- **角度持续偏转**：如果智能体学会用倾斜姿态维持位置但 angle 惩罚不足，可酌情提高角度惩罚系数。
- **悬停不降落**：虽然 distance 项会推动垂直分量归零，但若浮空高度维持一定状态，需观察是否需要更明确的垂直奖励；目前设计已通过 dy 纳入 distance，否则后续可单独对 `abs(py)` 加小惩罚。
