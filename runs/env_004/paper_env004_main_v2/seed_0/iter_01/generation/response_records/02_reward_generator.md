# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取关键信号
    torso_angle = next_obs[1]          # 躯干倾角
    forward_vel = next_obs[5]          # 前进速度（水平速度）
    torso_ang_vel = next_obs[7]        # 躯干角速度

    # 姿态因子：躯干越直立（角度接近0），因子越接近1
    k_angle = 10.0
    posture_factor = 2.718281828 ** (-k_angle * (torso_angle ** 2))

    # 主学习信号：前进速度 × 姿态因子
    #   - 前进速度为正则奖励，为负则惩罚，引导 agent 持续向前
    #   - 姿态因子在躯干倾斜时自动衰减，抑制为追求速度而过度倾斜
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：惩罚躯干角速度，抑制快速晃动/摔倒
    w_ang_vel = 0.05
    ang_vel_penalty = w_ang_vel * abs(torso_ang_vel)

    total_reward = speed_posture_reward - ang_vel_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 组件及其角色
- **speed_posture_reward（主学习信号）**：  
  形式为 `forward_vel * exp(-k * angle²)`。  
  - 它直接关联任务目标（持续向前移动），同时通过姿态因子对躯干倾角施加衰减，让 agent 学习“在保持直立的前提下加速前进”。  
  - 该信号每步都有连续梯度，且不依赖任何终止标志或 info 字段。
- **ang_vel_penalty（稳定约束）**：  
  对躯干角速度施加 L1 小惩罚。  
  - 角色是“方向盘”，抑制不必要的快速角运动，降低摔倒概率；权重很小，不会压制前进驱动力。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- 环境卡片明确声明 `explicit_success_flag_available = false` 和 `explicit_failure_flag_available = false`，且 `info` 字典为空；无法获得任何显式成功/失败标志，因此不能使用对应的终端奖励或惩罚。

## 为什么没有使用 alive_bonus / done 信号
- 函数签名没有 `done` 输入，也无法从 `obs` / `next_obs` / `info` 可靠推断是否已终止，因此存活奖励无法实现。

## 后续迭代可加入的组件
- **能量/动作代价**（如扭矩平方和惩罚）：待 agent 能稳定前跳后再引入，以避免早期“怕动”。
- **动作平滑惩罚**：需要 `previous_action` 输入，当前接口尚不支持。
- **高度维持约束**：若观察到 agent 频繁因高度过低终止，可加入软高度奖励或惩罚，但需先获取健康高度阈值或通过实验标定。

## 预期需观察的 failure mode
- **原地直立而不前进**：`forward_vel ≈ 0` 导致总奖励接近 0；agent 可能满足于此。需通过增大 `speed_posture_reward` 权重或引入前向进度奖励（如对速度的更强推动）来解决。
- **快跑但频繁摔倒**：若 `ang_vel_penalty` 过小，agent 可能为获得高前进速度而过度倾斜，导致终止。此时应适当增大 `w_ang_vel` 或 `k_angle`。
- **低效步态（如爬行而非跳跃）**：当前奖励仅驱动水平速度，未显式要求跳跃。若出现不跳跃的步态，后续可考虑引入垂直速度奖励或接触信号（若环境提供）来触发周期性跳跃。
