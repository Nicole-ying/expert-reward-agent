# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 配置参数 ---
    # 主学习信号权重
    w_progress = 1.0

    # 稳定性惩罚权重
    w_speed = 0.01
    w_angle = 0.01
    w_angvel = 0.001

    # 软着陆代理奖励权重及其阈值
    w_proxy = 0.2
    dist_threshold = 0.2      # 距离目标平台足够近
    speed_threshold = 0.1     # 速度足够低
    angle_threshold = 0.2     # 姿态接近水平
    # 接触标志：0.0 未接触，1.0 接触

    # --- 1. 进度差分奖励 (主学习信号) ---
    # 当前位置到目标 (0,0) 的距离
    d_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    # 下一位置到目标的距离
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_current - d_next

    # --- 2. 稳定性惩罚 (稳定约束) ---
    # 线速度大小
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    # 机体倾角绝对值
    angle_abs = abs(next_obs[4])
    # 角速度绝对值
    angvel_abs = abs(next_obs[5])

    stability_penalty = - (w_speed * speed + w_angle * angle_abs + w_angvel * angvel_abs)

    # --- 3. 软着陆代理奖励 (任务完成近似) ---
    # 条件：靠近目标、低速、姿态稳定、双脚均已接触
    near_target = d_next < dist_threshold
    low_speed = speed < speed_threshold
    stable_angle = angle_abs < angle_threshold
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    soft_proxy = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_proxy = w_proxy * 1.0

    # --- 总奖励 ---
    total_reward = progress + stability_penalty + soft_proxy

    # 组件字典
    components = {
        'progress': progress,
        'stability_penalty': stability_penalty,
        'soft_proxy': soft_proxy,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward（主学习信号）**  
   - 角色：引导智能体每一步向目标移动。  
   - 实现：`d_current - d_next`，其中`d`为当前位置到目标平台中心的欧氏距离。  
   - 权重：`1.0`（主导地位）。

2. **stability_penalty（稳定/安全约束）**  
   - 角色：抑制过高的线速度、机体倾角和角速度，为稳定接近与精确着陆提供压力。  
   - 实现：对线速度幅值、机体角绝对值和角速度绝对值进行线性惩罚，并乘以较小的权重。  
   - 权重：`w_speed = 0.01, w_angle = 0.01, w_angvel = 0.001`，轻量级约束，避免阻碍探索。

3. **soft_landing_proxy（任务完成近似信号）**  
   - 角色：在智能体非常接近目标、速度足够低、姿态稳定且双脚均接触时给予一个小额奖励，提供“着陆成功”的模糊引导。  
   - 实现：满足四个条件时给予固定正奖励 `w_proxy * 1.0`，否则为 0。  
   - 权重：`w_proxy = 0.2`，远小于主信号，避免被 exploit。

## 为何没有使用 terminal_success_reward / terminal_failure_penalty

环境卡片明确指出 `explicit_success_flag_available = false` 且 `explicit_failure_flag_available = false`，`info` 字典为空。任何依赖显式成功/失败标志的奖励组件都会引入幻觉信号（例如访问不存在的 `info['success']`），因此 v1 完全不使用。

## 后续迭代预留的组件

- **energy_penalty（动作/能耗约束）**：当前 v1 故意不惩罚引擎使用，避免“智能体不敢动”（agent afraid to move）。待 agent 能稳定接近目标后再加入以优化燃料消耗。  
- **time_penalty（效率惩罚）**：过早加入可能导致智能体选择快速失败或冒险行为，将在后续迭代中引入。  
- **gated_reward（安全门控）**：当需要区分“安全接近”与“危险受控飞行”时再添加。  
- **terminal_success_reward / terminal_failure_penalty**：只有训练 wrapper 在未来提供明确的 `success`/`failure` 标志后才启用。

## 训练后应观察的典型 failure modes

1. **目标附近震荡（goal_near_oscillation）**  
   - 主信号 `progress` 在接近目标时趋于零，可能不足以迫使精确稳定，导致智能体在平台上方来回摆动。  
   - 对策：后续可降低 `progress` 阈值、增强 `stability_penalty` 或加入抵近阶段的特定约束。

2. **高 reward 但未成功停靠（high_reward_without_success）**  
   - 智能体可能学会触发 `soft_proxy` 条件（短暂接触）却未真正稳定停留，甚至利用接触后即离开的漏洞。  
   - 对策：收紧 `soft_proxy` 的联合条件或改为连续奖励形式。

3. **高速撞击目标（fast_crash_near_goal）**  
   - 如果 `progress` 主信号过强而稳定性惩罚不足，智能体可能以高速冲向平台并导致碰撞。  
   - 对策：适当提高 `w_speed` 或添加接近目标时的速度衰减奖励。

4. **姿态失稳与角速度发散**  
   - 若 `w_angle` 或 `w_angvel` 过小，机体可能旋转过快，影响着陆成功率。  
   - 对策：观察角速度统计，必要时增大相应惩罚系数。

5. **早期不敢移动**  
   - 如果稳定性惩罚意外抑制了所有发动机使用，agent 可能静止不动。  
   - 对策：检查 `stability_penalty` 的尺度，确保 `progress` 远大于稳定性代价。
