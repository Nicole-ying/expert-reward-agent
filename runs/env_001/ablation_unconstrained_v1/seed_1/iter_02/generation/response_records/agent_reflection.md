# Response Record

## 分析

### 1. Agent 发生了什么？

**行为证据：**
- 所有 20 个 episode 均提前终止（terminated=20/20），平均长度仅 68 步
- 得分范围 [-125.7, -90.1]，全部严重负分
- **stability_penalty** 每回合 -4.71，占 signed_share 的 -53.1%，100% 激活 —— 是压倒性的负信号
- **progress_reward** 每回合仅 +1.12（13.1% magnitude），被 penalty 淹没（比值 |−4.71 / 1.12| = 4.2，远超 0.5 警戒线）
- **landing_bonus** active_rate 仅 0.4%，每回合均值 +3.0 来自极少触发的稀疏事件，agent 从未真正学会着陆

**根本原因：** stability_penalty 在每个 step 惩罚速度、角度和角速度，而这些是飞行过程中不可避免的。penalty 主宰了学习信号，agent 为避免惩罚而快速坠毁（或不做动作任其坠毁），无法存活到实现 progress 或触发 landing_bonus。landing_bonus 的硬性二值条件（双腿接触 + 近中心 + 低速 + 直立）对于从未成功到达平台的 agent 来说不可达。

### 2. 最值得干预的组件

**stability_penalty** 和 **landing_bonus** 需要同时改造：
- stability_penalty：系数降低 10 倍以上，并加距离门控（只在接近目标时生效），让 agent 在远处能自由移动
- landing_bonus：从硬性二值改为连续乘积形式的 soft proxy，提供可学习的梯度

### 3. 上一轮修改回顾

上一轮代码即当前版本（首次诊断，无历史对比）。三个组件均未奏效：penalty 太重、progress 太弱、bonus 太稀疏。

### 修改方案

1. **progress** 保留 `dist_obs - dist_next`，不缩放（它是真实的距离改善）
2. **stability_penalty** → 距离门控 + 系数降低 10 倍：`gate = max(0, 1 - dist/2.0)`，系数从 0.05/0.1/0.01 降至 0.005/0.01/0.001
3. **landing_bonus** → 连续 soft product：`both_legs × x_ok × y_ok × vel_ok × angle_ok`，每步最高 3.0，通过乘积提供平滑梯度
4. 新增 **leg_contact_reward**：双腿接触 + 靠近目标即给分，降低初始探索门槛

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 进度信号：向目标 (0,0) 靠近 ---
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = dist_obs - dist_next

    # 当前到目标的距离
    dist = dist_next

    # --- 距离门控：靠近目标时才关心稳定性 ---
    stability_gate = max(0.0, 1.0 - dist / 2.0)

    # 稳定性代价（系数降低约 10 倍）
    vel_sum = abs(next_obs[2]) + abs(next_obs[3])
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])
    stability_cost = 0.005 * vel_sum + 0.01 * angle_abs + 0.001 * angvel_abs
    stability_penalty = -stability_gate * stability_cost

    # --- 连续着陆奖励：soft product 替代硬性二值 ---
    both_legs = next_obs[6] * next_obs[7]              # 0 或 1，仅双腿同时触地时为 1
    x_ok = max(0.0, 1.0 - abs(next_obs[0]) / 0.3)      # |x| < 0.3 满分
    y_ok = max(0.0, 1.0 - abs(next_obs[1]) / 0.3)      # |y| < 0.3 满分
    vel_ok = max(0.0, 1.0 - vel_sum / 0.5)              # |vx|+|vy| < 0.5 满分
    angle_ok = max(0.0, 1.0 - angle_abs / 0.3)          # |angle| < 0.3 满分

    landing_quality = both_legs * x_ok * y_ok * vel_ok * angle_ok
    landing_reward = 3.0 * landing_quality

    # --- 腿部接触奖励：降低初始探索门槛 ---
    leg_contact_reward = 1.0 * both_legs * max(0.0, 1.0 - dist / 1.5)

    # --- 总奖励 ---
    total_reward = progress + stability_penalty + landing_reward + leg_contact_reward

    components = {
        'progress': progress,
        'stability_penalty': stability_penalty,
        'landing_reward': landing_reward,
        'leg_contact_reward': leg_contact_reward
    }

    return float(total_reward), components
```
