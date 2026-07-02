## 诊断

### 1. Agent 发生了什么？
Episode 全长 1000 步，无 crash，无 early termination。得分 142.68，比上轮仅 +1.1，实质上停滞。Agent 没有在 crash，也没有在徘徊——它在**exploit**：一旦到达目标附近并双足着地，就静止不动，每步收割 0.5 的 `soft_landing_proxy`。

### 2. 哪个组件是主要原因？
**`soft_landing_proxy`**。nonzero_rate = 44.1%，意味着 agent 有将近一半的步数都在触发这个奖励。ratio_to_progress = 4.81，是 progress 信号的近 5 倍。`total_reward` 均值 0.266 中有 0.221 来自它（83%）。这就是经典的 **"静坐收割"漏洞**——奖励的是"处于好状态"而非"完成好着陆"。

上轮我把 progress 放大了 10 倍（0.004→0.046），这步做对了，progress 信号健康了。但因为 soft_landing_proxy 仍占主导，总分几乎没有改善。

### 3. 我之前改了什么？
上轮改了 progress_reward 的 scale（1→10），效果符合预期（progress 信号增强 11 倍），但整体得分没上去。**本轮不应该再动 progress 或 stability，应聚焦 soft_landing_proxy 的数学形态。**

### 修复方案

**层次 2：改数学形式。** 将 `soft_landing_proxy` 从"每步状态奖励"改为"一次性触地事件奖励"。通过比较 `obs` 和 `next_obs` 的腿接触状态来检测"着陆瞬间"：双足从「未接触」变为「接触」的那一步。这从根本上消除了"坐着不动收割奖励"的可能——着陆事件只发生一次（或弹跳时寥寥几次）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：进度奖励（保持上轮的 scale=10，不做改动）
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = (dist - next_dist) * 10.0

    # 2. 稳定性惩罚（保持不变）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy：从"每步状态奖励"改为"一次性触地事件奖励"
    #    诊断：上轮 nonzero_rate=44%，agent 着陆后静坐收割 0.5/步，ratio=4.81 主导总奖励。
    #    修复：通过比较 obs vs next_obs 的腿接触状态，检测「着陆瞬间」——
    #    双足从未接触变为接触的那一步才给奖励，消除静坐 exploit。
    prev_both_contact = (obs[6] > 0.5 and obs[7] > 0.5)
    curr_both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    just_landed = curr_both_contact and not prev_both_contact

    # 着陆质量条件（阈值适当放宽，因为是事件触发，不用担心每步 exploit）
    near_target = (next_dist < 0.3)
    low_speed = (abs(next_obs[2]) + abs(next_obs[3]) < 0.5)
    stable_angle = (abs(next_obs[4]) < 0.2)

    soft_landing_proxy = 1.0 if (just_landed and near_target and low_speed and stable_angle) else 0.0

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```