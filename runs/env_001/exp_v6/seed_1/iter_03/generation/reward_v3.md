## 诊断

**1. Agent 状态：** episode_length=857.8，无 crash，得分 218。agent 在正常完成任务，没有徘徊或 exploit。但仔细看组件比例——soft_proxy 均值为 progress 的 **26.7 倍**（0.106 vs 0.004），这意味着主学习信号 progress 被 soft_proxy 完全淹没了。

**2. 根因：** 上一轮我把 soft_proxy 从二值改成连续乘积形式，同时设 w_proxy=0.2。这个系数太大了——连续乘积的 nonzero_rate 高达 61.7%（vs 旧二值形式的 0.38%），导致 soft_proxy 成为事实上的主信号。progress 的梯度方向是对的但强度完全不够。

**3. 我上次做了什么：** Iter 2 同时改了三个东西（降 stability_penalty、改 soft_proxy 数学形式、设 w_proxy=0.2）。得分从 -110 → 218，方向正确。但 soft_proxy 矫枉过正——现在需要把 w_proxy 降下来，让 progress 重新成为主导。

## 行动：层次 1（改系数）

只降 `w_proxy`：从 0.2 → 0.008。预期 soft_proxy 均值从 0.106 降到 ~0.004，与 progress 的 0.004 持平，让两个信号共同驱动学习而非 proxy 一家独大。stability_penalty 已经足够小（ratio 0.088x），不动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 配置参数 ---
    # 主学习信号权重
    w_progress = 1.0

    # 稳定性惩罚 — 保持上一轮的弱水平（ratio 0.088x progress，没问题）
    w_speed = 0.001
    w_angle = 0.001
    w_angvel = 0.0001

    # 软着陆代理 — 连续乘积形式保留，但大幅降权（上一轮 ratio 26.7x progress，压倒主信号）
    w_proxy = 0.008    # 原 0.2 → 降 25x，预期 soft_proxy ~ 0.004，与 progress 持平
    dist_threshold = 0.5
    speed_threshold = 0.3
    angle_threshold = 0.5

    # --- 1. 进度差分奖励 (主学习信号) ---
    d_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    d_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = d_current - d_next

    # --- 2. 稳定性惩罚 ---
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])

    stability_penalty = -(w_speed * speed + w_angle * angle_abs + w_angvel * angvel_abs)

    # --- 3. 软着陆代理 (连续乘积，密集梯度) ---
    near_factor = max(0.0, 1.0 - d_next / dist_threshold)
    speed_factor = max(0.0, 1.0 - speed / speed_threshold)
    angle_factor = max(0.0, 1.0 - angle_abs / angle_threshold)
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0

    soft_proxy = w_proxy * near_factor * speed_factor * angle_factor * contact_factor

    # --- 总奖励 ---
    total_reward = progress + stability_penalty + soft_proxy

    components = {
        'progress': progress,
        'stability_penalty': stability_penalty,
        'soft_proxy': soft_proxy,
        'total_reward': total_reward
    }

    return float(total_reward), components
```