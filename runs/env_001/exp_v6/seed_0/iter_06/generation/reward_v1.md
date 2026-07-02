## 诊断

**验证失败根因分析：**

从 Memory 可以清晰看到问题：

| iter | 方案 | score | soft_landing_proxy mean |
|------|------|-------|------------------------|
| 3 | 二值型 proxy | 142.68 | 0.221 |
| **4 (v1)** | **连续乘积型 proxy** | **66.52** | **0.077** |
| 5 | 二值型 proxy | **146.13** | 0.206 |

v1 把二值型 `soft_landing_proxy` 改成了四个因子相乘的连续型（`near_factor × speed_factor × angle_factor × contact_factor × 0.5`）。问题是：**四个因子相乘导致信号极度萎缩**——contact_factor（双脚乘积）在大多数 step 接近 0，把整个 proxy 压到几乎没有梯度。结果 soft_landing_proxy 均值从 0.22 暴跌到 0.08，得分从 142 暴跌到 66。

连续乘积型在这里失败的原因：LunarLander 环境中双脚同时接触地面是稀疏事件，乘积形式让 agent 在整个飞行过程中几乎收不到着陆相关的梯度引导，相当于丢失了 65% 的有效学习信号。

**修复方案：** 回退到 iter 5 的历史最佳二值型架构（score=146.13），不做任何改动。该版本已被验证有效。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist  # 每一步靠近目标获得正奖励

    # 2. 稳定性惩罚（弱背景信号，ratio ≈ -0.014，不干扰主信号）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.001 * (vel_x + vel_y) - 0.001 * angle - 0.001 * ang_vel

    # 3. 软着陆 proxy（二值事件型，iter 5 验证得分 146.13）
    near_target = (next_dist < 0.1)
    low_speed = (vel_x + vel_y < 0.2)
    stable_angle = (angle < 0.1)
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_proxy = 0.5 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```