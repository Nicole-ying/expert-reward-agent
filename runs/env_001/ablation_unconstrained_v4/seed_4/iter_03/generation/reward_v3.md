根据训练反馈，agent在所有评估轨迹中均早期崩溃终止（len=68.45, terminated=20/20），奖励分解显示 `progress` 正向贡献（+5.6）被四项独立惩罚（合计约-14）完全淹没，导致总得分 -110.66，较上一轮 best（-101.89）进一步恶化。证据表明，本轮新增的 `hspeed_penalty` 和 `x_center_penalty` 以及较重的惩罚项使 agent 无法接近目标，缺乏成功着陆的激励。因此，干预应聚焦于：①回退新增独立惩罚，②将速度/姿态约束改为乘性 gate 以避免负奖励积累，③引入基于支撑腿接触的着陆奖励，为稳定着垫提供明确正向信号。

以下为修改后的奖励函数：

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox = 10.0      # distance reduction weight
    w_contact = 25.0   # contact bonus weight
    k_vel = 0.5        # velocity gate slope
    k_angle = 2.0      # angle gate slope

    # ---------- 1. distance progress ----------
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = prev_dist - next_dist

    # ---------- 2. multiplicative gate (based on speed and tilt) ----------
    speed = (next_obs[2]**2 + next_obs[3]**2) ** 0.5
    angle = abs(next_obs[4])
    gate = 1.0 / (1.0 + k_vel * speed + k_angle * angle)

    # ---------- 3. contact bonus for stable landing ----------
    # both support legs in contact
    contact = next_obs[6] * next_obs[7]

    # ---------- reward composition ----------
    progress_reward = w_prox * progress * gate
    contact_reward = w_contact * contact * gate

    total_reward = progress_reward + contact_reward

    components = {
        'progress': progress_reward,
        'contact_reward': contact_reward,
    }
    return float(total_reward), components
```