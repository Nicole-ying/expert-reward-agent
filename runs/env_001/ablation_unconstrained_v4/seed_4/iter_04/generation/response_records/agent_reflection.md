# Response Record

## 分析

1. **agent行为**：大多数 episode（估计约 15-18/20）以失败终止并累积大量负 progress，拉低平均分；少数成功 episode 则依靠大量 contact reward 得到极高正分（+200 附近）。progress 每步都有但受 gate 压制，总体量级很小；contact 只在双腿着地时激活（active 7.3%），稀疏但量级极大。整体处于“偶尔成功、多数失败”的状态，缺乏足够强的中间引导信号。
2. **最值得干预的组件**：gate 结构。gate 同时作用在 progress 和 contact 上，导致 progress 在高速/大倾斜时的奖励过小，削弱了前进引导；而在接触时又限制 contact 奖励，迫使 agent 必须在高速/大倾斜下也尝试着陆。去除乘性 gate，改为独立的速度/角度 hinge 惩罚，既可恢复 progress 的引导强度，又能通过惩罚迫使 agent 主动减速和保持直立。
3. **上一轮动作**：上一轮从纯惩罚型设计（iter1/iter2）转向 contact+progress+gate，得分从 -110 大幅提升到 -0.24，表明接触奖励有效，但 gate 阻碍了进一步收敛。本轮在 best 基础上移除 gate，将速度/角度约束改为独立 hinge 惩罚，维持 progress 全时有效，并保留接触奖励。

## 修改方案
- 移除 gate，不再将速度/角度乘到 reward 上。
- 新增速度惩罚：`k_vel_pen * max(0, speed - v_threshold)`，用阈值 1.2 容忍低速行为。
- 新增角度惩罚：`k_ang_pen * max(0, abs(angle) - a_threshold)`，阈值 0.3 rad。
- 保持 progress 线性奖励和 contact 稀疏奖励，系数重新平衡（w_prox=12, w_contact=20）。
- 不添加燃料惩罚或其它未经验证的成分，保持最小改动。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox = 12.0           # distance progress weight
    w_contact = 20.0        # contact bonus weight
    k_vel_pen = 0.8         # velocity penalty coefficient
    k_ang_pen = 1.0         # angle penalty coefficient
    v_threshold = 1.2       # safe speed threshold
    a_threshold = 0.3       # safe upright angle threshold (rad)

    # ---------- 1. distance progress ----------
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_dist - next_dist
    progress_reward = w_prox * progress

    # ---------- 2. contact bonus ----------
    contact = next_obs[6] * next_obs[7]   # 1.0 only when both legs touch
    contact_reward = w_contact * contact

    # ---------- 3. speed and angle penalties (hinge) ----------
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])
    vel_penalty = k_vel_pen * max(0.0, speed - v_threshold)
    ang_penalty = k_ang_pen * max(0.0, angle - a_threshold)

    # ---------- total ----------
    total_reward = progress_reward + contact_reward - vel_penalty - ang_penalty

    components = {
        'progress': progress_reward,
        'contact_reward': contact_reward,
        'vel_penalty': vel_penalty,
        'ang_penalty': ang_penalty,
    }
    return float(total_reward), components
```
