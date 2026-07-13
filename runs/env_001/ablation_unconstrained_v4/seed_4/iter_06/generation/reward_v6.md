这次得分暴跌的直接原因很明确：`vel_penalty`的hinge惩罚全episode active rate高达83%，累计幅度（-69.49）严重压倒正向progress和contact奖励。agent发现“活得越久被罚得越狠”，因此学会了快速撞毁来提前终止惩罚，episode长度从460锐减到68，20场全部提前crash。这不是“着陆太快”，而是惩罚驱动的自杀策略。纯粹增大惩罚系数只会加剧这个问题。因此，我们不再使用独立的、无界的速度/角度惩罚，改为**把速度和角度做成软健康门控**，乘到主进度奖励上：偏差越大，progress越被抑制，从而引导agent主动保持低速、直立状态来获取正向进度。着陆瞬间依然保contact奖励带quality因子，以强化“刚好接触时状态良好”的目标。基于best骨架（progress + contact reward），保留这两个组件，但移除了独立惩罚项。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- hyperparameters ----------
    w_prox = 10.0               # progress weight
    w_contact_base = 2.0        # constant contact bonus
    w_contact_quality = 30.0    # quality multiplier for soft landing
    v_max = 1.0                 # max safe speed for progress gate
    a_max = 0.5                 # max safe angle (rad) for progress gate
    v_soft = 0.5                # target speed for contact quality
    a_soft = 0.25               # target angle for contact quality

    # ---------- 1. distance progress (gate modulation) ----------
    prev_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_dist - next_dist

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle = abs(next_obs[4])

    # health gates for progress: low speed & upright attitude are required to earn progress
    gate_speed = max(0.0, 1.0 - speed / v_max)
    gate_angle = max(0.0, 1.0 - angle / a_max)
    effective_progress = w_prox * progress * gate_speed * gate_angle

    # ---------- 2. contact reward (base + quality) ----------
    contact = next_obs[6] * next_obs[7]   # 1.0 only when both legs touch
    quality_speed = max(0.0, 1.0 - speed / v_soft)
    quality_angle = max(0.0, 1.0 - angle / a_soft)
    quality = (quality_speed * quality_angle) ** 0.5
    contact_reward = w_contact_base * contact + w_contact_quality * contact * quality

    # ---------- total (no independent penalties) ----------
    total_reward = effective_progress + contact_reward

    components = {
        'progress': effective_progress,
        'contact_reward': contact_reward,
    }
    return float(total_reward), components
```