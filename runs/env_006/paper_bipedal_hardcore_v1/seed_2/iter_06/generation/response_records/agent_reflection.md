# Response Record

# 设计理由

## 审计发现 (signal audit)
- **终止模式**: 全部 20 个 episode 均以 terminated (非 truncated) 结束，其中 3 个 early terminal（<150 step, score<-50）提示部分案例出现突然摔倒；其余 episode 步长中等（~250），可能在到达地形尽头或稳定摔倒前中止。  
- **观测使用扫描**: 当前 reward 仅使用了 `obs[0]`（hull_angle）和 `obs[2]`（horizontal_speed）。关键未使用维度 `obs[1]`（hull_angular_velocity）携带着身体是否即将失控的动态信息。lidar（`obs[14:23]`）虽然可用，但历史迭代 3/4 表明直接利用易引入噪声，暂保持不直接奖励。腿接触 (`obs[12,13]`) 也未使用。  
- **僵尸组件**: `hinge_penalty` 仅依赖静态角度超出阈值 (0.6 rad)，active_rate = 1.3%，几乎不触发，说明摔倒很少通过缓慢倾角超限实现，而是突然失控。该组件成为无效负载。

**结论**: 当前奖励信号不全——缺失对姿态动态（角速度）的惩罚或门控机制，导致 agent 在剧烈摆动或即将摔倒时未受到区分性反馈，这是阻碍稳定性和分数提升的主因之一。

## 行为诊断
- **agent 当前行为**: 保持中等倾角，以 ~0.295 per‑step 的 forward reward 稳步前进，但遇到地形变化或自身失衡时常常瞬间摔倒，导致 episode 过早终止，并且整段路程无法弥补原始环境负奖励（orig_reward ≈ -0.5/step），使总分仍为负。
- **干预目标**: 提升稳定性，让 agent 能够识别并抑制危险的身体摇晃/即将摔倒的状态，延长存活并维持较高的前进效率。
- **方向评估**: 当前骨架（forward_reward * gate + 轻量 energy_penalty）在上轮迭代中分数改善明显（-74.9 → -52.5），累积预判 ✅，表明继续在此骨架上深化是值得的，不需重建。

## 干预层级：Level 2 — 结构变换
**变换内容**: 将原有单因子姿态门控 `gate(angle)` 升级为 `gate(angle, angular_velocity)`，同时移除无效的 `hinge_penalty`。这使得 reward 能直接对“即将摔倒”的动态尖峰做出反应，而非仅在静态偏角过大时施加惩罚。

- **数学形式**: `gate = 1 / (1 + β_angle * |hull_angle| + β_angvel * |hull_angular_velocity|)`  
  选定参数：β_angle=2.0（维持原敏感度），β_angvel=1.0（对中等角速度开始压制）。
- **校准检查**:  
  - 在“安全但存在轻微摇晃”区域（angle=0.2 rad, ang_vel=0.5 rad/s）→ gate=0.526 ≥ 0.3，不会塌缩。  
  - 惩罚负担：原 hinge_penalty per‑step ≈0，新 gate 仅在摇晃时降低前进奖励，相当于“缺省前进奖励”，不会引入额外恒定惩罚，总惩罚负担保持极低（仅有 energy_penalty），远低于主信号 per‑step 的 0.3 倍。
  - 移除 hinge_penalty 避免僵尸组件残留。

**设计目标**: 让 agent 只有在姿态平稳（角度小 *且* 角速度低）时才能获得高前进奖励，从而学会抑制多余的躯干振荡，显著降低忽然摔倒率，延长有效行走距离。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- 基本观测提取 ----------
    hull_angle = obs[0]                 # 身体倾角 (rad)
    hull_angular_velocity = obs[1]      # 身体角速度 (rad/s)
    horizontal_speed = obs[2]           # 质心水平速度

    # 只奖励正向移动 (假设正速度朝前)
    forward_vel = max(0.0, horizontal_speed)

    # ---------- 1. 主学习信号：动态 gate‑controlled forward progress ----------
    # gate 由角度和角速度共同决定，剧烈摇晃 → gate 变小 → 前进奖励骤减
    beta_angle = 2.0   # 原有角度敏感度
    beta_angvel = 1.0  # 新增角速度敏感度
    gate = 1.0 / (1.0 + beta_angle * abs(hull_angle) + beta_angvel * abs(hull_angular_velocity))
    forward_reward = 1.0 * forward_vel * gate

    # ---------- 2. 效率约束：轻量力矩代价（保持不变） ----------
    action_cost = sum(action[i] ** 2 for i in range(4))
    energy_penalty = 0.01 * action_cost

    # ---------- 汇总（移除原有 hinge_penalty） ----------
    total_reward = forward_reward - energy_penalty

    components = {
        "forward_reward": forward_reward,   # 双因子 gate 调制后的前进奖励
        "energy_penalty": -energy_penalty   # 力矩代价 (记录为负值)
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 姿态角速度信息缺失，静态阈值 hinge 成为僵尸；导致 agent 对动态失控缺乏敏感反馈。
- **behavior**: agent 以中等步速前进，但碰到地形或失稳时突然摔倒，缺少预防性信号，总分仍为负。
- **signal**: 缺 hull_angular_velocity 对即将摔倒的预判；现有静态角度惩罚几乎不激活（1.3%）。
- **level**: Level 2（结构变换：gate 纳入角速度，删除僵尸 penalty）
- **hypothesis**: 让前进奖励的 gate 在身体摇晃时快速衰减， agent 将学会抑制危险振荡，摔倒率下降，有效存活步数和综合得分显著改善。
- **risk**: 角速度惩罚过强可能使 agent 过度减速，导致前进停滞；需观察 len 变化，若 len 回弹且 score 不增，可后续微调 β_angvel 降低至 0.5。
