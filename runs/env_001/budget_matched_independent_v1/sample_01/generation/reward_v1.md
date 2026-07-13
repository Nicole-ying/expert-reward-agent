# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测
    x_old = float(obs[0])
    y_old = float(obs[1])
    # 下一观测
    x_new = float(next_obs[0])
    y_new = float(next_obs[1])
    vx = float(next_obs[2])
    vy = float(next_obs[3])
    angle = float(next_obs[4])
    ang_vel = float(next_obs[5])

    # ---------- 1. 位置进步奖励（主学习信号）----------
    dist_old = (x_old ** 2 + y_old ** 2) ** 0.5
    dist_new = (x_new ** 2 + y_new ** 2) ** 0.5
    progress = dist_old - dist_new   # 接近目标为正

    # ---------- 2. 稳定性软门控（乘在主奖励上）----------
    speed = (vx ** 2 + vy ** 2) ** 0.5
    abs_angle = abs(angle)

    k_speed = 0.5
    k_angle = 1.0
    gate_speed = 1.0 / (1.0 + k_speed * speed)
    gate_angle = 1.0 / (1.0 + k_angle * abs_angle)
    gate = gate_speed * gate_angle

    # ---------- 3. 辅助约束 ----------
    w_angle = 0.1          # 角度平方惩罚权重
    angle_penalty = -w_angle * (angle ** 2)

    w_angvel = 0.01        # 角速度平方惩罚权重
    angular_vel_penalty = -w_angvel * (ang_vel ** 2)

    # ---------- 合成总奖励 ----------
    w_progress = 1.0
    gated_progress = w_progress * progress * gate
    total_reward = gated_progress + angle_penalty + angular_vel_penalty

    components = {
        "gated_progress": float(gated_progress),
        "angle_penalty": float(angle_penalty),
        "angular_vel_penalty": float(angular_vel_penalty)
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 1. 任务画像与动力学子类型
- **selected task_family**: `navigation_goal_reaching`
- **dynamics_subtype**: `goal_approach_and_soft_contact`
- **解读**：飞行器需要飞到中央平台正上方，同时完成减速、摆正姿态并实现双腿接触，最终稳定着陆。这既是位置导航，又对速度、姿态和接触有着严格的软着陆要求。

## 2. 选择的奖励职责（reward roles）
根据 `environment_card.md` 的强制 / 按需职责，v1 阶段选择以下角色：

| 职责角色 | 类型 | 理由 |
|----------|------|------|
| **main_learning (progress)** | 主学习信号 | 告诉 agent “向目标靠近”是核心得分行为，每步都有梯度。 |
| **soft_health_gate (stability)** | 稳定/安全约束 | 防止 agent “冲过去但刹不住”。当速度或倾斜过大时，衰减主进步奖励，迫使 agent 在靠近目标的同时降低速度、竖直姿态。 |
| **angle_penalty** | 稳定/安全约束 | 对躯干倾角给予二次惩罚，引导姿态保持竖直。 |
| **angular_vel_penalty** | 稳定/安全约束 | 抑制剧烈旋转，提高着陆平稳性。 |

## 3. 角色-信号-公式映射
- **progress** → `obs[0:2]` 与 `next_obs[0:2]` 的 **2D 距离差** `dist_old - dist_new`  
  - 公式类别：`improvement_delta`，天然表达“靠近即得分”。
- **stability gate** → `next_obs[2:4]` 的合速度 `speed` 和 `next_obs[4]` 的 `angle`  
  - 公式类别：`soft_health_gate`（乘积型倒数门）  
  - 具体形式：`gate = 1/(1+0.5*speed) * 1/(1+1.0*|angle|)`，在速度 ~2.0 或角度 ~0.5 rad 时开始明显衰减主奖励。
- **angle_penalty** → `next_obs[4]` 的 `angle`  
  - 公式类别：`quadratic_penalty`，轻量级抑制。
- **angular_vel_penalty** → `next_obs[5]` 的 `ang_vel`  
  - 公式类别：`quadratic_penalty`，进一步稳定旋转。

## 4. 排除的职责及原因
- **explicit terminal success / failure rewards**：`info` 为空，且无 `success`/`failure` 标志，**不能使用**。
- **contact‑based bonus**：`left_contact`/`right_contact` 虽是可用信号，但无法区分“接触平台”还是“摔在地面/障碍物”。v1 选择暂时搁置，避免奖励接触地面导致利用风险。着陆完成信号留到后续迭代（例如 v2 引入带距离‑速度门控的接触奖励）。
- **fuel / action penalty**：动作 2（主发动机）能耗高，但 v1 优先让 agent 学会安全着陆，能耗优化延后到 v2/v3。
- **curriculum / dynamic weighting**：v1 无可靠训练进度参考，且早期探索与晚期精细控制尚未出现明显冲突，暂不加。

## 5. 关键设计决定
- 主奖励 **不是** 简单的“越近越好”，而是 **progress × stability gate**。  
  - 如果飞行器高速冲向目标但姿态倾斜，`gate` 会大幅衰减 `progress`，反映“虽然靠近了但没减速，不算有效进步”。  
  - 这种结构比独立的速度惩罚更精准：惩罚不会在安全状态下无差别施加，而是在健康恶化时切断主信号，迫使 agent 在接近终点前主动减速、摆正。
- 角度与角速度的轻量惩罚作为辅助，避免 agent 长时间大幅倾斜或旋转。
- 所有组件均为每步连续信号，不存在稀疏二值反馈，保证梯度密度。

## 6. 预期需要观察的 failure modes
- **过早休眠**：agent 可能在尚未稳定时被休眠条件判定终止，需观察训练曲线中 episode 长度是否过早缩短。
- **接触但非平台**：虽然接触奖励未引入，但 agent 可能学会用侧边或姿态碰撞地面获得某种“减速”效果，需检查轨迹中 `y_position` 和 `left/right_contact` 的组合。
- **悬停不接触**：agent 可能学会停留在平台上方但双腿不接触，仅靠 gate 和惩罚维持稳定，此时应关注 `contact=0` 但 `progress≈0`、姿态良好的情况——这将在 v2 加入接触奖励解决。
- **速度/姿态抖动**：如果权重配比不当，角速度惩罚可能诱导过于保守的动作，抑制必要的姿态调整。

## 7. 后续迭代预留
- **v2**：引入联合条件 proxy（距离、速度、角度、双腿接触），作为软完成信号，鼓励 agent 实际落地。
- **v3**：若燃料优化成为瓶颈，加入推力度量 `action ≠ 0` 的轻量代价；同时可考虑软着陆后的存活奖励。
- 若训练后期仍出现“高速撞平台”或“超出视口”频繁，可适度增大 gate 系数或增加 `progress` 的凸化形式（`progress**2`）提升动力学区分度。