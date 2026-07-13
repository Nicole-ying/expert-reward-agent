## 完整分析

### 1. Evidence

iter_3 得分27.72，17/20 terminated，landing_bonus episode_sum_mean=312.24 且 active_rate=12.8%（约64步持续触发），magnitude_share=90.4%；shaped_progress 几乎全步激活但仅贡献~20.5；ang_vel_penalty 可忽略。上一轮被拒是因为提交了与 iter_3 完全相同的代码。

### 2. Behavior Diagnosis

智能体已学会着陆（17/20成功终止），但 landing_bonus 作为"持续状态奖励"在着陆后每步重复发放，形成持久 farming：智能体停靠后持续领取~5/步的奖励，累计312分。外部得分仅27.72远低于内部奖励量级，说明外部环境有独立的惩罚机制（如燃料消耗），而我们未对齐。

### 3. Signal Completeness

进度信号（shaped_progress）和姿态约束（ang_vel_penalty）职责完备且可达。着陆信号存在但数学形态错误：`both_contact * velocity_gate * angle_gate` 是持续状态值，着陆后每步重复触发，而非奖励"完成着陆"这一事件本身。缺少燃料效率信号但本轮不引入。

### 4. Selected Level

**Level 2：persistent_to_transition_event**。触发条件：landing_bonus 的 active_rate=12.8% 与 magnitude_share=90.4% 的组合明确指向"占据好状态即可持续获奖"模式，且智能体已能稳定着陆，不需要持续奖励来维持接触。

### 5. Selected Intervention

唯一目标组件 `landing_bonus`：从持续状态奖励 `both_contact * gates` 改为转移事件 `max(0, curr_both - prev_both) * gates`。利用 `obs` 和 `next_obs` 的接触状态差检测首次触地瞬间，系数从10.0升至100.0以补偿从多步累积到单次事件的量级变化。不触碰 shaped_progress 和 ang_vel_penalty。

### 6. Falsifiable Hypothesis

将着陆奖励从"处于着陆状态"改为"完成着陆动作"后，智能体无法再通过停在垫上刷分，必须优化接近与触地瞬间的姿态和速度来获取单次大额奖励。这将消除 persistent farming，使内部奖励与外部任务的"成功着陆"事件对齐，预期外部得分提升。

### 7. Expected Next Round

- `landing_bonus` 的 active_rate 应骤降至 ~1-3%（每 episode 仅触发一次），magnitude_share 下降，不再主导总奖励
- `shaped_progress` 的 magnitude_share 上升，成为相对主要的稠密信号
- episode_length 可能略微缩短（着陆后不再有动机停留）
- terminated 比例应保持或改善
- 外部 score 应提升

### 8. Main Risk

智能体可能学会"弹跳"策略：着陆获取 transition bonus → 轻微离地 → 再次着陆再获取 bonus。下一轮若 active_rate 异常升高（远超每 episode 一次）且 episode_length 未缩短，则确认弹跳 exploiting，需引入冷却或高度阈值。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 匿名环境观测假设（经典2D着陆器）：
    # obs[0]: x (水平偏移)    obs[1]: y (高度)
    # obs[2]: vx             obs[3]: vy
    # obs[4]: angle          obs[5]: ang_vel
    # obs[6]: left_contact   obs[7]: right_contact

    # 1. 主学习信号：距离改善 × 姿态门控
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next          # 正值表示靠近目标

    angle_abs = abs(next_obs[4])                     # 机体角度（弧度）
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)        # soft health gate
    shaped_progress = 25.0 * progress_raw * gate

    # 2. 安全约束：角速度轻量惩罚
    ang_vel_penalty = -0.1 * (next_obs[5] ** 2)

    # 3. 着陆奖励：转移事件 —— 仅首次触地瞬间触发
    prev_both = obs[6] * obs[7]                       # 上一步双腿接触
    curr_both = next_obs[6] * next_obs[7]             # 当前步双腿接触
    landing_transition = max(0.0, curr_both - prev_both)  # 0→1 上升沿

    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    velocity_gate = 1.0 / (1.0 + 2.0 * speed)         # 低速→1，高速→0
    angle_gate = 1.0 / (1.0 + 5.0 * angle_abs)        # 直立→1，倾斜→0

    landing_bonus = 100.0 * landing_transition * velocity_gate * angle_gate

    total_reward = shaped_progress + ang_vel_penalty + landing_bonus
    components = {
        "shaped_progress": shaped_progress,
        "angular_vel_penalty": ang_vel_penalty,
        "landing_bonus": landing_bonus
    }
    return (float(total_reward), components)
```