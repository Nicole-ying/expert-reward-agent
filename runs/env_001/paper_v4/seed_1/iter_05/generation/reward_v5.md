1. `evidence`：所有20个episode均terminated（crash），episode length从iter3的495骤降至77；shaped_progress的magnitude_share=55.1%主导奖励流，landing_bonus active_rate仅0.8%几乎不可达，score=-29.48远差于best=27.72。
2. `behavior_diagnosis`：agent学会了"冲向原点"来最大化shaped_progress，但未学会减速与姿态控制，导致快速坠毁；高系数25.0使进度信号压倒所有约束。
3. `signal_completeness`：职责结构基本完备——有进度引导、角速度约束、着陆事件奖励——但进度信号过强导致约束失效、着陆信号因过早坠毁而不可达。
4. `selected_level`：Level 1。组件数学形态合理（距离差分×姿态门控、二次惩罚、转移事件×门控），根本问题是shaped_progress系数相对其他组件严重失衡，符合"单个组件过强"的尺度修复条件。
5. `selected_intervention`：将shaped_progress系数从25.0降至5.0，其他组件不变。
6. `falsifiable_hypothesis`：降低进度奖励后，agent不再被"快速靠近原点"主导，有更长时间探索速度与姿态控制，从而更可能触达landing_bonus并学会安全着陆；episode length应回升，terminated比例应下降。
7. `expected_next_round`：terminated比例从20/20下降，episode length从~77回升，landing_bonus的active_rate从0.8%上升，score向best=27.72靠近或超越。
8. `main_risk`：若进度信号过弱（5.0可能偏低），agent可能失去靠近目标区域的基础引导，陷入长时间无目的徘徊；若出现此情况，下一轮可上调至8~10。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 匿名环境观测假设（经典2D着陆器）：
    # obs[0]: x (水平偏移)    obs[1]: y (高度)
    # obs[2]: vx             obs[3]: vy
    # obs[4]: angle          obs[5]: ang_vel
    # obs[6]: left_contact   obs[7]: right_contact

    # 1. 主学习信号：距离改善 × 姿态门控（系数下调至5.0，防止rush-and-crash）
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_current - dist_next          # 正值表示靠近目标

    angle_abs = abs(next_obs[4])                     # 机体角度（弧度）
    gate      = 1.0 / (1.0 + 5.0 * angle_abs)        # soft health gate
    shaped_progress = 5.0 * progress_raw * gate       # 从25.0降至5.0

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