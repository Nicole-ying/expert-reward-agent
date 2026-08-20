# Prompt Record

## System Prompt

```text
你是奖励函数修订 Agent。根据训练反馈，修改奖励函数以改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- 你将看到：最终评估的组件表（episode_sum_mean, magnitude_share, active_rate），以及训练过程中的组件轨迹（每个 checkpoint 的标量值，含 max/mean/min）。轨迹能告诉你组件是在学习还是停滞了。
- 组件统计是观察证据，不是因果贡献。必须结合 score、episode_length、terminated/truncated 判断。

# 反思规则

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

分析完每个组件后再写代码。一次只改一个组件，不要顺带调整其他。

# 代码约束

- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的 obs、next_obs、action 和 info 字段，不得发明字段、切片维度或新输入。
- 第一个 Python code block 只能包含一个完整的 `compute_reward` 函数；不要写 import、class、try/except 或额外函数，不要使用 self。
- 禁止 eval/exec/open，禁止使用 original_reward 或原始环境 reward。
- 需要平方根时使用 `** 0.5`，禁止 import numpy。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`；components 只放总公式中直接出现的奖励组件。

# 输出

Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.

```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: 52.523801
- gap_to_target: 147.476199

# 2. Current reward program (score: 52.523801)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state
    px, py, vx, vy, angle, ang_vel = next_obs[0], next_obs[1], next_obs[2], next_obs[3], next_obs[4], next_obs[5]
    left_contact, right_contact = next_obs[6], next_obs[7]

    # ---- New core signal: distance penalty ----
    # Agent is heavily pushed to minimise its distance to the centre (0,0)
    curr_dist = (px ** 2 + py ** 2) ** 0.5
    distance_penalty = -0.5 * curr_dist

    # ---- Attitude (keep upright) ----
    angle_penalty = -0.5 * (angle ** 2)

    # ---- Rotation smoothness ----
    angvel_penalty = -0.1 * (ang_vel ** 2)

    # ---- Linear speed: now a gentle penalty (was 0.2, dwarfed everything) ----
    vel_penalty = -0.02 * (vx ** 2 + vy ** 2)

    # ---- Landing contact: much stronger incentive ----
    contact_reward = 10.0 * left_contact + 10.0 * right_contact + 30.0 * left_contact * right_contact

    total = distance_penalty + angle_penalty + angvel_penalty + vel_penalty + contact_reward
    components = {
        "distance_penalty": distance_penalty,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "vel_penalty": vel_penalty,
        "contact_reward": contact_reward
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=52.523801, len=507.200000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[-91.792933, 208.111232]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 6100.000000 | 97.8% | 97.8% | 29.5% |
| distance_penalty | -129.115543 | -2.1% | 2.1% | 100.0% |
| angle_penalty | -4.202416 | -0.1% | 0.1% | 100.0% |
| vel_penalty | -1.095749 | -0.0% | 0.0% | 99.9% |
| angvel_penalty | -0.511493 | -0.0% | 0.0% | 96.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Checkpoint evaluations (every 10% of training)
| pct | score_mean | score_std |
|---|---|---|
| 40% | 0.00 | 0.00 |
| 80% | 0.00 | 0.00 |

# 4. Environment facts
## 1. 任务目标
该环境是一个 2D 飞行器轨迹优化任务。智能体需要从初始位置（视口顶部中心附近）出发，快速、稳定地降落到画面中央的目标接触垫上，同时尽可能少使用引擎推力。任务要求智能体学会靠近目标区域、减速、保持机体垂直，并用两条着陆腿安全接触垫面。

## 3. 观察空间 observation_space
-
```
