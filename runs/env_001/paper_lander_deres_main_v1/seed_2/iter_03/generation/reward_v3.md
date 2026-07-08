## 诊断

### 问题 0：环境理解
2D 飞行器着陆任务。飞行器从顶部出发，需飞到中央着陆垫并稳定停稳。观测包括位置、速度、姿态角、角速度、双腿接触标志。动作是离散的 4 选 1（无推力/左姿态/主引擎/右姿态）。

### 问题 1：Agent 发生了什么？
- 得分从 -161 → -121，提升了 25%，方向正确。
- 但 episode_length 仍只有 77.85，**70% 的 episode 提前终止（crash）**。
- 这意味着 agent 仍然在大量坠毁，尚未学会可控着陆。

### 问题 2：哪个组件是主要原因？
- `progress_reward`（-0.935 mean）始终为负，是唯一密集信号，但它只说"靠近目标"，不说"减速"。
- `orientation_penalty`（ratio -0.021）太弱，几乎不影响行为。
- `landing_bonus` 的 **nonzero_rate 仅 4.38%**——关键在于 `contact_factor` 是二值的（腿触地才 >0）。这意味着 95%+ 的 step 里 agent 只收到负信号，没有正向引导它"减速接近"。等到触地时已经太晚。

### 问题 3：上一轮改了什么？
上一轮把 landing_bonus 从二值改为连续乘积（dist × speed × contact），得分从 -161 提升到 -121。方向正确。但 contact_factor 仍是二值门控，导致 bonus 只在触地瞬间激活。

### 修复方案
**移除 contact_factor**，让 landing_bonus = `bonus_scale × dist_factor × speed_factor`。这样只要飞行器「靠近目标 + 低速」，就能获得正向梯度，无需等腿触地。这是对 landing_bonus 单个组件的数学形式修改（层次 2），不改其他组件。

`dist_thresh=0.5` 保证只在着陆垫附近才激活；`speed_thresh=1.0` 保证只有低速才激活。两者乘积天然 bounded 在 [0, 1]，不会数值爆炸。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v3: 移除 landing_bonus 中的 contact_factor，让接近+低速的梯度在触地前就生效。

    诊断：v2 的 landing_bonus 有 contact_factor（二值），导致 nonzero_rate 仅 4.38%。
    95%+ 的 step 没有正向信号，agent 缺乏"减速接近"的引导，crash 率 70%。

    修改：landing_bonus = bonus_scale * dist_factor * speed_factor（去掉 contact_factor）。
    靠近目标垫且速度低即有奖励，不等腿触地。

    Components:
    1. progress_reward:   -distance_to_landing_pad（密集引导，不变）
    2. orientation_penalty: 小惩罚非零姿态角和角速度（不变）
    3. landing_bonus:       连续乘积 —— 距离近 × 速度低（不再要求触地）
    """
    # ---- unpack observations ----
    x_pos = obs[0]
    y_pos = obs[1]
    body_angle = obs[4]
    angular_vel = obs[5]

    n_x_pos = next_obs[0]
    n_y_pos = next_obs[1]
    n_x_vel = next_obs[2]
    n_y_vel = next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: continuous proxy — proximity × low speed (no contact gate) ----
    # 去掉了 contact_factor，让 agent 在接近+减速时就收到正向梯度
    dist_thresh = 0.5
    speed_thresh = 1.0
    bonus_scale = 10.0

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5

    dist_factor = max(0.0, 1.0 - n_distance / dist_thresh)
    speed_factor = max(0.0, 1.0 - n_speed / speed_thresh)

    landing_bonus = bonus_scale * dist_factor * speed_factor

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```