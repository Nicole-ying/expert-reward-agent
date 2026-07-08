# Response Record

## 诊断

### 0. 任务理解
2D 飞行器，从顶部出发飞向画面中央的目标着陆垫。8 维观测：位置、速度、姿态角、角速度、双腿接触。4 个离散动作：无推力 / 左姿态引擎 / 主引擎 / 右姿态引擎。

### 1. 发生了什么？
**Agent 在 crash。** Score=-161，episode_length=77.85，80% 提前终止（<150步且 score<-50）。Agent 学不会减速降落，大概率高速冲向地面。

### 2. 哪个组件是主因？
- `progress_reward = -distance`：始终为负（mean=-0.945），提供向 pad 靠近的梯度，但**完全不教减速**——径直高速俯冲也是"靠近"。
- `orientation_penalty`：微不足道（ratio=-0.021），不是问题。
- **`landing_bonus` 是核心问题**：二值条件 `if close and slow and legs_contact: bonus=10.0`，nonzero_rate=2.54%。这意味着 97.5% 的 step 中着陆信号为 0，**零梯度**。Agent 无法从"接近但还不够好"的状态获得任何渐进式引导。当它偶然触发时，又是 10.0 的巨大 spike（约为 progress 均值的 10 倍），造成高方差。

### 3. 之前改了什么？
这是第一轮，没有历史包袱。

### 修复方案
**层次 2：改数学形式。** 将 `landing_bonus` 从二值 `if` 改为连续乘积因子（bounded continuous proxy）：`bonus_scale × dist_factor × speed_factor × contact_factor`。每个因子用 `max(0, 1 - x/threshold)` 提供梯度，阈值放宽（dist: 0.3→0.5, speed: 0.5→1.0），让 agent 在接近着陆条件的过程中就获得逐步增强的信号。

只改这一个组件。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v2: 将 landing_bonus 从二值条件改为连续乘积形式，提供着陆梯度。

    Components:
    1. progress_reward:   -distance_to_landing_pad (密集引导)
    2. orientation_penalty: 小惩罚非零姿态角和角速度
    3. landing_bonus:       连续乘积 proxy —— 距离近 × 速度低 × 腿触地
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
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: continuous proxy for successful landing ----
    # 三个连续因子相乘，每个提供梯度，共同约束"接近 + 低速 + 触地"
    dist_thresh = 0.5      # 距离阈值（放宽，提前给梯度）
    speed_thresh = 1.0     # 速度阈值（放宽，提前给梯度）
    bonus_scale = 10.0     # 最大可能奖励（完美着陆时）

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5

    dist_factor = max(0.0, 1.0 - n_distance / dist_thresh)
    speed_factor = max(0.0, 1.0 - n_speed / speed_thresh)
    contact_factor = min(1.0, n_left_contact + n_right_contact)  # 0, 0→0; 1, 0→1; 1, 1→1

    landing_bonus = bonus_scale * dist_factor * speed_factor * contact_factor

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```
