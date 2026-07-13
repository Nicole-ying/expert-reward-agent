# ⚠️ 上一版代码验证失败
错误信息：Reward v8 failed validation: 缺少准确函数签名; 代码无法解析 AST: invalid character '（' (U+FF08) (<unknown>, line 2) (record: runs\env_001\ablation_eureka_feedback_v4\seed_3\iter_08\generation\validations\reward_v8.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
# 设计理由
当前 `landing_quality` 给每步持续接触高额奖励（单步可达 2.0），导致 agent 学会刷双足接触分而不完成稳定停靠；外部任务得分却停滞在 1.56，远低于最佳 115.51。本轮将此组件替换为**一次性双足着陆事件奖励**，只在“从上一步无双足接触到这一步同时双足接触”时发放，并根据水平位置、速度、角度的连续因子缩放（基础值 10.0×质量因子），从而消除持续刷分可能，同时通过密集的 `proximity` 保持接近引导。系数 10.0 使成功着陆时给予显著但不过大的信号，质量因子阈值（x<0.2, 速度<0.3, 角度<0.2 等）设计为足够苛刻以保证着陆质量。

```python
def compute_reward(obs, action, next_obs

# Search objective
- target_score: 200.000000
- current_score: 1.562037
- gap_to_target: 198.437963
- target_achievement_ratio: 0.781%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 1.562037）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前状态
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nleft, nright = next_obs[6], next_obs[7]

    # 1. Proximity: 向目标移动的距离减少量
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_next = (nx ** 2 + ny ** 2) ** 0.5
    progress = max(0.0, dist_curr - dist_next)

    # 2. Landing quality: 基于即时状态的着陆姿态、速度与接触奖励
    if ny < 0.3:
        angle_penalty = -0.2 * abs(nangle)
        vel_penalty = -0.1 * max(0.0, -nvy - 0.2)
        contact_reward = 1.0 * (nleft + nright)
        landing_quality = angle_penalty + vel_penalty + contact_reward
    else:
        landing_quality = -0.05 * abs(nangle)

    # 3. Energy penalty: 惩罚不必要的引擎使用
    energy_penalty = -0.01 if action != 0 else 0.0

    total_reward = progress + landing_quality + energy_penalty

    components = {
        "proximity": progress,
        "landing_quality": landing_quality,
        "energy_penalty": energy_penalty
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback (EUREKA-style)

## Final-policy outcome

score=1.562037, len=954.350000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-56.841969, 157.605284]

## Reward component values (mean per episode)
- landing_quality: 43.895766
- energy_penalty: -9.336000
- proximity: 1.316316

# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器着陆/停靠任务。飞行器起始于视野上方中央区域，受到随机初速度扰动。主要目标是尽快抵达中央的目标垫（target pad）并稳定停靠（settle），以最小的引擎推力消耗完成。飞行器需要学会靠近目标、降低速度、保持姿态平稳，并实现安全的双足（左右支撑）接触，最终停在目标垫上。避免与目标垫以外的任何部位发生碰撞、飞出视口边界或长时间不稳定晃动。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断，因为 Box 默认 float32）
- obs[0]: x_position（相对于目标垫的水平坐标），含义：飞行器在 x 方向上偏离目标垫中心的距离，可用于奖励接近目标；reward_usable: true
- obs[1]: y_position（相对于目标垫高度的垂直坐标），含义：飞行器高度与目标垫高度之差，可用于奖励下降/靠近垫面；reward_usable: true
- obs[2]: x_velocity（水平线速度），含义：横向移动速度，可用于惩罚过大侧向速度或奖励静止；reward_usable: true
- obs[3]: y_velocity（垂直线速度），含义：竖直方向速度，可用于惩罚硬着陆（大负值）或奖励稳定；reward_usable: true
- obs[4]: body_angle（机体角），含义：飞行器倾斜角度，可用于奖励保持平正姿态；reward_usable: true
- obs[5]: angular_velocity（角速度），含义：机体旋转速率，可用于奖赏姿态稳定；reward_usable: true
- obs[6]: left_support_contact（左支撑腿接触标志，1.0 或 0.0），含义：左腿是否与目标垫良好接触，可用于奖励双足着陆；reward_usable: true
- obs[7]: right_support_contact（右支撑腿接触标志，1.0 或 0.0），含义：右腿是否与目标垫良好接触，可用于奖励双足着陆；reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine（不操作）—— 不做任何推力/姿态调整，依赖当前动量。
- action 1: left_orientation_engine（左姿态引擎）—— 点火一个姿态引擎，产生逆时针/顺时针力矩以调整机体角度。
- action 2: main_engine（主引擎）—— 点火主引擎，产生主要推力（可能方向固定，对 body 坐标系）。
- action 3: right_orientation_engine（右姿态引擎）—— 点火相反的另一个姿态引擎。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled —— 机体不再活跃或已稳定停靠。这可能意味着飞行器已处于静止状态且（通常）两个支撑腿接触目标垫。高风险：如果发生于 crash 后也可能导致不再活跃，因此该终止条件不能单方面被认定为成功。需要结合接触标志、位置、速度等信号综合判断。
- failure-like termination: 
  - crash_or_body_contact —— 身体某部位（非支撑腿）发生碰撞，可能是撞击地面或目标垫以外的区域。
  - horizontal_position_outside_viewport —— 水平方向飞出视口边界。
- ambiguous termination: 无其他。
- truncation: 无显式时间截断（step 源码未显示 max_steps，默认无截断，由 gym wrapper 决定，但环境无）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （empty dict，无可用字段）
- forbidden_or_uncertain_info_fields: 所有未在源中列出的信息字段（none）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1]) 以及 next_obs 对应值。
- velocity: x_velocity (obs[2]), y_velocity (obs[3])。
- orientation: body_angle (obs[4]), angular_velocity (obs[5])。
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])。
- action/engine: 当前动作可以用于奖励/惩罚引擎使用（鼓励使用 no_engine）。
- other: 差分信号，如位置变化、速度变化、角度变化，均可从 obs 和 next_obs 构建。