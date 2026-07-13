# 分析

## 1. 这个 agent 发生了什么？
- **高度奖励塞满信号，导致 reward hacking**  
  `height_near` 的 `episode_sum_mean` 高达 4738.6，占 signed_share 的 94.9%。这是因为 `height_near = 5*(max(0, 1.5-y))²` 在下降过程中持续给予正奖励，且未与任何安全着陆条件关联。agent 学会了尽可能贴近地面以获取大量奖励，但并没有学会稳定、安全的着陆。
- **原始任务得分极差（-119.6）**  
  即使 shaping 奖励组件总和为正（约 4965），评估用的原始环境累积奖励仍为负，说明 agent 频繁坠毁（terminated=16/20，很可能是 crash 终止），因为缺乏水平对准、姿态控制和最终的接触质量。
- **关键 sub‑signal 极度稀疏或无效**  
  `contact_reward`（二进制双腿触碰 max）的 active_rate 仅 1.2%，表明支撑腿极少同时接触地面。`progress` 几乎为零贡献，`lat_penalty`、`att_penalty`、`angvel_penalty` 虽被激活但权重太低，无法改变行为。

## 2. 哪个组件最值得干预？
**`height_near` 的数学形态**。单一的二阶正奖励在整段下降中提供密集信号，使 agent 找到捷径（快速下降）而忽略水平对准与姿态稳定。必须将其与着陆质量（双腿接触 + 姿态竖直）进行 gate 耦合，迫使 agent 只有在安全着陆条件下才能获得高度进度回报。同时，垂直速度惩罚过于宽松（limit=0.5，active_rate 0.4%），需要收紧以防止高速撞击。

## 3. 我之前改了什么？
上一轮（iter 7）在 best（iter 3）的基础上引入了 `height_near` 的凸化形态、`alignment` 的门控和多重速度/姿态惩罚，但未解决 `height_near` 的无条件高权重问题，导致 reward hacking。best（iter 3）使用了 alignment + contact_reward + height_near_reward + horizontal_penalty + progress 等组件，但形态不明，得分 71.06。本次修改以 best 的结构为灵感，但做出实质改变：`height_near` 乘以 `landing_quality` gate，并重构 contact 为两腿同时接触的连续激励，移除 progress，增加 `still_bonus` 和更严格的下降速度惩罚。

# 修改后的奖励函数
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 读取状态
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    vx_new = next_obs[2]
    vy_new = next_obs[3]
    angle_new = next_obs[4]
    angvel_new = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 高度近地奖励（受着陆质量 gate 调制）----------
    height_threshold = 2.0
    height_near_raw = max(0.0, height_threshold - y_new)
    # 着陆质量因子：双腿平均接触度 × 姿态竖直度
    contact_factor = (left_contact + right_contact) / 2.0
    angle_factor = 1.0 / (1.0 + 10.0 * abs(angle_new))
    landing_quality = contact_factor * angle_factor
    height_near = 0.5 * (height_near_raw ** 2) * landing_quality

    # ---------- 水平对准（全程激励，不依赖高度）----------
    alignment = 2.0 / (1.0 + 10.0 * (x_new ** 2))

    # ---------- 接触奖励（双腿同时接触时给予高奖励）----------
    both_contact = left_contact * right_contact
    contact_reward = 5.0 * both_contact

    # ---------- 垂直速度惩罚（更严格的下降限速）----------
    vy_limit = 0.3
    down_penalty = -5.0 * (max(0.0, -vy_new - vy_limit) ** 2)

    # ---------- 水平速度惩罚 ----------
    vx_limit = 0.2
    lat_penalty = -2.0 * (max(0.0, abs(vx_new) - vx_limit) ** 2)

    # ---------- 姿态角惩罚 ----------
    angle_limit = 0.05
    att_penalty = -2.0 * (max(0.0, abs(angle_new) - angle_limit) ** 2)

    # ---------- 角速度惩罚 ----------
    angvel_limit = 0.05
    angvel_penalty = -2.0 * (max(0.0, abs(angvel_new) - angvel_limit) ** 2)

    # ---------- 趋近静止奖励（鼓励最终稳定）----------
    speed_norm = (vx_new ** 2 + vy_new ** 2) ** 0.5
    still_bonus = 0.5 * max(0.0, 0.1 - speed_norm)

    total_reward = (
        height_near +
        alignment +
        contact_reward +
        down_penalty +
        lat_penalty +
        att_penalty +
        angvel_penalty +
        still_bonus
    )

    components = {
        'height_near': height_near,
        'alignment': alignment,
        'contact_reward': contact_reward,
        'down_penalty': down_penalty,
        'lat_penalty': lat_penalty,
        'att_penalty': att_penalty,
        'angvel_penalty': angvel_penalty,
        'still_bonus': still_bonus
    }

    return float(total_reward), components
```