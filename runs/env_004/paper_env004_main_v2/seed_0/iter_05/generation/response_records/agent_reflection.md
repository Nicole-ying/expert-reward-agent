# Response Record

`evidence`: 当前得分724远低于best的2926，episode长度从1000骤降至237且全部terminated(20/20)；本次仅比best多加了energy_penalty(w=0.05)，其magnitude_share仅1.7%但策略完全崩溃。

`behavior_diagnosis`: 机器人快速摔倒（len=237全终止），energy_penalty虽数值小但改变了动作分布，导致跳跃无力和姿态失控。

`signal_completeness`: best结构（forward_vel×posture_factor + ang_vel_penalty）已具备前向引导和稳定约束两个必要职责，能存活完整1000步并获2926分；缺失对跳跃动作本身的直接鼓励。

`selected_level`: Level 2 — 需要加入新的跳跃引导信号来缩小2926→3800的差距，不能简单回退best。

`selected_intervention`: 以best代码为基础，新增`vertical_reward`组件：用`max(0, next_obs[6])`奖励向上的垂直速度，系数0.3，鼓励更有力的跳跃以提升前进效率。

`falsifiable_hypothesis`: 当前best的agent能稳定存活但跳跃力度不足，限制前进步幅；向上速度奖励应激励更强跳跃，从而增加每跳的前进距离，提升总分。

`expected_next_round`: score应向3800方向上升（超过best的2926），episode长度保持~1000且terminated保持0，vertical_reward的active_rate应在跳跃阶段非零、magnitude_share约5-15%。

`main_risk`: 垂直奖励可能诱导agent原地垂直跳而非向前跳；若forward_vel未同步提升而vertical_reward占比较高，后续需加门槛或改用跳跃高度差。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 从 next_obs 提取关键信号
    torso_angle = next_obs[1]           # 躯干倾角
    forward_vel = next_obs[5]           # 前进速度（水平速度）
    torso_ang_vel = next_obs[7]         # 躯干角速度
    vertical_vel = next_obs[6]          # 垂直速度

    # 姿态因子：躯干越直立（角度接近0），因子越接近1
    k_angle = 10.0
    posture_factor = 2.718281828 ** (-k_angle * (torso_angle ** 2))

    # 主学习信号：前进速度 × 姿态因子
    #   - 前进速度为正则奖励，为负则惩罚，引导 agent 持续向前
    #   - 姿态因子在躯干倾斜时自动衰减，抑制为追求速度而过度倾斜
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：惩罚躯干角速度，抑制快速晃动/摔倒
    w_ang_vel = 0.05
    ang_vel_penalty = w_ang_vel * abs(torso_ang_vel)

    # 垂直速度奖励：奖励向上的速度分量，鼓励有力跳跃
    #   仅对正值（上升阶段）生效，下降阶段为0，避免惩罚落地
    #   系数0.3：单步贡献约0~0.6，约为speed_posture的5%~15%
    w_vertical = 0.3
    vertical_reward = w_vertical * max(0.0, vertical_vel)

    total_reward = speed_posture_reward - ang_vel_penalty + vertical_reward

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty,
        'vertical_reward': vertical_reward
    }

    return float(total_reward), components
```
