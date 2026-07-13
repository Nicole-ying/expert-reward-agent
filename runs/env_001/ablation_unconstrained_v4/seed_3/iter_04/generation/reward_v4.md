分析：上一轮agent在1000步后均truncated，无终止，接触和着陆组件激活率为0，说明agent从未接触地面，只是在空中盘旋获取高度和目标靠近奖励。引擎使用率87%，累积惩罚大，导致总分为负。问题症结在于无接触引导，密集型高度/目标奖励使agent满足于悬浮。历史最佳iter 1采用了“接触奖励+门控目标奖励”，有部分接触（contact_reward均值0.237），说明门控结构有效。因此，本次修改将以门控接触激励为核心，用线性高度奖励替代倒数式高度奖励以避免悬浮，并增加连续接触引导，使agent必须下降并接触来获得主要奖励。

修改方案：
- 用线性下降奖励（最高限制）替代倒数高度奖励，驱动agent持续下降。
- 设置连续接触奖励（接触时按接近目标程度给分），使接触行为即可获得正向激励。
- 保留成功着陆一次性大奖励。
- 引擎惩罚保持不变。
- 移除原本的密集goal_reward和height_reward。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = obs[0]
    y = obs[1]
    vy = obs[3]
    angle = obs[4]
    left_contact = obs[6]
    right_contact = obs[7]

    # 下降激励：高度高于0.2时给予线性惩罚，推动接近接触高度
    descent_reward = -0.1 * max(y - 0.2, 0.0)

    # 接触奖励：任意支撑腿触地即给分，再乘以接近目标垫的程度
    any_contact = max(left_contact, right_contact)
    distance = (x**2 + y**2) ** 0.5
    contact_proximity = 1.0 / (1.0 + 2.0 * distance)
    contact_reward = any_contact * contact_proximity * 0.8

    # 完美着陆奖励
    both_contact = left_contact and right_contact
    if both_contact and abs(vy) < 0.3 and abs(angle) < 0.3:
        landing_bonus = 10.0
    else:
        landing_bonus = 0.0

    # 引擎使用惩罚
    engine_penalty = -0.2 if action != 0 else 0.0

    total_reward = descent_reward + contact_reward + landing_bonus + engine_penalty

    components = {
        'descent_reward': descent_reward,
        'contact_reward': contact_reward,
        'landing_bonus': landing_bonus,
        'engine_penalty': engine_penalty
    }
    return float(total_reward), components
```