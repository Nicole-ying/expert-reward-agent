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