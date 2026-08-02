1. `evidence`：第6轮score -111.4, len=68.45, 20/20 terminated；progress_delta(+49.9%)和speed_penalty(-41.1%)互相抵消；completion_proxy、angle、angvel、engine全为僵尸组件(active_rate<5%)；累积记录第4-6轮同一骨架连续3轮下降，第6轮加入engine_penalty后崩溃。
2. `behavior_diagnosis`：agent在68步内快速失败（出界或坠毁），engine_penalty(-0.05/步)惩罚了主引擎使用，导致agent不敢点火对抗重力，同时全时speed_penalty(阈值0.4, active_rate 90%)过紧，agent在下降正常速度下也被持续惩罚，最终放弃引擎直接坠落。
3. `signal_completeness`：缺少强着陆引导（第4轮虽得分195但len=754在空中徘徊）、缺少接触奖励（双足接触仅在乘积proxy中且active_rate<1%）、缺少出界预警；速度约束需要从全时惩罚改为hinge只在越界时触发。
4. `selected_level`：Level 3重建——同一骨架族(progress_delta+speed_penalty+completion_proxy+angle/angvel)已迭代3轮(第4-6轮)且第6轮崩溃，best从195降至-111，触发"连续≥3轮未刷新best且4轮后恶化"条件。
5. `selected_intervention`：新骨架移除engine_penalty和completion_proxy乘积，加入contact_reward(双足接触直接正向)、landing_bonus(接近中心且有接触时bonus)、boundary_warning(出界软约束)，将speed_penalty从全时二次改为hinge(阈值0.6)，progress_delta权重从5.0提升至8.0。
6. `falsifiable_hypothesis`：移除引擎惩罚+提高速度hinge阈值到0.6将允许agent正常使用主引擎，len应回升到>200；contact_reward+landing_bonus提供着陆梯度，减少空中徘徊，若下一轮len>300但score仍<180说明着陆引导仍不足；若len仍<100说明hinge阈值或boundary_warning不够阻止出界。
7. `expected_next_round`：len从68回升到>150，terminated率下降（部分episode转为truncated），progress_delta保持正主导，speed_penalty active_rate降至30-60%，contact_reward active_rate>10%，landing_bonus开始触发，score从-111回升至>-50。
8. `main_risk`：progress_delta权重8.0可能在agent远离中心时产生过强负信号（如初始随机速度向外），与boundary_warning叠加导致奖赏剧烈波动，若下一轮score方差增大需降低progress权重。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8

    # 1. 主进展信号：向中心靠近
    progress_delta = 8.0 * (dist - next_dist)

    # 2. 接触奖励：双足接触平台即给正向
    contact_reward = 0.3 * (nleft + nright)

    # 3. 速度约束(higne)：仅在速度过大时惩罚
    speed_threshold = 0.6
    vx_violation = max(0.0, abs(nvx) - speed_threshold)
    vy_violation = max(0.0, abs(nvy) - speed_threshold)
    speed_penalty = -0.1 * (vx_violation + vy_violation)

    # 4. 角度约束(higne)
    angle_threshold = 0.15
    angle_violation = max(0.0, abs(nangle) - angle_threshold)
    angle_penalty = -0.2 * angle_violation

    # 5. 角速度约束(higne)
    angvel_threshold = 0.3
    angvel_violation = max(0.0, abs(nangvel) - angvel_threshold)
    angvel_penalty = -0.1 * angvel_violation

    # 6. 着陆奖励：接近中心且有接触
    landing_bonus = 0.0
    if next_dist < 0.3 and (nleft + nright) >= 1.0:
        landing_bonus = 0.5

    # 7. 边界预警：出界风险软约束
    boundary_warning = -0.5 * max(0.0, next_dist - 1.0)

    total_reward = (
        progress_delta +
        contact_reward +
        speed_penalty +
        angle_penalty +
        angvel_penalty +
        landing_bonus +
        boundary_warning
    )

    components = {
        'progress_delta': progress_delta,
        'contact_reward': contact_reward,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'landing_bonus': landing_bonus,
        'boundary_warning': boundary_warning
    }

    return float(total_reward), components
```