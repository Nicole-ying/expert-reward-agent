# Response Record

1. `evidence`：best=195.26（iter4）后续4轮未刷新；当前score=-58.91，全truncated(1000步)，completion_bonus/angvel_penalty dead(active=0)，angle/speed penalty active<3%；乘积式5条件同时满足从未触发，agent仅靠微弱progress(3.97)徘徊，高安全阈值使约束信号丢失。
2. `behavior_diagnosis`：agent学会朝中心移动但progress回报偏小，安全约束几乎沉默，没有完成信号牵引，无法实现精确着陆；徘徊在平台附近不终止，累计净奖励转负。
3. `signal_completeness`：主进展(delta)存在但弱，完成信号(同时多条件要求)未形成有效梯度，安全约束阈值过高导致无声，缺少近端吸引子→必要职责瑕疵。
4. `selected_level`：Level 3重建——连续≥3轮未刷best(iter5-8均低于195.26)且当前score<-50，历史乘积式框架已证明不可修复。
5. `selected_intervention`：将乘积式completion替换为min‑joint completion（多维因子取最小值），加入contact_reward引导双脚触地，加入boundary_penalty防止出界，降低所有hinge阈值使约束active率提升。
6. `falsifiable_hypothesis`：min‑joint completion在每个维度均提供非塌缩梯度，迫使agent同时改善距离/速度/姿态/接触；contact_reward增强着陆意图；降低的hinge阈值让安全约束被感知→应引导agent从徘徊转为稳定着陆，score回升。
7. `expected_next_round`：score>50（有望接近150），len缩短至600-850（触发稳定终止），completion active_rate>10%，speed/angle/angvel penalty active_rate>10%，terminated占比上升；若无改善→min结构本身错误。
8. `main_risk`：boundary_penalty过早触发可能抑制早期远离中心的探索，若active_rate高且len仍1000需削弱系数。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft, nright = next_obs

    # 到目标中心距离
    dist = (x**2 + y**2)**0.5 + 1e-8
    next_dist = (nx**2 + ny**2)**0.5 + 1e-8

    # 基准进展信号（delta）
    progress = 5.0 * (dist - next_dist)

    # 接触奖励（引导脚触地）
    contact_reward = 0.2 * (nleft + nright)

    # 完成因子（各子条件连续映射到[0,1]）
    proximity_factor = max(0.0, 1.0 - next_dist / 0.3)            # 距中心<0.3
    velocity_factor  = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 0.3)  # 合速度<0.3
    angle_factor     = max(0.0, 1.0 - abs(nangle) / 0.15)         # 倾角<0.15 rad
    angvel_factor    = max(0.0, 1.0 - abs(nangvel) / 0.2)         # 角速度<0.2 rad/s
    contact_factor   = (nleft + nright) / 2.0                     # 双脚接触程度

    # min‑joint completion：只有最差条件改善总分才提高
    completion = 10.0 * min(proximity_factor, velocity_factor, angle_factor, angvel_factor, contact_factor)

    # 安全阈值惩罚（降低阈值使约束可感知）
    speed_penalty    = -0.5 * (max(0.0, abs(nvx) - 0.4) + max(0.0, abs(nvy) - 0.4))
    angle_penalty    = -1.0 * max(0.0, abs(nangle) - 0.15)
    angvel_penalty   = -0.3 * max(0.0, abs(nangvel) - 0.3)
    boundary_penalty = -2.0 * max(0.0, abs(nx) - 0.8)   # 水平出界预警

    total_reward = (progress + contact_reward + completion +
                    speed_penalty + angle_penalty + angvel_penalty + boundary_penalty)

    components = {
        'progress': progress,
        'contact_reward': contact_reward,
        'completion': completion,
        'speed_penalty': speed_penalty,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'boundary_penalty': boundary_penalty
    }

    return float(total_reward), components
```
