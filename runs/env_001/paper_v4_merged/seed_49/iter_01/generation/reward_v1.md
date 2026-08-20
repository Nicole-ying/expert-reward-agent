# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重和阈值
    w_progress = 1.0
    w_angle = 0.5
    w_angvel = 0.1
    w_soft_land = 2.0
    w_eff = 0.02
    
    angle_thresh = 0.3   # rad
    angvel_thresh = 1.0  # rad/s
    max_speed_land = 1.0 # 着陆容许最大合速度
    max_angle_land = 0.5 # 着陆容许最大倾角 rad
    
    # 距离进展（步间距离减少量）
    old_dist = (obs[0]**2 + obs[1]**2)**0.5
    new_dist = (next_obs[0]**2 + next_obs[1]**2)**0.5
    delta_dist = old_dist - new_dist   # 正值表示向目标接近
    progress = w_progress * delta_dist
    
    # 姿态稳定性（hinge 惩罚）
    angle = next_obs[4]
    angvel = next_obs[5]
    angle_penalty = -w_angle * max(0.0, abs(angle) - angle_thresh)
    angvel_penalty = -w_angvel * max(0.0, abs(angvel) - angvel_thresh)
    
    # 软着陆奖励（仅在支撑腿接触时有效）
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    any_contact = 1.0 if (left_contact + right_contact) > 0.5 else 0.0
    
    speed = (next_obs[2]**2 + next_obs[3]**2)**0.5
    speed_factor = 1.0 - min(1.0, speed / max_speed_land)
    angle_factor = 1.0 - min(1.0, abs(angle) / max_angle_land)
    soft_landing_score = speed_factor * angle_factor
    soft_landing = w_soft_land * soft_landing_score * any_contact
    
    # 发动机使用惩罚（离散动作每次非零动作）
    eff_penalty = -w_eff * (0.0 if action == 0 else 1.0)
    
    total_reward = progress + angle_penalty + angvel_penalty + soft_landing + eff_penalty
    
    components = {
        'progress': progress,
        'angle_penalty': angle_penalty,
        'angvel_penalty': angvel_penalty,
        'soft_landing': soft_landing,
        'efficiency': eff_penalty
    }
    return float(total_reward), components
```

# reward_v1 设计说明

**任务类型**  
- task_family: `navigation_goal_reaching`  
- dynamics_subtype: `goal_approach_and_soft_contact`  
- control_type: discrete (4 actions)

**选择的奖励角色及信号映射**  
1. **approach_progress（主学习信号）**  
   - 角色：引导飞行器每步向目标垫靠近。  
   - 信号：`obs[0]`, `obs[1]`, `next_obs[0]`, `next_obs[1]` → 计算欧氏距离差 `old_dist - new_dist`。  
   - 公式算子：`improvement_delta`，确保只有“缩短距离”的行为得到正向奖励，避免停滞刷分。  

2. **orientation_stability（安全约束）**  
   - 角色：防止过大倾斜和高角速度导致失控或侧翻。  
   - 信号：`next_obs[4]` (倾角), `next_obs[5]` (角速度)。  
   - 公式算子：`hinge_penalty`，只在倾角 > 0.3 rad 或角速度 > 1.0 rad/s 时施加惩罚，保留正常调整空间。  

3. **soft_landing_condition（任务完成近似）**  
   - 角色：在支撑腿接触垫面时，要求低速度、小角度，给出连续的正向奖励。  
   - 信号：`next_obs[2]`, `next_obs[3]` (速度), `next_obs[4]` (角度), `next_obs[6]`, `next_obs[7]` (接触标志)。  
   - 公式算子：`joint_condition_proxy`（乘积形式），分别将速度和角度映射到 [0,1] 区间后相乘，最终乘以接触指示。完美着陆得约 2.0，不满足条件时奖励为 0，不会产生负奖励。  

4. **engine_efficiency（效率约束）**  
   - 角色：轻微惩罚发动机使用，鼓励节能（次目标）。  
   - 信号：`action`。  
   - 公式算子：`action_efficiency`，非零动作（任一发动机启动）每次步给予 -0.02。权重极小，避免压制探索和必要的主发动机使用。  

**排除的角色及原因**  
- `terminal_success_bonus` / `crash_penalty`：环境无显式成功/失败标志（`info` 为空），从观测推断终止原因的可靠性不足，v1 暂不引入稀疏终端奖励。  
- `survival_time_reward`：任务不奖励停留，该角色会鼓励拖延，不符合目标。  
- `constant_forward_velocity`：本环境要求减速而不是维持高速，该角色会加剧硬着陆风险。  
- `path_smoothness_penalty`：动作序列的平滑性并非 v1 必须，且离散动作难以定义相邻动作差异，留到后续迭代。  

**为何没有使用 terminal_success_reward / terminal_failure_penalty**  
环境未提供 `info` 中的 `success` 或 `failure` 字段，强行推断存在较大风险，可能奖励错误的终止行为，因此 v1 完全依赖 step-wise 信号。  

**后续迭代可补充的职责**  
- 终端推断的稀疏成功奖励 / 失败惩罚（待观测数据验证推断阈值）  
- 更精细的燃料消耗度量（如持续推力的时间积分）  
- 动态课程权重，在学会基本接近后再强化姿态和效率要求  

**训练后应关注的 failure modes**  
1. 飞行器悬停在上空，利用姿态调整获得少量 progress，但长期不降落。  
2. 在支持腿触地时仍带有高速或大角度，获得低 soft_landing 分数，可能导致硬着陆甚至 crash。  
3. 过度使用姿态发动机频繁摆动，徒增效率惩罚而影响动作效率。  
4. 学习后期平衡权重：若 progress 权重过小，接近目标后可能缺乏降落动力；若 soft_landing 过大，可能导致提前接触后主动脱离以重复收割奖励（需结合环境终止逻辑观察）。