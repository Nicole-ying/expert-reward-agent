# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs\env_002\paper_bipedal_main_v1\seed_2\iter_03\generation\validations\reward_v3.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
## 诊断与干预分析

`evidence`：所有20个评估回合均以抵达地形终点终止（terminated=20/20），episode长度1038步，外部得分307.9。forward_reward episode总和504.2（magnitude_share 83%），energy_penalty -74.4（12.2%），stability_penalty -29.2（4.8%）。上一轮添加energy_penalty后得分从272升至308，episode从1284步缩短至1038步——惩罚推动效率改善，但改善机制已接近耗尽。

`behavior_diagnosis`：agent已学会稳定完成地形行走，但处于中等速度平台。因为forward_reward = forward_velocity 的总和近似等于地形长度（固定值），与速度无关；agent提升速度的唯一激励是减少penalty累积步数。当penalty已经较小时（energy仅占总magnitude的12%），这个间接激励极弱——agent没有理由在"够用"的速度之上继续加速。

`signal_completeness`：前向进度（velocity）、姿态约束（angle+vertical）和能量约束（torque magnitude）三个职责均已覆盖且可达。缺失的是**直接的速度激励**：当过程最优（走完全程）不再等价于速度最优时，当前线性forward reward无法区分v=0.5走1000步和v=1.0走500步——两者总forward reward相同。这是一个结构性的"dense proxy形成中等分平台"的信号完备性问题。

`selected_level`：Level 2。触发条件：职责覆盖基本完备但forward_reward的线性数学形态导致速度激励塌缩——agent完成地形后缺乏加速动力，已构成"dense_to_task_event"模式中描述的稠密proxy平台效应。这不是系数失调（Level 1无法修复，因为提高w_fwd不改变速度无关的数学性质），而是需要改变forward_reward的数学结构使其对速度超线性敏感。

`selected_intervention`：唯一目标组件为forward_reward。将其从线性 `forward_velocity` 改为凸函数 `forward_velocity + 0.2 * forward_velocity²`。线性项保留基础前进引导，二次项创造直接速度激励：总quadratic贡献 = α·L·v，依赖速度本身。其他组件（stability_penalty、energy_penalty）完全不变。

`falsifiable_hypothesis`：超线性forward reward使总奖励直接依赖前进速度（而不仅仅是距离），agent会为最大化总奖励而提升步速。episode长度应缩短，forward_reward的episode_sum_mean（含二次项）应上升，外部得分应提高。如果episode长度不变或forward_velocity不变，说明agent已接近物理速度上限或二次项系数过小。

`expected_next_round`：episode_length下降到800–950步区间；forward_reward的episode_sum_mean因二次项贡献上升（预估520–560）；energy_penalty可能因更高扭矩略有上升；stability_penalty应保持低位；外部score突破315+。

`main_risk`：二次项可能诱导agent在起步阶段过度加速导致摔倒。但现有的stability_penalty（|hull_angle|和|vertical_velocity|）和energy_penalty（action²）构成自然约束——更高速度需要更大扭矩，能量惩罚二次增长会制衡。若下一轮出现early_terminal增加或stability_penalty飙升，则需回退二次系数或加入速度

# 上一轮奖励函数代码（该轮得分: 307.917359）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D bipedal locomotion on rough terrain.
    Drives forward progress, maintains stable posture, and encourages energy-efficient motion.
    """
    # ---------- forward progress component ----------
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # ---------- stability penalty component ----------
    hull_angle = next_obs[0]
    vertical_velocity = next_obs[3]
    w_angle = 0.5
    w_vertical = 0.1
    stability_penalty = -w_angle * abs(hull_angle) - w_vertical * abs(vertical_velocity)

    # ---------- energy efficiency component ----------
    w_energy = 0.05
    energy_penalty = -w_energy * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)

    # ---------- total reward ----------
    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=307.917359, len=1038.650000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[306.582755, 309.758612]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.190645 | 82.9% | 83.0% | 100.0% |
| energy_penalty | -74.388368 | -12.2% | 12.2% | 100.0% |
| stability_penalty | -29.165124 | -4.8% | 4.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 双足机器人运动任务。机器人需要在崎岖地形上稳定行走，尽可能走得远、走得快，同时尽量节省能量。机器人拥有髋关节和膝关节，需要协调四条关节（双腿各一髋一膝）产生连续的步态。如果身体摔倒，回合立即终止；如果抵达地形终点，回合也终止。任务希望机器人学会快速、高效的前进步态，避免摔倒。

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: (24,)
- dtype: 推断为 float32 或 float64
- obs[0]: hull_angle —— 主躯干相对于直立方向的角度
- obs[1]: hull_angular_velocity —— 主躯干角速度
- obs[2]: horizontal_velocity —— 水平（前进/后退）线速度
- obs[3]: vertical_velocity —— 垂直方向线速度
- obs[4]: hip1_angle —— 腿1髋关节角度
- obs[5]: hip1_speed —— 腿1髋关节角速度
- obs[6]: knee1_angle —— 腿1膝关节角度
- obs[7]: knee1_speed —— 腿1膝关节角速度
- obs[8]: leg1_contact —— 腿1触地标志（1.0 触地，0.0 离地）
- obs[9]: hip2_angle —— 腿2髋关节角度
- obs[10]: hip2_speed —— 腿2髋关节角速度
- obs[11]: knee2_angle —— 腿2膝关节角度
- obs[12]: knee2_speed —— 腿2膝关节角速度
- obs[13]: leg2_contact —— 腿2触地标志（1.0 触地，0.0 离地）
- obs[14..23]: lidar_0..lidar_9 —— 10 个激光雷达距离测量值，用于感知前方地形

## 4. 动作空间 action_space
- type: Box（连续动作）
- shape: (4,)
- 取值范围: 每维均在 [-1.0, 1.0]（关节扭矩）
- action 0: hip_torque_leg1 —— 腿1髋关节扭矩
- action 1: knee_torque_leg1 —— 腿1膝关节扭矩
- action 2: hip_torque_leg2 —— 腿2髋关节扭矩
- action 3: knee_torque_leg2 —— 腿2膝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（抵达地形终点，通常算成功完成任务）
- failure-like termination: body_fallen_over（身体摔倒，失败）
- ambiguous termination: 无
- truncation: 无（step 返回的 terminated 标志直接表示结束，没有另行截断）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，没有 success 字段）
- explicit_failure_flag_available: false（info 为空，没有 failure 字段）
- allowed_info_fields: 无（info 为空字典 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 7. 可用于奖励函数的信号
- position（通过观测可间接获得位移增量）：
  - 可利用连续帧的 horizontal_velocity（obs[2] 与 next_obs[2]）计算前进距离增量
  - 但注意没有直接的位置坐标
- velocity：
  - obs[2] horizontal_velocity（前进速度）
  - obs[3] vertical_velocity（垂直速度，可用于惩罚跳跃或坠落）
- orientation：
  - obs[0] hull_angle（躯干倾角，接近 0 表示直立）
  - obs[1] hull_angular_velocity（倾摆角速度，可用于惩罚快速翻滚）
- contact：
  - obs[8] leg1_contact（触地标志）
  - obs[13] leg2_contact（触地标志）
- action/engine：
  - 动作扭矩（action[0..3]）可用于衡量能耗（例如平方和）