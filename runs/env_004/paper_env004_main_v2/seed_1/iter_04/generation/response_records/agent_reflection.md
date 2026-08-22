# Response Record

## 1. evidence
V3将二次速度bonus从无加到0.1系数后得分从3168升至3260（+92），所有episode跑满1000步，forward_progress与bonus合计占signed_share的96.1%，两个惩罚项极小（合计-3.9%），说明策略已稳定但卡在中等速度平台，距离target还差540分。

## 2. behavior_diagnosis
策略已学会稳定前进并利用二次bonus获取额外奖励，但当前v_fwd≈2.26的步态已成为局部最优——二次系数0.1产生的边际激励不足以推动策略冒险尝试更高速的步态。

## 3. signal_completeness
前向进度（线性+二次）、躯干倾角惩罚、角速度惩罚四个职责完备且均高active_rate；缺少动作能耗约束但当前无证据表明动作浪费是主因；核心问题是高速激励的梯度不够陡。

## 4. selected_level
Level 1：forward_speed_bonus职责正确、符号正确、数学形态（二次）合理，证据表明该组件强度不足以克服速度平台，属于单组件尺度过弱。

## 5. selected_intervention
仅修改`w_fwd2`从0.1提升至0.25，其他三个组件系数保持不变。

## 6. falsifiable_hypothesis
更强的二次系数使高速区域的奖励梯度变陡（v_fwd从2.26到3.0时bonus增量从0.39/步升至0.97/步），应推动策略突破当前中等速度局部最优，向更高v_fwd方向探索，从而缩小与target的差距。

## 7. expected_next_round
forward_speed_bonus的episode_sum_mean和magnitude_share应显著上升，forward_progress也应随速度提升而增加，总score应突破3260向3800靠近，episode应仍保持满长；如果score不变或下降且active_rate保持99%以上，说明二次激励的梯度仍不足以改变步态。

## 8. main_risk
更强的速度激励可能导致策略牺牲稳定性换取速度，躯干倾角和角速度惩罚绝对值可能增大，极端情况下可能出现early termination；若出现此类不稳定，下一轮需考虑同步增强稳定性约束。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    V4 reward: V3 base with stronger quadratic forward speed bonus.
    Increases w_fwd2 from 0.1 to 0.25 to provide steeper gradient
    for higher forward velocities and push past the moderate-speed plateau.
    """
    # Weights
    w_fwd = 1.0          # main linear forward progress
    w_fwd2 = 0.25        # quadratic bonus for higher speeds (increased from 0.1)
    w_angle = 0.5        # torso tilt penalty
    w_angvel = 0.1       # torso angular velocity penalty

    # Main learning signal: forward velocity (clipped to non-negative)
    v_fwd = max(0.0, next_obs[5])
    forward_progress = w_fwd * v_fwd

    # Quadratic bonus: disproportionately rewards higher forward speeds
    forward_speed_bonus = w_fwd2 * (v_fwd ** 2)

    # Stability constraints based on next observation
    torso_angle_penalty = -w_angle * abs(next_obs[1])
    torso_angular_velocity_penalty = -w_angvel * abs(next_obs[7])

    total_reward = forward_progress + forward_speed_bonus + torso_angle_penalty + torso_angular_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "forward_speed_bonus": forward_speed_bonus,
        "torso_angle_penalty": torso_angle_penalty,
        "torso_angular_velocity_penalty": torso_angular_velocity_penalty,
    }

    return float(total_reward), components
```
