# 分析与干预

## 1. 证据与诊断

`evidence`：iter 5（纯forward_reward）得分仅7.35，episode_length=13.5，所有20条轨迹均因摔倒终止。组件统计显示forward_reward均值仅0.11，active_rate=17.4%。历史最佳（iter 1-2）得分2663.09、满1000步，使用乘性门控 `forward_vel * exp(-|angle|/0.3)` + 轻量稳定性惩罚，但iter 1与iter 2得分完全相同，表明策略收敛到同一局部最优。iter 3和iter 4尝试添加垂直分量均导致严重退化。

`behavior_diagnosis`：纯前向奖励下agent几乎立即摔倒——缺乏任何直立信号导致策略无法学习基本平衡。最佳奖励中乘性门控将前向速度与躯干角度严格绑定：机器人跳跃时必须略微前倾以获得推进力，但前倾会通过`exp(-|angle|/0.3)`大幅削减前向奖励，形成鸡生蛋问题，可能迫使策略退化为保守的"小碎步"而非真正的跳跃步态。

`signal_completeness`：最佳奖励已具备前向引导（forward_vel）和稳定约束（torso_angle + torso_ang_vel），职责基本完备。但乘性耦合使两个必要信号互相抑制——跳跃所需的前倾被前向奖励的门控惩罚，信号在结构上互相矛盾，而非独立可达。

## 2. 干预层级选择

`selected_level`：**Level 2** —— 有方向的数学结构变换。触发条件：乘性门控结构可能限制了跳跃步态的探索，证据为iter 1/2得分停滞且iter 3/4添加独立信号失败。变换类型：**multiplicative_gate → additive_independent**，将前向奖励与直立奖励从乘性耦合改为加性独立。

## 3. 干预设计

`selected_intervention`：唯一目标组件是前向-直立关系的数学形态。将 `forward_reward = forward_vel * exp(-|angle|/0.3)` 拆分为两个独立加性组件：(1) `forward_reward = max(0, forward_vel)` 无条件奖励前进，(2) `upright_bonus = exp(-|angle|/0.5)` 独立奖励直立。保持稳定性惩罚不变。

`falsifiable_hypothesis`：解耦后agent可以在跳跃推地阶段短暂前倾（获得forward_reward但upright_bonus暂时下降），随后恢复直立（upright_bonus恢复），整个周期净奖励为正。这应使agent能够探索真正的跳跃步态，而非只能选择永远直立的小碎步。

`expected_next_round`：episode_length应恢复至接近1000（不再早期摔倒）；forward_reward的episode_sum_mean应显著高于best的1.176（因不再被门控削减）；upright_bonus应维持在较高水平表明整体直立良好；外部score有望超越2663。

`main_risk`：加性结构中forward_reward不再受直立度调制，极端情况下agent可能学会"前倾冲刺→摔倒"的短暂高分策略（早期终止但瞬时速度极高）。稳定性惩罚和upright_bonus需提供足够制衡。若出现此模式，下一轮需强化稳定性约束或引入bounded forward_reward。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant signals from next_obs (outcome of action)
    torso_angle = next_obs[1]          # rad, 0 = upright
    forward_vel = next_obs[5]          # positive = forward
    torso_ang_vel = next_obs[7]        # rad/s

    # Forward progress: reward positive forward velocity unconditionally
    forward_reward = max(0.0, forward_vel)

    # Uprightness bonus: independent additive reward for staying near-upright
    temp = 0.5
    upright_bonus = 2.718281828 ** (-abs(torso_angle) / temp)

    # Stability penalty: light quadratic penalty on tilt and angular velocity
    stability_penalty = -0.1 * (torso_angle ** 2) - 0.01 * (torso_ang_vel ** 2)

    total_reward = forward_reward + upright_bonus + stability_penalty

    components = {
        'forward_reward': forward_reward,
        'upright_bonus': upright_bonus,
        'stability_penalty': stability_penalty
    }

    return float(total_reward), components
```