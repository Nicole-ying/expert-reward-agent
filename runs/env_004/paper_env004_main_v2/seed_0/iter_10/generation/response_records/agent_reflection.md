# Response Record

1. **evidence**: iter 9 additive posture penalty (quadratic k=10) fails catastrophically — all 20 episodes terminate early at avg len=223.6, forward_reward dominates with 99.1% signed share (486.6 episode_sum_mean vs posture_penalty -1.37, only 0.3% magnitude share). Best (iter 1) uses multiplicative posture gate `exp(-10*angle²)` and achieves 2926 with full survival (len=1000). The additive structure is fundamentally unable to compete with unbounded forward_vel.

2. **behavior_diagnosis**: The agent learns to tilt aggressively for forward speed, ignoring the negligible posture penalty, and falls within ~224 steps every episode. The quadratic penalty at k=10 produces only ~0.9 at 0.3 rad tilt, dwarfed by forward_vel ~2.0 per step.

3. **signal_completeness**: Forward progress and posture constraint are both present and have correct signs, but the additive posture penalty is structurally disabled by unbounded forward_vel. The multiplicative gate in best solves this by making posture a prerequisite for any forward reward. However, best plateaus at 2926 (77% of target 3800), suggesting room to improve the gate's permissiveness for natural hopping dynamics.

4. **selected_level**: Level 2 — `unbounded_to_bounded` applied to the posture gate function, replacing the exponential decay with a piecewise-linear hinge gate. The additive penalty structure (iter 9) is broken, so starting from best's multiplicative framework is required. Level 1 scale fix on additive penalty cannot overcome the structural dominance of unbounded forward_vel.

5. **selected_intervention**: Single component modification to `speed_posture_reward`: replace `posture_factor = exp(-k*angle²)` with `posture_factor = max(0.0, 1.0 - k*|angle|)` at k=2.0. All other components unchanged from best. This is materially different from every historical iteration, which all used exponential gates when multiplicative.

6. **falsifiable_hypothesis**: The hinge gate provides constant gradient (-k) for posture correction throughout its active range, unlike the exponential gate whose gradient vanishes near zero. This should maintain the survival-enforcing "posture-as-prerequisite" property while allowing slightly more natural torso oscillation during the hopping cycle. At angle=0.3, hinge gives 0.4 (vs exp 0.407 — nearly identical at the danger boundary); at angle=0.15, hinge gives 0.7 (vs exp 0.80 — slightly stronger incentive to tighten posture mid-cycle). If correct, terminated drops to ~0/20, len returns to ~1000, and score moves toward or beyond best (2926).

7. **expected_next_round**: terminated should drop from 20/20 to close to 0/20; episode_length should rise from 223 to near 1000; score should rise from 707 toward 2900+; `speed_posture_reward` should remain the dominant positive component at 95%+ signed share; `ang_vel_penalty` magnitude share should stay modest (<5%).

8. **main_risk**: The hinge gate has a non-differentiable kink at angle=0 and a hard zero-cliff at |angle|=0.5 rad. If the agent operates near the 0.5 rad boundary during aggressive hops, the abrupt reward cutoff could create gradient confusion. Mitigation: the cliff sits at a similar effective threshold as exp(-10·0.5²)=0.082 ≈ effectively zero in best, so the behavioral boundary is comparable.

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    torso_angle = next_obs[1]
    forward_vel = next_obs[5]
    torso_ang_vel = next_obs[7]

    # 姿态门控：线性铰链，替换指数衰减
    # k=2.0, 在|angle|=0.5时归零，与best的exp(-10*0.5²)=0.082行为边界相近
    # 恒定梯度鼓励持续姿态修正，同时在中小角度区域比指数更宽松
    k_angle = 2.0
    posture_factor = max(0.0, 1.0 - k_angle * abs(torso_angle))

    # 主学习信号：前进速度 × 姿态因子
    speed_posture_reward = forward_vel * posture_factor

    # 轻量稳定约束：抑制躯干角速度
    ang_vel_penalty = 0.05 * abs(torso_ang_vel)

    total_reward = speed_posture_reward - ang_vel_penalty

    components = {
        'speed_posture_reward': speed_posture_reward,
        'ang_vel_penalty': ang_vel_penalty
    }

    return float(total_reward), components
```
