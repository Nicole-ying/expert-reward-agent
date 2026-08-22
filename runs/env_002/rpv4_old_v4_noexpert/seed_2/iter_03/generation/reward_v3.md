## 诊断报告

### evidence
- iter1 `score=233.56`（best），`forward_progress_reward` ep_sum=412.14，len=1092。
- iter2 加入 `energy_penalty` 后 `score=187.89`，`forward_progress_reward` 降至 332.11，len 降至 803，`energy_penalty=-34.19`。
- 关键规律：`score / forward_progress_reward` 两轮恒定（0.567 / 0.566），说明外部分数基本由前进速度积分驱动；energy 罚项不进入外部分数，只通过缩短存活与降速间接伤害 score。
- checkpoint 显示 iter2 的 forward 在 cp70% 后平台化（~354-377），score 平台在 ~186-229：前进速度是当前瓶颈。
- 迭代 2 的 energy 罚项使 episode 变短、forward 积分下降，未换来更高效步态，是不健康信号。

### behavior_diagnosis
- 当前策略学到“中等速度、较长存活”步态：平均每步 forward 分量约 0.38-0.41，len 803-1090，20/20 摔倒终止。
- 该步态稳定但太慢。要达 300 分需 forward_sum≈530，而 iter1 仅 412，继续拉长存活空间有限，**速度是瓶颈**。
- 速度平台源于 `v/sqrt(v²+1)` 在 v>0.4 后边际收益快速衰减，缺少对更快前进的持续梯度。

### signal_completeness
- `forward_progress`：职责在，但归一化速度曲线饱和，缺少高速梯度。
- `upright_stability`：职责在，但 `angle_weight=1.0` 太弱——1.0 rad 倾斜每步仅罚 0.25，小于一步 forward 收益，无法有效阻止摔倒。
- `energy_efficiency`：iter2 证实有害且不进入外部分数，应移除。

### selected_level
- **L1**：只调系数/权重，不改数学表达式。forward 归一化公式与 upright 死区二次罚方向都对，只是系数配错。

### selected_intervention
- 删除 `energy_penalty` 组件。
- `forward`：`speed_scale` 1.0 → 3.0，`forward_weight` 1.0 → 3.0 —— 在保持当前速度下奖励量级近似不变的同时，延后饱和点，给高速更大梯度。
- `upright`：`angle_weight` 1.0 → 5.0 —— 使 1.0 rad 倾斜罚约 1.25/步，超过单步前进收益，真正形成摔倒约束。

### falsifiable_hypothesis
- 移除 energy 后 episode 不再被无谓压缩，forward 积分可恢复并超过 iter1；去饱和后策略有动力学出更快步态；加强的 angle 罚则约束更快步态下的摔倒风险。

### expected_next_round
- len 回升至 900+，forward ep_sum 突破 iter1 的 412 并继续上升，score 向 250+ 逼近；upright 罚项略升但由 angle 主导，说明更快步态被有效约束而非摔倒。

## 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- forward progress (primary signal) ---
    # Desaturate the normalized speed curve: larger speed_scale pushes the
    # saturation point to higher speeds, keeping gradient alive for faster gaits.
    # forward_weight is raised together so the reward magnitude at current
    # speed (~0.4) stays roughly unchanged.
    forward_weight = 3.0
    speed_scale = 3.0
    forward_speed = 0.5 * (obs[2] + next_obs[2])
    forward_progress_reward = (
        forward_weight
        * forward_speed
        / ((forward_speed ** 2 + speed_scale ** 2) ** 0.5)
    )

    # --- upright stability (health constraint) ---
    # Strengthen the angle penalty: at ~1.0 rad tilt the old weight (1.0) only
    # cost ~0.25/step, less than one forward step, so falls were cheap.
    angle_deadband = 0.5
    angvel_deadband = 1.5
    angle_weight = 5.0
    angvel_weight = 0.05
    eps = 1e-6

    avg_angle = 0.5 * (obs[0] + next_obs[0])
    avg_angvel = 0.5 * (obs[1] + next_obs[1])

    angle_raw = abs(avg_angle) - angle_deadband
    angle_excess = 0.5 * (angle_raw + (angle_raw ** 2 + eps) ** 0.5)
    angle_penalty = angle_weight * angle_excess ** 2

    angvel_raw = abs(avg_angvel) - angvel_deadband
    angvel_excess = 0.5 * (angvel_raw + (angvel_raw ** 2 + eps) ** 0.5)
    angvel_penalty = angvel_weight * angvel_excess ** 2

    upright_stability_penalty = -(angle_penalty + angvel_penalty)

    # energy penalty removed: it shortened episodes (1092 -> 803) and cut
    # forward integration without producing a more efficient gait.

    total_reward = forward_progress_reward + upright_stability_penalty

    components = {
        "forward_progress_reward": float(forward_progress_reward),
        "upright_stability_penalty": float(upright_stability_penalty),
    }

    return float(total_reward), components
```