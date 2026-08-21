## 1. 诊断报告

**evidence**
- `read_memory()`：iter_1（纯加性 `forward_progress=2v/(1+|v|)` + `upright_stability`）是历史最佳 **278.99**；本轮 iter_5（乘性直立门控）只有 **271.21**，仍低于 best 7.78 分。
- `read_training_feedback(1)` vs `(5)`：
  - iter_1：`forward_progress ep_sum=697.4`，len=1275.6，反推 `v≈0.376`；`upright_stability=-6.68`。
  - iter_5：`forward_progress ep_sum=660.6`，len=1109.5，反推 `v≈0.430`；`upright_stability=-6.85`。
  - 两者 upright 几乎一样稳定，但 iter_5 更快、回合更短，导致**累计 forward_progress 反而更低**，外部 score 也更低。
- `get_component_history`：iter_1→iter_5 的 forward_progress ep_sum 依次为 697.4 / 720.9 / 861.5 / 718.6 / 660.6；upright_stability 依次为 -6.7 / -43.2 / -73.0 / -45.6 / -6.85。说明门控确实把 iter_4 的前倾 exploit 压回稳定区，但没有超过 iter_1。
- `read_past_reflection(5)` + `read_reward_code(5)`：门控是为了修复 iter_4 线性 clip 造成的 `θ≈0.172` 前倾 exploit。但 iter_1 的饱和形态本身就从未出现过该 exploit，门控属于对一个不存在问题的过度修正。

**behavior_diagnosis**
- 当前策略（iter_5）行为是：躯干稳定（upright≈-6.85）、速度较快（v≈0.43）、回合偏短（len≈1109）。不是摔倒/抖动/前倾等病态模式，但「更快却更短」使得累计前向奖励和外部得分都低于 iter_1 的「稍慢但更长」步态。整体是次优的速度-稳定性权衡，不是灾难性失败。

**signal_completeness**
- 必需职责完备：`forward_progress`（主信号）+ `upright_stability`（防摔倒）都在，且 active_rate 100%。
- 能量项在 iter_2/3 已实证为负贡献（拖慢步态、恶化稳定性），继续缺省是合理的。
- 不缺必要信号；缺的是把 iter_5 引入的「速度与躯干角乘性耦合」去掉，恢复 iter_1 已验证的干净加性梯度。

**selected_level**
- **L1**。best=278.99，距目标 300 差距约 7%（<10%），按规则优先 L1；且历史经验显示：接近目标时做 L2（iter_2 加能量项、iter_3 调增益+能量项、iter_4 换线性形态、iter_5 加乘性门控）全部 ≤ iter_1 的 278.99。

**selected_intervention**
- 目标组件：`forward_progress`。
- 具体修改：**删除乘性直立门控 `upright_gate=1/(1+3θ²)`**，恢复 iter_1 的纯加性饱和形态 `2v/(1+|v|)`；同时把前向增益从 2.0 微调到 **2.1**，给一个温和、仍有界的向前推动（远低于 iter_3 触发 exploit 的 2.5）。
- `upright_stability` 保持 iter_1 已验证的形态不变。

**falsifiable_hypothesis**
- 门控把 `forward_progress` 与 `torso_angle` 乘性耦合，训练中形成了「θ 增大→速度奖励被压低」的非平稳梯度，使策略收敛到「更快但更短」的次优步态。移除门控后，恢复 iter_1 的干净加性梯度，策略应回到「稍慢但更长、累计前向奖励更高」的更优步态；+5% 增益在饱和形态下边际梯度仍然递减，不会重蹈 iter_3/iter_4 的前倾 exploit。
- 预计本轮 score 至少回到 iter_1 水平（≈279），并有机会靠温和增速小幅突破。

**expected_next_round**
- 策略应学到躯干稳定（upright≈-6~-7）、速度约 0.38–0.40、回合长度回升到 1200+ 的步态；`forward_progress` 累计值回升，score 预计 ≥279，向 300 靠近。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度，恢复 iter_1 验证过的纯加性饱和形态。
    # 证据：iter_1（2v/(1+|v|) 无门控）=278.99 为历史最佳；
    # iter_5 加了乘性直立门控后 upright 虽稳定（-6.85），但速度被推到 0.43、
    # len 从 1275 掉到 1109，forward_progress ep_sum 从 697 降到 660，score 271 < iter_1。
    # 说明饱和形态本身不需要门控；门控扭曲了速度-稳定性权衡。
    # 这里移除门控，并把增益从 2.0 微调到 2.1，给温和且仍有界的向前推动
    # （远低于 iter_3 触发前倾 exploit 的 2.5）。
    horizontal_velocity = next_obs[2]
    forward_progress = 2.1 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    # 稳定性约束：保持 iter_1 已验证的形态不变（躯干倾角 + 躯干角速度）。
    torso_angle = next_obs[0]
    torso_angle_sq = torso_angle * torso_angle
    angle_penalty = 1.5 * torso_angle_sq / (1.0 + torso_angle_sq)

    torso_angular_velocity = next_obs[1]
    torso_angular_velocity_sq = torso_angular_velocity * torso_angular_velocity
    angular_velocity_penalty = 0.5 * torso_angular_velocity_sq / (1.0 + torso_angular_velocity_sq)

    upright_stability = -(angle_penalty + angular_velocity_penalty)

    total_reward = forward_progress + upright_stability

    components = {
        "forward_progress": forward_progress,
        "upright_stability": upright_stability,
    }

    return float(total_reward), components
```