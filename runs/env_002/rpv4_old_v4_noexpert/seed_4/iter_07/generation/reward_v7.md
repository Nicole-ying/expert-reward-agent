## 1. 诊断报告

**evidence**
- `read_memory()`：iter_6 得分 **286.16**（历史最佳），iter_1 得分 **278.99**。iter_6 的奖励代码与 iter_1 唯一差异是 `forward_progress` 增益从 2.0 → 2.1（`upright_stability` 形态完全不变）。
- `read_training_feedback(1)` vs `(6)`：
  - iter_1：`forward_progress ep_sum=697.4`，len=1275.6 → 单步 forward=0.547，反推 **v≈0.376 m/s**；`upright_stability=-6.68`（近乎完全直立）。
  - iter_6：`forward_progress ep_sum=745.4`，len=1223.6 → 单步 forward=0.609，反推 **v≈0.409 m/s**；`upright_stability=-60.53`。
  - 结论：+0.1 增益把步态从「慢但极稳」推进到「更快但略前倾/抖动」，**净收益 +7.17 分**，是迭代中唯一被实证过的有效改进方向。
- `get_component_history('forward_progress')`：iter_6 的 `forward_progress` 是历史最高的正贡献来源，且 `magnitude_share=92.5%`，是主信号。
- iter_3 教训：增益 2.5（且叠加 energy penalty）时崩溃到 240.53，表现为慢速（v≈0.331）却大幅前倾（upright -72.99）。当前 2.1 → 2.2 仍远离 2.5 的 exploit 阈值。
- 目标 300，当前 286.16，差距 **13.84（4.6%）**，< 10% → 按规则优先 L1 微调。

**behavior_diagnosis**
当前策略学到的是一套健康的前向双足步态：v≈0.409 m/s、len≈1224、20/20 正常终止、无 early-terminal（<150 步且 score<-50 的为 0）、无 reward_errors。相比 iter_1，它以「适度消耗稳定性预算」换取了更高速度，score 因此创新高。该行为不是病态的（没有前倾 exploit、没有原地抖动），但稳定性预算已被用到 -60.5，进一步大尺度加码有重演 iter_3 前倾 exploit 的风险。

**signal_completeness**
- 已具备主职责 `forward_progress`（饱和前向速度）与 `upright_stability`（躯干倾角 + 躯干角速度），二者 active_rate 均 100%，非死组件。
- 能量效率项在 iter_2/3 实证为负贡献，已被正确移除，不补回。
- 环境无显式摔倒/完成 flag，当前间接稳定性约束是合理替代。
- 信号完备，无需新增职责。

**selected_level**：L1
理由：当前得分距目标仅 4.6%，且 iter_6 已证明「增益微调」方向有效；框架无需改数学形态，只做系数微调，避免重演接近目标时大改导致变差的历史教训。

**selected_intervention**
目标组件 `forward_progress`：增益 **2.1 → 2.2**，其余（`upright_stability` 的倾角/角速度惩罚系数、饱和形态）全部保持不变。

**falsifiable_hypothesis**
iter_1→iter_6 的唯一变量实验显示：`+0.1` 增益 → 速度 0.376→0.409 → score +7.17。再 `+0.1` 应使前向速度的边际价值略增，把「速度-稳定性」最优点推向略快的步态，从而提升前向距离、推动 score 向 293–296 靠近；且 2.2 远低于 iter_3 触发前倾 exploit 的 2.5，预计不会跨过稳定性崩溃阈值。

**expected_next_round**
score 预计 **293–296**；v 预计 **0.42–0.44 m/s**；len 预计稳定在 **1200–1240**；`upright_stability` 可能小幅恶化（约 -65~-75），但不会出现 early-terminal 或 reward_errors，仍 20/20 正常终止。

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前向速度，保持 iter_6 验证过的纯加性饱和形态。
    # 证据：iter_1（2v/(1+|v|)）=278.99；iter_6 把增益 2.0→2.1 后速度
    # 0.376→0.409、score 278.99→286.16，是本迭代唯一被实证的正向改进。
    # 本轮继续同一方向做 L1 微调：增益 2.1→2.2，给前向速度更强的边际激励，
    # 同时仍远低于 iter_3 触发前倾 exploit 的 2.5，避免稳定性崩溃。
    horizontal_velocity = next_obs[2]
    forward_progress = 2.2 * horizontal_velocity / (1.0 + abs(horizontal_velocity))

    # 稳定性约束：保持 iter_1/iter_6 已验证的形态不变（躯干倾角 + 躯干角速度）。
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