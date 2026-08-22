## 诊断报告

**evidence**
- 全局记忆：iter_1 `score=296.08`（目标 300，差距 1.3%），`len=1173.6`；iter_2 `score=237.50`，`len=1050.75`，delta=-58.58。
- iter_1 组件：`forward_progress` ep_sum=504.92、magnitude_share=95.5%；`torso_stability`=-23.70（4.5%）；`vertical_velocity_penalty`=-0.27。
- iter_2 组件：`forward_progress` ep_sum=555.11、magnitude_share=96.4%；`torso_stability`=-20.52（3.6%）。iter_2 是加强了 forward（1.0/1.5 → 1.2/1.8）的结果。
- checkpoint 趋势：iter_2 的 score 在 cp70=272 → cp80=243 → cp90=285 → cp100=272，出现中后期塌陷；而 forward_progress 在 cp70 后反而持续高企（585→550→605→599）。这是速度信号被过度利用的典型迹象。
- 两个迭代的 `terminated=20/20`、无早期崩溃（early_terminal=0/20），说明不是初始摔倒问题，而是速度驱动过强导致更激进步态、更早终止。

**behavior_diagnosis**
当前策略被 `forward_progress` 单一信号驱动（magnitude_share≈96%），学到的是一种“跑得快但终止早”的步态。iter_2 加强速度奖励后，内部 forward 累计值上升（505→555），但外部得分下降（296→237）、episode 长度缩短（1173→1050），说明原始速度 proxy 已经过强，与“走得远、走稳”的外部目标不再同向，属于不健康的 exploit 方向。

**signal_completeness**
必要职责完整：`forward_progress`（主前进信号）、`torso_stability`（防摔倒）都在。缺失的绝对位置与成功标志在环境中不可用，无法补充。当前不是缺信号，而是**权重失衡**：速度信号过大，稳定性信号相对过弱。

**selected_level**
L1。当前 best=296.08，距目标 300 差距 < 10%，属于接近目标区间；iter_2 已经证明在同组件上加强力度会恶化，因此本轮只做系数微调，不改数学形态。

**selected_intervention**
与上一轮“把 forward 还原回 iter_1”的方向不同，本轮采取相反语义：**降低 forward_progress 的系数至 iter_1 以下**（正向 1.0→0.8，负向 1.5→1.2），同时**小幅加强 torso_stability**（-0.15/-0.05 → -0.20/-0.07），共同服务于“抑制速度过度利用、减少激进步态导致的早终止”这一目标。vertical_velocity_penalty 保持不变。

**falsifiable_hypothesis**
若 iter_2 的 -58.58 分主要来自速度信号过强导致的激进步态/早终止，那么把 forward 权重压到 iter_1 以下、并小幅提高稳定性惩罚，应使策略转向更保守稳定的步态：`forward_progress` ep_sum 预计降至 ~400 附近，magnitude_share 略降到 ~93-94%，`torso_stability` 占比升至 ~6-7%，episode 长度回升到 iter_1 水平或更长，外部得分应 ≥ iter_1（理想 ≥300）。若得分反而下降，说明外部目标需要的是更多而非更少速度激励，下一轮应反向调整。

**expected_next_round**
策略不再追求极限水平速度，步态更稳、更持久；`len` 预计 ≥1173，`forward_progress` 累计值下降但外部 score 上升；训练 checkpoint 趋势应比 iter_2 更平滑，不再出现 cp80 附近的中后期塌陷。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # Forward progress: encourage positive horizontal velocity, penalize backward motion.
    # L1: iter_2 showed that 1.2/1.8 over-weights raw speed (agent walks faster but
    # terminates earlier and drops external score). Pull the coefficient BELOW iter_1's
    # 1.0/1.5 to suppress speed exploitation while keeping forward motion rewarded.
    if horizontal_velocity >= 0.0:
        forward_progress = 0.8 * min(horizontal_velocity, 2.0)
    else:
        forward_progress = 1.2 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    # Slightly strengthen torso stability so the less speed-driven gait stays upright
    # longer and does not trade stability for raw forward velocity.
    torso_stability = -0.20 * (hull_angle ** 2) - 0.07 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    total_reward = forward_progress + torso_stability + vertical_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
    }

    return float(total_reward), components
```