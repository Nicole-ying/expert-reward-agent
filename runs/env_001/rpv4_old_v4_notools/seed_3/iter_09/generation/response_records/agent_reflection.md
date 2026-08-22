# Response Record

**1. 诊断报告**

**evidence**
- `score=-110.44`，且 20/20 episodes 均为 early terminal（<150 步且 score<-50），agent 从未完成安全着陆。
- 组件统计显示 `stability_penalty` 100% 活跃且负向主导（ep_sum_mean -0.21），`soft_landing_bonus` 和 `contact_alignment` 活性仅 1.7%，`energy_penalty` 活性 0%—agent 不使用主引擎。
- `approach_delta_gated` 全程 100% 活跃但均值仅 0.07，速度门控极大削弱了进展信号。
- Checkpoint 趋势：score 在 -108~-111 之间几乎无变化，所有组件值平坦，说明无有效学习。
- **关键矛盾**：组件总和约 +0.25，与 score -110 存在数量级缺口。结合 early terminal 和大量负分，推断存在环境内建的巨大负奖励（如坠毁惩罚），自定义奖励本身过小无法覆盖，agent 的最优策略是尽快坠毁以最小化累积惩罚。

**behavior_diagnosis**
Policy 学会了“自杀式坠毁”：由于稀疏且微弱的正向奖励无法补偿每步的姿态惩罚和环境内建的坠毁代价，agent 选择在最短时间内坠毁以终止 episode，从而降低总负奖励。这不是任务期望的行为，而是奖励结构崩溃的表现。

**signal_completeness**
- 缺少稠密且正确的进展信号：`approach_delta_gated` 被速度门控压制，无法鼓励接近。
- 缺少引导安全着陆的终端奖励：`soft_landing_bonus` 触发率极低且量级过小。
- 稳定性惩罚全局施加，在早期探索阶段增加了负向偏置。
- 整体奖励框架使“快速坠毁”成为最优解，与任务目标相反。

**selected_level**
L3 — 分数远低于目标（差距 >300 分）、全部 episode 早终，且当前奖励框架的根本性缺陷（鼓励自杀）无法通过微小调参纠正，必须更换主信号框架。

**selected_intervention**
重写奖励函数，采用 **进展差 + 速度/姿态惩罚 + 阶段门控的巨大着陆奖励** 框架：
1. 主信号：`delta_dist = current_dist - next_dist`（正值代表接近），无门控，提供稠密的进展引导。
2. 速度惩罚：`-w_speed * (|vx|+|vy|)`，抑制高速撞击。
3. 姿态惩罚：`-w_attitude * (|angle| + |ang_vel|)`，保持稳定。
4. 着陆成功奖励：当 `both_legs==1` 且 `speed<0.5` 且 `next_dist<1.0` 时授予 **+100**，以此克服环境的隐式坠毁惩罚并大幅牵引成功率。
5. 接触对齐小奖：`both_legs` 的即时反馈（保留引导）。
6. 去除无效的 energy_penalty（agent 从未使用主引擎，可后续再加）。

**falsifiable_hypothesis**
若主要矛盾是缺乏足够的终端成功激励与稠密进展信号，则新框架将驱动 agent (a) 每步通过降低距离获得正向奖励，(b) 避免高速和大幅度倾转以减少惩罚，(c) 最终在安全着陆时获取巨大奖金，从而显著提升生存率和着陆尝试次数。预计 early terminal 率下降，score 由 -110 向正向移动，可能首次出现部分 episode 成功触地。

**expected_next_round**
- early terminal 降低，episode 长度增加（agent 尝试接近平台而非立即坠毁）。
- 姿态和速度惩罚 active_rate 接近 100%，但 magnitude 可控，不再主导总奖励。
- 着陆成功奖励 active_rate 极低但一集一次即改善平均 score。
- 可能出现少量 episode 到达靠近平台位置，甚至首次安全着陆，score 上限改善。

---

**2. 完整 Python 代码**

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    新主信号：基于距离差值的进展奖励 + 速度/姿态约束 + 阶段门控的着陆大奖
    目标：克服自杀式坠毁，引导 agent 学习接近、减速、稳定着陆
    """
    # ========== 权重 ==========
    w_progress    = 1.0    # 进展信号权重（每步靠近的奖励）
    w_speed       = 0.3    # 速度惩罚系数
    w_attitude    = 0.1    # 姿态/角速度惩罚系数
    w_contact     = 0.2    # 双腿接触即时奖励
    w_landing     = 100.0  # 成功着陆大奖

    # ========== 状态提取 ==========
    # 当前步绝对位置
    x, y = obs[0], obs[1]
    # 下一步
    nx, ny   = next_obs[0], next_obs[1]
    vx, vy   = next_obs[2], next_obs[3]
    angle    = next_obs[4]
    ang_vel  = next_obs[5]
    left_con = next_obs[6]
    right_con = next_obs[7]

    # 当前与下一步距离
    current_dist = (x**2 + y**2) ** 0.5
    next_dist    = (nx**2 + ny**2) ** 0.5

    # ========== 组件计算 ==========

    # 1. 稠密进展：每步靠近目标的正奖励（无门控）
    delta_dist = current_dist - next_dist
    progress_reward = w_progress * delta_dist

    # 2. 速度惩罚：抑制高速移动
    abs_speed = abs(vx) + abs(vy)
    speed_penalty = -w_speed * abs_speed

    # 3. 姿态/角速度惩罚（全局但权重适度）
    attitude_penalty = -w_attitude * (abs(angle) + abs(ang_vel))

    # 4. 双腿接触即时反馈
    both_legs = left_con * right_con  # 0 或 1
    contact_reward = w_contact * both_legs

    # 5. 成功着陆大奖（阶段门控：接触、低速、近心）
    landing_condition = (both_legs > 0.5) and (abs_speed < 0.5) and (next_dist < 1.0)
    landing_bonus = w_landing if landing_condition else 0.0

    # 6. 主引擎燃料惩罚（保留但默认权重为 0，待后续观察再启用）
    main_engine_used = 1.0 if action == 2 else 0.0
    energy_penalty = 0.0 * main_engine_used   # 当前关闭

    # ========== 总奖励 ==========
    total_reward = (
        progress_reward
        + speed_penalty
        + attitude_penalty
        + contact_reward
        + landing_bonus
        + energy_penalty
    )

    components = {
        "progress_reward":   progress_reward,
        "speed_penalty":     speed_penalty,
        "attitude_penalty":  attitude_penalty,
        "contact_reward":    contact_reward,
        "landing_bonus":     landing_bonus,
        "energy_penalty":    energy_penalty,
    }

    return float(total_reward), components
```
