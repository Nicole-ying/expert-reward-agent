## 诊断

### 0. 理解环境
这是一个 2D 着陆器任务：从顶部出发，飞到中央着陆垫，以低速、直立姿态、双腿触地稳定降落。观测 8 维，动作 4 离散（无推力/左姿态/主引擎/右姿态）。

### 1. 发生了什么？

| 指标 | iter 1 (best) | iter 2 (current) |
|---|---|---|
| score | -108.90 | -566.80 |
| len | 68.45 | 114.90 |
| progress_delta ep_sum_mean | +0.016 | **-0.127** |
| stability_penalty ep_sum_mean | -0.138 | -0.167 |
| soft_landing_proxy active_rate | ~0% | **0%** |
| early_terminal (<150步, <-50分) | ? | **17/20** |

- **迭代 2 剧烈退化**：唯一改动是给 stability_penalty 加了距离门控（远处关闭惩罚）。但门控移除后，agent 在远处失去了速度/姿态约束，progress_delta 从正向（+0.016）翻转为负向（-0.127），说明 agent 在远离目标而非靠近。
- **soft_landing_proxy 从未触发**（两轮 active_rate 都接近 0%）：二元条件过于严苛，无法提供学习信号。
- **根本问题**：progress_delta 是唯一的引导信号，但它无界且噪声大；stability_penalty 约束全局生效但主导了总奖励；landing proxy 完全稀疏，对策略无贡献。

### 2. 哪个组件最值得干预？

**soft_landing_proxy**。它是唯一可能提供任务完成引导的组件，但 active_rate=0% 意味着它不存在于策略的学习信号中。progress_delta 无法区分"靠近目标"和"着陆条件满足"，stability_penalty 只约束不引导。将 landing proxy 从稀疏二元转为稠密连续信号是核心突破口。

### 3. 我之前改了什么？

上一轮给 stability_penalty 加了距离门控（global_to_local_gate），导致 score 从 -108.90 暴跌至 -566.80。**必须 revert 到 best 代码，然后做一个新修改**——本次修改聚焦 soft_landing_proxy。

---

## 修改方案

- **evidence**：soft_landing_proxy active_rate=0%，二元条件从未满足；progress_delta 信号弱且噪声大。
- **diagnosis_dimension**：**稀疏性** — 任务完成信号完全不存在，没有任何梯度引导 agent 朝向着陆条件。
- **selected_transformation**：**sparse_to_dense** — 将二元 if 条件替换为连续 bounded 因子（proximity × speed × upright × contact 的加权组合），使每步都有可区分的引导信号。
- **falsifiable_hypothesis**：连续 landing proxy 会给 agent 提供朝向目标的渐进式反馈，使其学会减速、调姿、接近，从而提升外部 score 并降低 early termination。
- **expected_next_round**：soft_landing_proxy active_rate 大幅上升（接近 100%），episode_sum_mean 为正且非零；progress_delta 可能改善（不再为负）；early_terminal 比例下降；外部 score 改善。
- **main_risk**：连续 proxy 可能让 agent 学会"靠近目标但不完成着陆"（proxy exploitation），即徘徊在目标附近刷分而不真正着陆。这需要下一轮根据行为证据判断是否需要 dense_to_task_event 收紧。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测与下一观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：progress delta ----------
    def dist(px, py):
        return (px**2 + py**2)**0.5

    prev_dist = dist(x, y)
    next_dist = dist(next_x, next_y)
    progress_delta = prev_dist - next_dist    # 正值 = 靠近目标

    # ---------- 稳定/安全约束：stability penalty ----------
    speed_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    angvel_penalty = abs(angular_vel)
    stability_penalty = -0.1 * speed_penalty - 0.1 * angle_penalty - 0.05 * angvel_penalty

    # ---------- 稠密着陆引导：continuous soft landing proxy ----------
    # 将原来的二元 if 条件替换为连续 bounded 因子
    # 每个因子 ∈ [0, 1]，提供渐进式反馈

    # 距离因子：越靠近目标越好
    proximity_factor = max(0.0, 1.0 - next_dist / 1.0)

    # 速度因子：越慢越好（使用欧氏速度）
    speed = (vx**2 + vy**2)**0.5
    speed_factor = max(0.0, 1.0 - speed / 0.5)

    # 姿态因子：越正越好
    upright_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    # 接触因子：双腿着地更好
    contact_factor = (left_contact + right_contact) / 2.0

    # 综合着陆质量：三个核心因子取几何平均（要求同时满足），
    # 接触因子作为额外加权（避免乘积塌缩到零）
    # 几何平均防止单一因子优秀时掩盖其他因子的不足
    core_quality = (proximity_factor * speed_factor * upright_factor) ** (1.0 / 3.0)
    soft_landing_proxy = 0.5 * core_quality * (0.3 + 0.7 * contact_factor)

    # ---------- 合成 ----------
    total_reward = progress_delta + stability_penalty + soft_landing_proxy

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_proxy': soft_landing_proxy,
    }
    return float(total_reward), components
```