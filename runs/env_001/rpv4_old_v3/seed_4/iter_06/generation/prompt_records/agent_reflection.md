# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的工作是：根据 user prompt 中的调查指引，自主查阅历史记录，诊断问题，做出一个语义方向上的修改，输出新代码。

# 可用工具

以下工具随时可用，在诊断过程中主动调用：

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action。快速掌握全貌。
- `get_component_history(component_name)` — 某个组件在所有迭代中的 ep_sum_mean 和 active_rate 演化轨迹。
- `read_training_feedback(iteration)` — 某轮训练的详细反馈：精确分数、每个组件的 ep_sum/active_rate/magnitude_share。
- `read_past_reflection(iteration)` — 你过去某轮的诊断和干预记录。动手前查询以避免重复失败的修改。
- `read_reward_code(iteration)` — 某轮的完整奖励函数代码。对比代码差异时使用。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。

# 输出

输出两个部分：

**1. 诊断报告**（在代码之前）：

- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的失败行为
- `signal_completeness`：必要职责是否完备，缺什么信号
- `selected_level`：L1 / L2 / L3 及选择理由
- `selected_intervention`：目标组件及具体修改内容
- `falsifiable_hypothesis`：为什么该修改应改善策略
- `expected_next_round`：下一轮哪些指标应如何变化
- `main_risk`：最可能引入的新问题

**2. 完整 Python 代码**（```python ... ``` 代码块）。

```

## User Prompt

```markdown
第 5 轮迭代训练已经结束，得分 -40.666010（目标: 200）。

**你需要自主调查、诊断、并做出一个语义方向上的修改。**

### 调查清单（按优先级）

1. **先看本轮反馈**——得分离目标还有多远？每个组件的激活率正常吗（死组件 < 2%，全覆盖惩罚 > 95% 且负贡献）？有没有哪个组件 magnitude_share > 90% 主导了所有信号？Checkpoint 在训练过程中是否有改善趋势？

2. **查上一轮你做了什么**——用 `read_past_reflection(5)` 看你的诊断和干预。用 `get_component_history` 查你改的那个组件，它的 active_rate / ep_sum 变化是否符合你的预期？如果不符合——你的上一轮诊断可能错了，先搞清楚为什么。

3. **查全局历史避免打转**——用 `read_memory()` 看所有迭代。**关键判断：你是不是在同一个语义功能上反复修改但无突破？**（注意：组件名字可能变了，但语义功能相同——比如 success_landing_bonus 和 success_bonus 都是"着陆成功信号"。）如果某个语义已经改了 2-3 轮都不好，不要再改它了——换一个语义方向。

4. **参考 best code 但要验证**——用 `read_reward_code(best_iter)` 看历史最佳奖励函数的代码。它的哪些组件可能促成了当时的最高分？但不要盲目照搬——结合 best 那轮的 feedback（`read_training_feedback(best_iter)`）看：它的得分分布、终止率、组件 activation 是否健康。best 只是记录最高分，不一定代表正确的引导方式。

5. **Checkpoint 是训练过程的证据**——如果某个组件在 checkpoint 40%→80% 期间 ep_sum 持续为 0 或不变，说明它没有提供有效梯度，agent 在训练中忽略了它。

### 修改原则

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态（稀疏→连续、状态值→改善量、均匀→高度感知等）。适用于组件的引导逻辑有问题（如均匀惩罚导致无近地减速梯度）。
- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。
- **⚠️ 组件被 exploit 时必须干预**：如果一个组件 magnitude_share > 90%，即使它刚被激活，也必须修改它——加触发条件限制、加每 episode 次数上限、或收紧阈值。"刚激活所以只做 L1"不适用于 exploit 场景。
- **避免打转**：如果 `read_past_reflection` 显示你之前已经改过同一语义且结果不好，换方向。

### 你需要的信息都在这里

实验记录位于 `rpv4_old_v3/seed_4/`，你可以随时用工具访问：
- `read_memory()` — 全局记忆表（所有迭代的 skeleton/score/best/action）
- `read_training_feedback(N)` — 任意历史迭代的详细训练反馈
- `read_reward_code(N)` — 任意历史迭代的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某个组件在所有迭代中的演化轨迹（注意：历史迭代中同一语义可能用不同组件名，如 progress_shaping→progress，action_cost→fuel_penalty。如果查不到，试试同语义的旧名字）
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件

现在开始调查。

# 上一轮训练反馈 (iter_5, score=-40.666010)
# Training Feedback

## Final-policy outcome
score=-40.666010, len=991.500000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-134.657520, 7.001501]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| pos | 8.455809 | 63.6% | 72.5% | 100.0% |
| contact | 2.287864 | 17.2% | 17.2% | 0.1% |
| vel | -0.903246 | -6.8% | 6.8% | 100.0% |
| angvel | -0.239863 | -1.8% | 1.8% | 100.0% |
| angle | -0.227131 | -1.7% | 1.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# Checkpoint 评估
### Overall score
| pct | score_mean | score_std |
|---|---|---|
| 40% | -96.65 | 0.00 |
| 80% | -54.47 | 0.00 |

### Per-component ep_sum_mean at each checkpoint
| pct | angle (ep_sum) | angvel (ep_sum) | contact (ep_sum) | pos (ep_sum) | vel (ep_sum) |
|---|---||---||---||---||---|
| 40% | -0.797 | -0.777 | 50.450 | 4.663 | -1.613 |
| 80% | -0.338 | -0.324 | 2.291 | 6.690 | -1.316 |

### Per-component active_rate at each checkpoint
| pct | angle (active%) | angvel (active%) | contact (active%) | pos (active%) | vel (active%) |
|---|---||---||---||---||---|
| 40% | 100.0% | 99.8% | 0.5% | 100.0% | 100.0% |
| 80% | 100.0% | 99.9% | 0.4% | 99.9% | 99.9% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs indices:
    # 0: x_position (relative to pad center)
    # 1: y_position (relative to pad height)
    # 2: x_velocity
    # 3: y_velocity
    # 4: body_angle
    # 5: angular_velocity
    # 6: left_support_contact (0.0 or 1.0)
    # 7: right_support_contact (0.0 or 1.0)

    ox = obs[0]
    oy = obs[1]
    nx = next_obs[0]
    ny = next_obs[1]
    nvx = next_obs[2]
    nvy = next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft = next_obs[6]
    nright = next_obs[7]

    # Distances
    prev_dist = (ox**2 + oy**2 + 1e-8) ** 0.5
    next_dist = (nx**2 + ny**2 + 1e-8) ** 0.5

    # -- 1) approach progress: reward reduction in distance to pad --
    w_pos = 8.0
    pos_reward = w_pos * (prev_dist - next_dist)   # >0 when approaching, <0 when retreating

    # -- 2) soft landing: speed & attitude penalty gated by distance --
    k_gate = 10.0
    gate = 1.0 / (1.0 + k_gate * next_dist)         # near pad -> gate ~1, far away -> gate ~0

    w_vel = 0.5
    vel_penalty = -w_vel * (nvx**2 + nvy**2) * gate

    w_angle = 0.5
    angle_penalty = -w_angle * (nangle**2) * gate

    w_angvel = 0.5
    angvel_penalty = -w_angvel * (nangvel**2) * gate

    # -- 3) safe contact: reward each leg touching near center (gradual) --
    contact_proxy = nleft + nright                    # 0, 1, or 2: gradual (was binary nleft*nright)
    contact_threshold = 0.5                           # wider reach (was 0.3)
    contact_factor = max(0.0, 1.0 - next_dist / contact_threshold)
    w_contact = 6.0                                   # adjusted: old max=10, new max=12
    contact_reward = w_contact * contact_proxy * contact_factor

    total_reward = pos_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        'pos': pos_reward,
        'vel': vel_penalty,
        'angle': angle_penalty,
        'angvel': angvel_penalty,
        'contact': contact_reward
    }

    return total_reward, components
```

# Subagent 数据报告
# 训练数据报告 — iter_05

---

## 1. 行为事实
- 得分：score=-40.67，range=[-134.66, 7.00]
- 终止方式：terminated 1/20（成功着陆），truncated 19/20（时间上限）
- 回合长度：len=991.50
- reward_errors=0

### Checkpoint 得分轨迹
| 进度 | 得分 |
|------|------|
| 40% | -96.65 |
| 80% | -54.47 |
| 40%→80% | 上升 +42.18 |

---

## 2. 骨架演进表

| iter | score | len | term (T/t) | pos ep_sum | pos active% | contact ep_sum | contact active% | vel ep_sum | vel active% | angle ep_sum | angle active% | angvel ep_sum | angvel active% |
|------|-------|-----|------------|------------|-------------|-----------------|------------------|------------|-------------|--------------|--------------|---------------|-----------------|
| 1 | 26.74 | 341.75 | 15/5/0 | -88.13 | 100.0% | 555.99 | 31.9% | -5.30 | 100.0% | -0.44 | 100.0% | -1.24 | 92.7% |
| 2 | 64.24 | 924.05 | 10/10/0 | 3.55 | 99.9% | 305.31 | 12.0% | -1.50 | 99.8% | -0.55 | 100.0% | -0.37 | 99.2% |
| 3 | -21.40 | 1000.00 | 0/20/0 | 59.09 | 100.0% | 0.00 | 0.0% | -1.08 | 100.0% | -0.24 | 100.0% | -0.18 | 100.0% |
| 4 | 24.69 | 754.05 | 10/10/0 | 7.91 | 100.0% | 5.37 | 0.5% | -1.68 | 100.0% | -0.45 | 100.0% | -0.31 | 99.9% |
| 5 | -40.67 | 991.50 | 1/19/0 | 8.46 | 100.0% | 2.29 | 0.1% | -0.90 | 100.0% | -0.23 | 100.0% | -0.24 | 100.0% |

*第三列终止方式：terminated/truncated/early_terminal*

---

## 3. 数字标记

### 本轮组件状态
| 标记类型 | 组件 | 数值 | 判定条件 |
|----------|------|------|----------|
| **死组件** | contact | active_rate=0.1% | < 2% |
| — | pos | active_rate=100.0% | 正常 |
| — | vel | active_rate=100.0% | 正常 |
| — | angle | active_rate=100.0% | 正常 |
| — | angvel | active_rate=100.0% | 正常 |

无主导组件、无淹没组件、无全覆盖惩罚。

### 量级对比（每步均值 = ep_sum ÷ len）
- 最大：pos = 8.46 ÷ 991.5 = **0.00853**
- 次大：angle = -0.227 ÷ 991.5 = **-0.000229**（以绝对值计）
- contact = 2.29 ÷ 991.5 = 0.00231（以绝对值计但 active_rate=0.1%，实际仅少数步触发）
- 次小：angvel = -0.240 ÷ 991.5 = **-0.000242**（以绝对值计）
- 最小：vel = -0.903 ÷ 991.5 = **-0.000911**（以绝对值计）

注：contact 虽 ep_sum=2.29，但 active_rate 仅 0.1%，其每步均值在触发步上极高（约 2.29），在非触发步上为 0。

### 修改变化（vs iter_04）
| 组件 | Δ ep_sum | Δ active_rate |
|------|----------|---------------|
| pos | +0.54 | 0.0pp |
| contact | -3.08 | -0.4pp |
| vel | +0.78 | 0.0pp |
| angle | +0.23 | 0.0pp |
| angvel | +0.07 | +0.1pp |

---

## 4. 语义演化（跨轮组件功能追踪）

### 进度信号（pos）
| 轮次 | 组件名 | 得分 | 设计特征 |
|------|--------|------|----------|
| 1 | pos | 26.74 | -1.0 * dist²（绝对值惩罚，连续负信号） |
| 2 | pos | 64.24 | 3.0 * (prev_dist - next_dist)（改善量，delta形式） |
| 3 | pos | -21.40 | 50.0 * (prev_dist - next_dist)（改善量，权重 x16.7） |
| 4 | pos | 24.69 | 8.0 * (prev_dist - next_dist)（改善量，权重从 50→8） |
| 5 | pos | -40.67 | 8.0 * (prev_dist - next_dist)（改善量，与 iter_04 相同） |

**方向切换**：绝对值惩罚 ↔ 改善量：1 次切换（iter_01→iter_02）

### 着陆/接触信号（contact）
| 轮次 | 组件名 | 得分 | 设计特征 |
|------|--------|------|----------|
| 1 | contact | 26.74 | binary (nleft*nright), threshold=0.3, w=10.0 |
| 2 | contact | 64.24 | binary (nleft*nright), threshold=0.3, w=10.0（同 iter_01） |
| 3 | contact | -21.40 | binary (nleft*nright), threshold=0.3, w=10.0（同 iter_01）— 但 active_rate=0% |
| 4 | contact | 24.69 | binary (nleft*nright), threshold=0.3, w=10.0（同 iter_01）— active_rate=0.5% |
| 5 | contact | -40.67 | gradual (nleft+nright), threshold=0.5, w=6.0 — active_rate=0.1% |

**方向切换**：binary（nleft*nright）↔ gradual（nleft+nright）：1 次切换（iter_04→iter_05）

### 稳定性/约束信号（vel, angle, angvel）
| 轮次 | 组件名 | 得分 | 设计特征 |
|------|--------|------|----------|
| 1 | vel, angle, angvel | 26.74 | 门控惩罚（gate=1/(1+10*dist)），w 均为 0.5 |
| 2 | vel, angle, angvel | 64.24 | 同 iter_01 |
| 3 | vel, angle, angvel | -21.40 | 同 iter_01 |
| 4 | vel, angle, angvel | 24.69 | 同 iter_01 |
| 5 | vel, angle, angvel | -40.67 | 同 iter_01 |

**方向切换**：0 次切换 — 稳定性组件设计在所有 5 轮中完全一致

### 完成/成功代理
无此语义功能的组件存在。contact 组件承担了部分的着陆成功信号角色，但其数学形态属于"着陆/接触信号"类别。

---

**数据报告结束。**

# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs indices:
    # 0: x_position (relative to pad center)
    # 1: y_position (relative to pad height)
    # 2: x_velocity
    # 3: y_velocity
    # 4: body_angle
    # 5: angular_velocity
    # 6: left_support_contact (0.0 or 1.0)
    # 7: right_support_contact (0.0 or 1.0)

    ox = obs[0]
    oy = obs[1]
    nx = next_obs[0]
    ny = next_obs[1]
    nvx = next_obs[2]
    nvy = next_obs[3]
    nangle = next_obs[4]
    nangvel = next_obs[5]
    nleft = next_obs[6]
    nright = next_obs[7]

    # Distances
    prev_dist = (ox**2 + oy**2 + 1e-8) ** 0.5
    next_dist = (nx**2 + ny**2 + 1e-8) ** 0.5

    # -- 1) approach progress: reward reduction in distance to pad --
    w_pos = 3.0
    pos_reward = w_pos * (prev_dist - next_dist)   # >0 when approaching, <0 when retreating

    # -- 2) soft landing: speed & attitude penalty gated by distance --
    k_gate = 10.0
    gate = 1.0 / (1.0 + k_gate * next_dist)         # near pad -> gate ~1, far away -> gate ~0

    w_vel = 0.5
    vel_penalty = -w_vel * (nvx**2 + nvy**2) * gate

    w_angle = 0.5
    angle_penalty = -w_angle * (nangle**2) * gate

    w_angvel = 0.5
    angvel_penalty = -w_angvel * (nangvel**2) * gate

    # -- 3) safe contact: reward both legs on the pad near center --
    contact_proxy = nleft * nright                    # both legs touching -> 1.0, else 0.0
    contact_threshold = 0.3
    contact_factor = max(0.0, 1.0 - next_dist / contact_threshold)
    w_contact = 10.0
    contact_reward = w_contact * contact_proxy * contact_factor

    total_reward = pos_reward + vel_penalty + angle_penalty + angvel_penalty + contact_reward

    components = {
        'pos': pos_reward,
        'vel': vel_penalty,
        'angle': angle_penalty,
        'angvel': angvel_penalty,
        'contact': contact_reward
    }

    return total_reward, components
```

# 环境事实（只据此理解任务和变量）
## 1. 任务目标
这是一个 **2D 飞行器轨迹优化任务**。  
主体在视口上方中央位置出发，受到初始随机力作用。  
**首要目标**：到达画面中央的着陆垫，并稳定停靠（减速、保持姿态、安全接触）。  
**次要目标**：同时尽可能 **快速到达** 且 **节省引擎燃料**。  
**不应混淆的目标**：不是单纯的无限制飞行或持续移动；也不是简单的平衡或抓取任务；核心在于“安全抵达+稳定着陆”。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: `x_position` — 水平相对坐标（相对于目标垫中心），可用于奖励： true
- obs[1]: `y_position` — 垂直相对坐标（相对于着陆垫高度），可用于奖励： true
- obs[2]: `x_velocity` — 水平线速度，可用于奖励： true
- obs[3]: `y_velocity` — 垂直线速度，可用于奖励： true
- obs[4]: `body_angle` — 机体倾斜角度，可用于奖励： true
- obs[5]: `angular_velocity` — 角速度，可用于奖励： true
- obs[6]: `left_support_contact` — 左支撑腿接触标志（1.0 表示接触），可用于奖励： true
- obs[7]: `right_support_contact` — 右支撑腿接触标志，可用于奖励： true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine`（不做任何推进/姿态调整）
- action 1: `left_orientation_engine`（启动左姿态引擎，产生逆时针旋转力矩）
- action 2: `main_engine`（启动主引擎，产生与机体朝向一致的推力）
- action 3: `right_orientation_engine`（启动右姿态引擎，产生顺时针旋转力矩）

> 说明：所有引擎动作都会消耗燃料，而燃料总量在观测中不可见。主体通过姿态调整+主引擎实现2D空间的移动。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination**：  
  `body_not_awake_or_settled` — 主体静止/稳定，若同时满足 **当前坐标已落在目标垫上**（x,y接近0且接触着陆），视为成功着陆。但环境**未显式标记**成功。
- **failure‑like termination**：  
  `crash_or_body_contact` — 任意身体部分（如支撑腿之外的部位）与地面/障碍物接触，可能表示坠落/翻倒；  
  `horizontal_position_outside_viewport` — 水平位置超出视口边界，表示飞出任务区。
- **ambiguous termination**：  
  仅靠 `body_not_awake_or_settled` 而不检查位置会无法区分“卡在垫外停止”或“悬空静止”的情况，需要结合观测判断。
- **truncation**： 无 episode 长度截断（未在 step 代码中体现）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: 空（info 始终为 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（包括任何 `success`、`failure`、`termination_reason`）

## 7. 可用于奖励函数的信号
- **position**：`obs[0]` (x), `obs[1]` (y) 及 `next_obs[0:2]`
- **velocity**：`obs[2]` (vx), `obs[3]` (vy) 及对应 next_obs
- **orientation / angular**：`obs[4]` (角度), `obs[5]` (角速度) 及对应 next_obs
- **contact**：`obs[6]` (左接触), `obs[7]` (右接触) 及对应 next_obs  
  注意：接触标志为 0.0/1.0，可组合判断单脚/双脚接触。
- **action/engine**：`action` 可取 0–3，可映射为燃料消耗（主引擎或姿态引擎使用）
- **other**：无额外环境信息
```
