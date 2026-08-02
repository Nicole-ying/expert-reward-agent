# ⚠️ REBUILD MODE
系统接受了你的 Level 3 重建建议。你不是在修改上一轮代码——你是在基于全部历史设计新骨架。
参考 #6 完整公式算子库选新的主信号框架，基于 #3 累积记录避开已失败的路径。
不要受上一轮代码结构约束。


# 1. Search objective
- target_score: 200.000000
- current_score: -166.964935
- gap_to_target: 366.964935
- target_achievement_ratio: -83.482%

# 2. 上一轮奖励函数代码（该轮得分: -166.964935）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_cur, y_cur = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 超参数
    PROGRESS_WEIGHT = 2.0
    FAIL_PENALTY = -10.0
    LANDING_PROXY_WEIGHT = 2.0
    ANGLE_PENALTY = 0.5
    ANG_VEL_PENALTY = 0.1

    X_BOUNDARY = 1.0
    ANGLE_CRASH = 0.8          # 弧度，约45度
    GROUND_Y_CLOSE = 0.2       # 接近垫面的高度
    VEL_CRASH = 1.5            # 撞击速度阈值
    DIST_LAND = 0.3            # 着陆判定距离范围
    VEL_LAND = 0.5
    ANGLE_LAND = 0.3

    # 1. 进展信号：每步距离的变化量
    dist_cur = (x_cur**2 + y_cur**2) ** 0.5
    dist_next = (x_next**2 + y_next**2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)   # 期望 >0

    # 2. 失败惩罚（推断终止原因）
    out_of_bounds = abs(x_next) > X_BOUNDARY
    crash = False
    if (left_contact == 1.0 or right_contact == 1.0):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > ANGLE_CRASH
        high_impact = abs(y_vel_next) > VEL_CRASH
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    failure = out_of_bounds or crash
    failure_penalty = FAIL_PENALTY if failure else 0.0

    # 3. 软着陆近似信号（多条件代理）
    # 距离因子
    dist_to_pad = (x_next**2 + y_next**2) ** 0.5
    dist_factor = max(0.0, 1.0 - dist_to_pad / DIST_LAND)
    # 速度因子
    speed = abs(x_vel_next) + abs(y_vel_next)
    vel_factor = max(0.0, 1.0 - speed / VEL_LAND)
    # 姿态因子
    angle_factor = max(0.0, 1.0 - abs(body_angle_next) / ANGLE_LAND)
    # 接触因子
    contact_factor = 0.5 * (left_contact + right_contact)   # 0, 0.5, 或1

    landing_proxy = (dist_factor + vel_factor + angle_factor + contact_factor) / 4.0
    landing_proxy_reward = LANDING_PROXY_WEIGHT * landing_proxy

    # 4. 姿态/稳定轻惩罚
    stability_penalty = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    total_reward = progress + failure_penalty + landing_proxy_reward + stability_penalty

    components = {
        'progress': progress,
        'failure_penalty': failure_penalty,
        'landing_proxy': landing_proxy_reward,
        'stability_penalty': stability_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录
（第一轮反思，无历史记录）

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-166.964935, len=997.950000, terminated=1/20, truncated=19/20, reward_errors=0
score_range=[-242.400897, -126.702705]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 865.369103 | 99.7% | 99.7% | 100.0% |
| stability_penalty | -1.060243 | -0.1% | 0.1% | 100.0% |
| progress | -0.673900 | -0.1% | 0.1% | 100.0% |
| failure_penalty | -0.500000 | -0.1% | 0.1% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 轨迹优化与精准着陆任务。一架拥有主引擎和双定向引擎的飞行器从视口顶部中心附近随机初始力释放，需要尽快飞到中心目标垫（着陆平台）上，实现安全、稳定的接触着陆。主要目标是**快速到达并悬停/降落在目标垫上**，次要目标是**节约燃料（减少引擎使用）**，同时在接近过程中保持姿态稳定，避免猛烈碰撞或飞出视口。任务不应被误解为持续前进或单纯生存平衡：它是以到达指定位置并“软接触”为核心的导航目标到达任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据典型实现推断）
- **obs[0]**: `x_position` – 飞行器相对于目标垫的水平坐标（偏移），reward_usable: true （可用作距离度量）
- **obs[1]**: `y_position` – 飞行器相对于垫高度的垂直坐标（偏移），reward_usable: true （距离度量）
- **obs[2]**: `x_velocity` – 水平线速度，reward_usable: true （可惩罚/奖励减速）
- **obs[3]**: `y_velocity` – 垂直线速度，reward_usable: true （同上）
- **obs[4]**: `body_angle` – 机体倾角，reward_usable: true （可鼓励保持水平）
- **obs[5]**: `angular_velocity` – 角速度，reward_usable: true （可惩罚过快的旋转）
- **obs[6]**: `left_support_contact` – 左侧支撑脚接触标志（0 或 1），reward_usable: true （可用于判断着陆状态）
- **obs[7]**: `right_support_contact` – 右侧支撑脚接触标志，reward_usable: true （同上）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- **action 0**: `no_engine` – 不点火，飞行器仅受重力、风等影响（节省燃料的动作）
- **action 1**: `left_orientation_engine` – 启动左侧定向引擎，产生力矩调整姿态
- **action 2**: `main_engine` – 启动主引擎，产生向上的推力（可能也有水平分量，取决于姿态）
- **action 3**: `right_orientation_engine` – 启动右侧定向引擎，反向力矩调整姿态

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: 飞行器在目标垫附近稳定停靠（触发 `body_not_awake_or_settled`，且位置靠近原点，速度极小，倾角接近 0）。环境未显式提供成功 flag，需从观测组合推断。
- **failure-like termination**: 
  - `horizontal_position_outside_viewport`：飞行器飞出视口水平边界→失败。
  - `crash_or_body_contact`（但与目标垫安全接触不同的碰撞）：若飞行器身体（非支撑脚）触地或触垫，或姿态严重倾覆后接触→失败。
- **ambiguous termination**: 在远离目标处触发 `body_not_awake_or_settled`（如早期静止在顶部或其他位置）→应视为失败或无效终止。
- **truncation**: 未在 masked step 中出现，可能无时间截断，但实际环境通常有最大步数（未说明，此处只考虑给定的终止条件）。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 没有任何明确可用的字段（step 返回 `{}`）
- **forbidden_or_uncertain_info_fields**: 不允许依赖 info 中的任何字段；如环境中存在的“success”、“failure”等均为未知，不能直接使用。

**推断路径**：通过观测信号间接区分成功/失败。成功着陆特征：`terminated=True` 时 `|x_position| < ε_x`、`|y_position| < ε_y`（接近垫中心）、`|x_velocity|, |y_velocity|` 接近 0、`|body_angle|` 接近 0、`left_support_contact` 与 `right_support_contact` 至少一个为 1（或两者为 1）。失败则表现为出界、远离目标时的静止、或高速/大角度接触。这些均属于 derived_possible 信号。

## 7. 可用于奖励函数的信号
从观测与终止推断的角度：
- **position**:
  - `x_position`（obs[0]）、`y_position`（obs[1]）— 反映到目标垫的欧氏距离或分量距离。
  - 可从 `next_obs` 获取下一步位置。
- **velocity**:
  - `x_velocity`（obs[2]）、`y_velocity`（obs[3]）— 反映接近速度，可用于减速奖励。
- **orientation**:
  - `body_angle`（obs[4]）— 应保持接近 0（水平）。
  - `angular_velocity`（obs[5]）— 抑制过快自旋。
- **contact**:
  - `left_support_contact`（obs[6]）、`right_support_contact`（obs[7]）— 脚部与垫接触，可用于终端着陆检测或稳定性奖励。
- **action/engine**:
  - `action` 本身可用于惩罚引擎使用（燃料惩罚），三个有推力动作为 1,2,3，可赋予不同权重。
- **other**:
  - 终端事件推断：根据终止时的 `next_obs` 状态组合判断成功着陆或失败（出界、坠毁等），提供 derived_possible 奖励/惩罚。

# 7. Formula Operator Library（完整版，用于 Level 3 重建）
# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

---

## 1. Expert Schema 使用规则

- environment_card.md 中的任务画像和可用信号优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role（任务需要什么类型的奖励信号），再选 signal（哪个观测维度承载这个 role），再选 formula operator（用什么数学形式表达），最后写代码。
- 如果某个 role 需要的信号在观测空间中不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- reward_v1 以主学习信号和必要的稳定/安全约束为重点。效率、能耗、复杂门控和动态权重可以在后续迭代中按需加入，但不应因"模板没列"而排除合理的设计。

---

## 2. 信号完备性自查清单

在完成初始设计后，逐一检查以下信号类型是否被覆盖——不是每个任务都需要全部，但每一项的缺失应是有意选择：

- **主进展信号**：agent 朝任务目标前进时是否获得正向反馈？该信号是否每步都有梯度？
- **灾难性失败信号**：是否存在明确的终止惩罚（如摔倒、飞出边界）？如果观测中可推断失败状态，是否给予了足够强的负向信号？
- **效率/代价信号**：连续动作空间中是否有能量消耗或控制代价约束？离散动作空间中是否有不必要的动作惩罚？
- **任务完成信号**：终止条件中是否包含 success-like 条件？相应的观测是否可被用来构造任务完成的软近似信号？
- **健康/稳定约束**：agent 是否因缺少姿态/速度/位置约束而产生不安全行为？

---

## 3. Formula Operator Library

每个算子包含：数学形式、使用条件、适用证据。

### 3.1 dense_state_signal
数学形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2`
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
使用条件：该状态信号每步可观测，且与某项任务职责直接相关。
适用证据：
  - 凸化 → episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小（agent 满足于低水平稳态）。
  - hinge → 约束组件的 active_rate≈100%（全时惩罚）但 terminated 率仍高，说明 agent 在安全范围内也被持续惩罚，需要只在越界时生效的 hinge。
风险：线性正奖励在信号平台期无梯度；凸化权重过大可能诱导极端行为；hinge 的 threshold 需根据环境卡片的观测范围设定。

### 3.2 improvement_delta
数学形式：`old_measure - new_measure`（期望减少时）或 `next_value - current_value`（期望增加时）
使用条件：obs 和 next_obs 中存在可比较的标量度量，该度量沿最优路径应单调变化。
适用证据：有明确的进展度量（位置、距离、高度、角度等），且该度量的变化比瞬时速率更能反映真实进展。
与 dense_state_signal 的选择：如果要鼓励"处于某种好状态"，用 `w * signal`。如果要鼓励"朝好方向改变"，用 delta。delta 的优势是 agent 无法在好状态上停滞不前，必须持续改善。适合：agent 当前的绝对状态值不能完全反映进展（如位置——站在原点不动 vs. 走到终点但位置绝对值可能相同）。
注意：对观测中直接给出的速度信号（如 `horizontal_velocity`）不要做 delta——速度本身已经是变化率。对观测中的位置/角度/距离类信号优先考虑 delta。

### 3.3 potential_based_shaping
数学形式：`potential(next_obs) - potential(obs)`
使用条件：(1) 任务有一个可量化的进展度量（如位置、距离、高度）；(2) 该度量沿最优路径应单调变化；(3) 能从观测中构造一个标量的 potential function。
如何构造 potential：从观测中选择一个在任务完成时达到极值、且沿最优路径单调变化的信号（或信号组合）。potential 的计算只能依赖观测，不能依赖环境内部状态。
与 improvement_delta 的关系：两者数学上等价。potential_based_shaping 的优势在于允许将多个信号编码到一个 potential 中（如同时考虑位置和姿态），而 improvement_delta 通常用于单个度量。
风险：potential 若与任务目标不一致会系统性地误导策略。reward_v1 中如果存在天然的进展度量，优先使用 improvement_delta 的简单形式；当需要组合多个信号构造进展度量时，使用 potential_based_shaping。

### 3.4 quadratic_penalty
数学形式：`-w * error**2` 或 `-w * sum(action_i**2)`
使用条件：约束信号连续可观测，惩罚不应压制主学习信号。用于轻量抑制——需要约束但不至于触发终止的行为。
适用证据：某维度出现高频大幅波动或极端值但未触发终止。
与 hinge 的选择：如果约束有明确的安全边界（如身体倾角超过 X 度必摔），用 hinge（3.1）。如果只是希望"越小越好"没有硬边界（如控制代价、小幅抖动），用 quadratic。
风险：权重过大导致 agent 不敢行动。

### 3.5 soft_health_gate
数学形式：`main_reward * gate_factor`，gate_factor ∈ [0, 1] 在身体状态恶化时平滑衰减。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (safe_bound - current) / margin))`
使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
适用证据：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍 >0——agent 在"先冲后死"，需要在健康恶化时切断主奖励而非额外加罚。
风险：gate 太严格抑制探索；衰减区间应设在"接近危险但尚未终止"的范围内。

### 3.6 terminal_event
数学形式：`if failure_condition: reward = -PENALTY`（硬覆盖 per-step 奖励），或 `if success_condition: reward = +BONUS`
使用条件：(1) 存在可从观测推断的灾难性失败状态（如身体倾角超过阈值 + 接触地面）或任务完成状态；(2) 环境 info 为空因此无法直接读取终止原因。
如何构造：不要依赖 info 字段判断终止原因。可从观测推断：摔倒 → hull_angle 突然偏转 + 身体位置急剧下降；到达终点 → 持续前进中 episode 突然终止（truncated）；出界 → 位置坐标超出有效范围。
适用证据：agent 频繁触发某种终止模式，但当前奖励没有针对该模式提供差异化信号——比如所有终止回合 reward 都一样，agent 无法区分成功和失败。
与 hinge/gate 的区别：hinge 在越界前提供连续梯度，gate 在恶化时衰减主信号。terminal_event 在事件发生的那一刻提供硬信号——没有梯度，但语义明确（"这就是你应该避免/追求的结果"）。

### 3.7 action_efficiency
数学形式：`-w * sum(|action_i|)` 或 `-w * sum(action_i**2)`
使用条件：动作空间 ≥ 2 维连续控制，且任务包含隐含的效率需求（如 locomotion、manipulation）。
适用证据：agent 学会完成任务但动作幅度异常大、能耗高——说明缺效率约束。通常系数较小（主信号 per-step 的 1-5%），避免压制探索。
注意：离散动作空间通常不需要此算子，因为离散动作的选择隐含了代价。首次迭代可不加入，后续迭代若观察到无效动作频繁出现再考虑。

### 3.8 joint_condition_proxy
数学形式：`factor_1 * factor_2 * ...`（每个 factor 为连续 bounded 形式）或 `(f1 + f2 + ...) / n` 或 `(f1 * f2 * ...) ** (1/n)`
使用条件：没有显式 success flag，但有连续信号可构造任务完成的软近似。
适用证据：agent 能在各子条件分别取得进展但无法同时满足。
风险：乘积塌缩（一个 factor→0 则整体→0）；用几何平均或算术平均可缓解。

### 3.9 bounded_signal
数学形式：`x / (1 + abs(x))` 或 `1 / (1 + k * abs(error))` 或 `max(0, 1 - abs(error) / threshold)`
使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。用于压缩极端值而非施加约束。
与 hinge 的区别：bounded 是从两端压缩信号范围，hinge 是只在超出阈值时施加惩罚。如果目标是"值不应超过 X"，用 hinge；如果目标是"值不应该爆炸但无所谓具体范围"，用 bounded。

### 3.10 preview_conditioned_reward
数学形式：`main_reward * preview_factor`，preview_factor 基于观测中能反映**未来状态**的信号（如距离传感器、高度采样、前方地形探测），在不利前景下从 1 平滑衰减到下限。
使用条件：(1) 观测中存在提供前方/未来信息的维度；(2) 该维度可以映射到"前景好/坏"的连续度量；(3) agent 的失败模式与"无法提前调整行为以应对即将到来的状态变化"相关。
如何构造：从提供未来信息的观测中选择一个标量信号，设计一个在安全前景下接近 1、危险前景下接近下限（如 0.3-0.5）的衰减函数。下限不为零以避免完全抑制探索。
适用证据：agent 在相似的瞬时状态下表现差异大（同样的速度/姿态，有时成功有时失败），说明当前状态本身不足以区分好坏——缺少关于"接下来会发生什么"的信息。
与 soft_health_gate 的区别：gate 用当前的**身体状态**乘主奖励（"我已经歪了，别冲了"——被动响应）。preview 用**未来信息**乘主奖励（"前面是坑，别冲了"——主动预判）。两者可以共存：`main_reward * health_gate * preview_factor`。
风险：preview 信号若有噪声会导致主奖励波动；衰减下限设太低会抑制必要探索。

---

## 4. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"定位到合适的算子变换。
以数学语义和训练表现证据为准，不要求组件名完全匹配。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |



# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | failure_penalty + landing_proxy + progress + stability_penalty | -166.96 | -166.96 | 0.00 | 997.95 | failure_penalty=-0.005 landing_proxy=0.764 progress=0.001 stability_penalty=-0.008 | new_best |
