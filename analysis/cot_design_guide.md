# CoT 诊断引导：从组件统计到修改决策

## Step 0：组件统计 → 行为模式诊断

按以下优先级逐一检查组件统计表的数字特征，匹配行为模式：

### 优先级 1：信号不可达（active_rate = 0%）
| 特征 | 行为诊断 | 优先变换 |
|------|---------|---------|
| 某组件 active_rate = 0%，但环境卡标记为 mandatory role | 信号不可达——乘积塌缩或硬阈值过高 | product_to_noncollapsing_joint 或 sparse_to_dense |
| 某组件 active_rate < 2% 且 signed_share < 5% | 僵尸组件——几乎不参与训练 | 检查是否存在不必要的硬条件门控 |

### 优先级 2：奖励耕作（magnitude_share 异常高 + active_rate 异常）
| 特征 | 行为诊断 | 优先变换 |
|------|---------|---------|
| active_rate < 5% 但 magnitude_share > 80%，且 sum 值大 | 稀疏高值事件支配——弹跳/偶然触发刷分 | global_to_local_gate（加门控）|
| active_rate > 50% 且 magnitude_share > 80%，且 truncated > 50% | 持续状态耕作——占据好状态不完成任务 | state_to_improvement 或 dense_to_task_event |
| active_rate > 50% 且 magnitude_share > 90%，且 terminated 率正常但 score 未达 target | 悬停耕作——在目标附近停留刷分但不完成 | state_to_improvement（改为势能差）|

### 优先级 3：快速失败或徘徊
| 特征 | 行为诊断 | 优先变换 |
|------|---------|---------|
| ep_len < 100 且 terminated = 20/20 | rush-and-crash——信号诱导莽撞 | 检查主信号是否无界（unbounded_to_bounded）或过强（L1 降系数）|
| ep_len > 900 且 truncated > 80% | 徘徊——缺少有效梯度 | 检查主信号是否过弱（L1 提系数）或缺失完成信号 |
| score 方差巨大（max-min > 300）| 策略不稳定——部分 episode 偶然成功 | 需要更稠密的成功引导 |

### 优先级 4：信号平衡
| 特征 | 行为诊断 | 优先变换 |
|------|---------|---------|
| 惩罚组件 magnitude_share > 40% | 惩罚支配——agent 因恐惧而不行动 | L1 降惩罚系数（目标：share < 20%）|
| 进度信号 magnitude_share < 10% | 主信号淹没——被其他组件压制 | L1 提系数 或 L2 检查其他组件是否过大 |
| terminated < 30% 且 truncated > 70% 但 score 接近 target | 效率问题——能完成任务但太慢 | 考虑加 mild time/fuel penalty |

---

## Step 1：确定干预目标组件

按以下规则排序：
1. active_rate = 0% 的 mandatory role > 
2. magnitude_share > 80% 且 active_rate 异常的组件 >
3. magnitude_share < 10% 的主信号组件 >
4. 其他

**一次只改一个组件。**

---

## Step 2：选 Level

### L1（调系数）的条件——必须全部满足：
- 组件的数学形态合理（符号方向对、有界性合理、激活阶段正确）
- 问题仅在于该组件相对其他组件过强或过弱
- 历史中没有对同一个组件做过 L1 但无效的尝试

### L2（改结构）的条件——任一满足：
- active_rate = 0%（不可达）→ 结构问题
- magnitude_share > 90% 且存在 exploit 行为证据（耕作/弹跳/悬停）
- 同一组件 L1 修复后尺度已正常但行为无改善
- 数学形态被证据直接否定（如无界导致极端值支配）

### L3（重建）的条件——任一满足：
- 同一骨架族已迭代 ≥4 轮且最佳得分 < target × 50%
- 连续 ≥2 轮 L2 后未刷新 best
- 当前 best 的骨架在后续 ≥3 轮中无法复现

---

## Step 3a：L1 权重调整指南

### 系数倍率推断逻辑

1. **先算参考比例**：
   - 对于 progress/主信号组件：计算 `|penalty_sum| / progress_sum` 或 `|competitor_sum| / target_sum`
   - 对于惩罚组件：计算 `|penalty_sum| / |主信号_sum|`

2. **倍率经验范围**（从 paper_v4 全部 seed 统计）：

| 场景 | 合理倍率范围 | 成功案例 | 失败案例 |
|------|------------|---------|---------|
| 主信号太弱（share < 10%）| ×3 ~ ×10 | seed_1 iter_03: ×25 → +46 分 | seed_1 iter_04: ×25 → crash（过大）|
| 主信号太强导致 crash | ÷2 ~ ÷5 | seed_1 iter_05: 25→5 → +149 分 | — |
| 惩罚太强压制探索 | ÷5 ~ ÷20 | seed_3 iter_06: 0.05→0.005 → 稳定 | — |
| 终端 bonus 太弱 | ×2 ~ ×5 | — | seed_1 iter_02: ×20 → clip 饱和 |
| 微调（已接近 solve）| ×0.5 ~ ×2 | seed_2 iter_03: 0.4→0.2 | — |

3. **关键约束**：
   - **单次倍率不超过 ×5**（除非当前值 < 原始值 × 0.1）
   - **终端 bonus 不能设到 clip 边界**（±20 是 clip，bonus 应 ≤ ±15）
   - **惩罚系数初设后应满足 `|penalty| < 0.5 × |progress|`**（避免恐惧行为）

---

## Step 3b：L2 数学形态修改指南

### 常见数学形态问题及标准修复

**1. 无界信号（unbounded_to_bounded）**
```
识别：dist² 或 raw_value，值域 (-∞, ∞)
问题：极端值支配 → 截断策略或莽撞
修复：改为 1/(1+k·|x|) 或 max(0, 1-|x|/D)
示例：-2.0×dist² → 1.0/(1.0+dist)
```

**2. 乘积塌缩（product_to_noncollapsing_joint）**
```
识别：N 个 [0,1] 因子的乘积，任一为零则全零
问题：active_rate→0%，信号不可达
修复：改为加权和 Σ(w_i × factor_i)，保留所有质量因子
关键：不要丢弃原乘积中的因子——每个都独立贡献
示例：A×B×C×D×E×F → 0.4×(A+B+C+D+E+F)
```

**3. 状态值耕作（state_to_improvement）**
```
识别：奖励 = f(state)，agent 占优状态即可持续获奖
问题：徘徊/悬停，truncated 率高
修复：改为 f(next) - f(prev) 或 progress delta
```

**4. 持续状态被刷（persistent_to_transition_event）**
```
识别：只要处于某状态就持续给分，active_rate>50%
问题：agent 占据状态后不完成任务（着陆后不终止）
修复：改为上升沿检测 max(0, curr - prev)
```

**5. 稀疏不可达（sparse_to_dense）**
```
识别：只在严格联合条件满足时触发，active_rate<1%
问题：信用分配不可达——agent 永远触不到
修复：拆成连续梯度层（松条件）+ 质量加成层（紧条件）
```

**6. 缺少阶段门控（global_to_local_gate）**
```
识别：约束在所有阶段等同激活，妨碍早期探索
问题：在无关阶段也受惩罚，agent 不敢行动
修复：加 proximity/height gate，只在相关阶段激活
```

---

## Step 4：验证链

修改后必须输出**可证伪预测**：
1. 改了什么组件的什么属性
2. 预期该组件的 active_rate、magnitude_share、ep_sum 如何变化
3. 预期 episode_length、terminated 率、score 如何变化
4. 最可能的新 exploit 是什么

下一轮用实际数据验证。若预测错误，必须分析原因后再修改。
