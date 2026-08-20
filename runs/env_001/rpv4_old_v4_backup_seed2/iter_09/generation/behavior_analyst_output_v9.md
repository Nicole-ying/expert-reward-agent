# 训练数据报告

## 1. 行为事实

- **得分**：score=199.42，range=[-41.00, 281.71]
- **终止方式**：terminated=17/20（成功着陆/接触终止），truncated=3/20（时间耗尽）
- **回合长度**：len=477.65
- **Checkpoint 轨迹**：
  - cp10%=-81.4 → cp30%=45.4 → cp50%=7.1 → cp70%=89.2 → cp80%=189.7 → cp90%=150.1 → cp100%=228.0
  - 总体趋势：前期上升，cp50%小幅回落，cp70%后快速攀升，cp80%-cp90%间有波动(cp80%=189.7→cp90%=150.1→cp100%=228.0)
- **组件层面 checkpoint 关键观察**：
  - `landing_completion`：cp10%-cp60%基本为0，cp70%=6.33，cp80%=36.71，cp90%=121.52，cp100%=79.61（先升后降）
  - `fuel_penalty`：从cp10%=-0.15逐步恶化至cp90%=-3.94，cp100%回升至-2.65
  - `descent_bonus`：全程稳定在5.2-6.5之间
  - `stability`：全程在-2.2到-3.5之间波动，cp90%恶化至-3.54

## 2. 骨架演进表

注：iter 1-5 使用完全不同的组件骨架（progress/speed_penalty/angular_vel_penalty/landing/velocity_guidance），与 iter 6-8 不可比。仅列出 iter 6-8 的演进。

| iter | score | len | term | descent_bonus_ep_sum | descent_bonus_active | distance_progress_ep_sum | distance_progress_active | fuel_penalty_ep_sum | fuel_penalty_active | landing_completion_ep_sum | landing_completion_active | stability_ep_sum | stability_active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | -13.57 | 1000.00 | 0/20, 20/20 | — | — | 6.33 | 100.0% | -9.89 | 98.9% | 0.00 | 0.0% | -1.42 | 100.0% |
| 7 | 146.32 | 673.35 | 14/20, 6/20 | 3.28 | 62.8% | 5.73 | 99.5% | -4.57 | 67.9% | 47.04 | 19.0% | -3.28 | 100.0% |
| 8 | 199.42 | 477.65 | 17/20, 3/20 | 6.38 | 69.7% | 6.30 | 98.5% | -3.05 | 63.9% | 70.98 | 32.3% | -1.87 | 100.0% |

iter 6 无 `descent_bonus` 组件，用 `—` 表示。

## 3. 数字标记

### 主导
- **无组件 magnitude_share > 80%**：`landing_completion` 的 magnitude_share 为 79.7%，未超过 80% 阈值。

### 死组件
- **无组件 active_rate < 2%**：所有组件活跃率均 ≥ 32.3%。

### 淹没
- **无组件同时满足 active_rate > 90% 且 magnitude_share < 3%**。

### 全覆盖惩罚
- **`stability`**：active_rate=100.0%，signed_share=-2.1%（负值），即该组件在所有 episode 的所有步都被激活，且始终贡献负奖励。但 magnitude_share 仅为 2.1%，非全覆盖惩罚（全覆盖惩罚需 active_rate > 95% + signed_share 为负；此处 active_rate=100% 但 magnitude_share 未达到显著性阈值，仅标记为全覆盖负信号而非"淹没"级惩罚）。

### 量级对比（每步均值 = ep_sum ÷ len）

| 组件 | ep_sum | len | 每步均值 |
|------|--------|-----|---------|
| landing_completion | 70.98 | 477.65 | 0.1486 |
| descent_bonus | 6.38 | 477.65 | 0.0133 |
| distance_progress | 6.30 | 477.65 | 0.0132 |
| fuel_penalty | -3.05 | 477.65 | -0.0064 |
| stability | -1.87 | 477.65 | -0.0039 |

- **最大两个**：`landing_completion`(0.1486)、`descent_bonus`(0.0133)
- **最小两个**：`fuel_penalty`(-0.0064)、`stability`(-0.0039)

### 修改变化（iter 7 → iter 8 Δ）

| 组件 | Δ ep_sum | Δ active_rate |
|------|----------|---------------|
| landing_completion | +23.94 | +13.3pp |
| descent_bonus | +3.10 | +6.9pp |
| distance_progress | +0.57 | -1.0pp |
| fuel_penalty | +1.52 | -4.0pp |
| stability | +1.41 | 0.0pp |

## 4. 语义演化（跨轮组件功能追踪）

按语义功能分组（基于 iter 6-8，iter 1-5 使用不同骨架，仅在功能层面可比较的位置加入）：

### 功能角色识别

- **接近/进度信号**：`distance_progress`（距离减少奖励×5.0）、`descent_bonus`（垂直下降奖励×6.0，x门控）
- **着陆/接触信号**：`landing_completion`（双脚接触+近中心+低速低角度的乘积bonus）
- **稳定性/约束信号**：`stability`（速度/角度/角速度平方惩罚×距离门控）、`fuel_penalty`（引擎使用固定惩罚-0.01）

### 语义演化表

| 语义功能 | 各轮组件名 | 各轮 ep_sum | 方向切换 |
|---------|----------|-----------|---------|
| 接近/进度信号 | iter6: distance_progress(6.33) → iter7: distance_progress(5.73)+descent_bonus(3.28) → iter8: distance_progress(6.30)+descent_bonus(6.38) | iter6: 6.33 → iter7: 9.01 → iter8: 12.68 | 无切换（descent_bonus 从 iter7 加入后持续存在，均为连续奖励+改善量形式） |
| 着陆/接触信号 | iter6: landing_completion(0.00, 死) → iter7: landing_completion(47.04) → iter8: landing_completion(70.98) | iter6: 0.00 → iter7: 47.04 → iter8: 70.98 | 稀疏bonus（iter6-8均为条件激活的乘积bonus，无切换） |
| 稳定性/约束信号 | iter6: stability(-1.42)+fuel_penalty(-9.89) → iter7: stability(-3.28)+fuel_penalty(-4.57) → iter8: stability(-1.87)+fuel_penalty(-3.05) | iter6: -11.31 → iter7: -7.85 → iter8: -4.92 | stability：门控惩罚（iter6-8均为exp(-2d)门控，无切换）；fuel_penalty：固定惩罚（iter6-8均为-0.01 per action，无切换） |

### 方向切换统计
- **着陆/接触信号**：0 次切换（iter6-8 均使用条件稀疏bonus）
- **接近/进度信号**：0 次切换（descent_bonus 于 iter7 新增，未发生替代/对立设计切换）
- **稳定性/约束信号**：0 次切换（stability 始终为门控惩罚，fuel_penalty 始终为固定惩罚）

**跨 iter 1-8 全局视角**（补充 iter 1-5 的功能轨迹，基于代码形态）：

| 语义功能 | 各轮组件名 | 得分趋势 | 方向切换 |
|---------|----------|---------|---------|
| 着陆/接触信号 | iter1-3: landing（稀疏contact门控或proximity门控）→ iter4: 删除，velocity_guidance 替代 → iter5: landing（proximity+contact放大）→ iter6-8: landing_completion（稀疏双脚contact+质量乘积） | iter1: 死(0.00) → iter2: 697.44(主导) → iter3: 死(0.00) → iter4: 无 → iter5: 死(0.00) → iter6: 死(0.00) → iter7: 47.04 → iter8: 70.98 | 稀疏bonus↔连续奖励：iter2为连续proximity+sum-of-factors，iter3/5/6/7/8为稀疏contact门控→**至少2次切换**（iter2连续→iter3稀疏→iter5稀疏→iter6-8稀疏）；proximity门控↔contact门控：iter2/5为proximity门控，iter3/6-8为contact门控→**1次切换** |
| 接近/进度信号 | iter1-5: progress（Δdistance×1.0）→ iter6-8: distance_progress（Δdistance×5.0）+ descent_bonus（Δy×系数×x_gate，iter7新增） | iter1-5: progress 逐步萎缩（-0.20→0.66→-0.30→-0.03→-0.18）→ iter6-8: distance_progress 稳定在5.7-6.5 | 缩放系数：1.0→5.0（1次切换）；纯距离→距离+垂直下降分解（iter7新增descent_bonus，1次新增，非对立切换） |
| 稳定性/约束信号 | iter1-5: speed_penalty（closeness×speed）、angular_vel_penalty（绝对值）→ iter6-8: stability（平方惩罚×exp门控）、fuel_penalty（固定值） | iter1-5: 惩罚持续存在→ iter6-8: 惩罚总量改善（-11.31→-4.92） | 线性和→平方惩罚（1次切换）；绝对值→平方（1次切换）；新增fuel_penalty（iter6，1次新增） |