# 训练数据报告

## 1. 行为事实

- **得分**：-119.506414，范围 [-142.824994, -96.557485]
- **终止方式**：terminated=20/20（全部因早停条件触发），truncated=0/20
- **回合长度**：均值 68.3 步
- **Checkpoint 轨迹**：
  - 40% 训练步数：官评得分 0.00（std=0.00）
  - 80% 训练步数：官评得分 0.00（std=0.00）
- **奖励组件统计**：
  - `progress_shaping`：ep_sum=5.583, active_rate=100.0%
  - `success_landing_bonus`：ep_sum=5.000, active_rate=0.1%
  - `angle_penalty`：ep_sum=-0.205, active_rate=100.0%
  - `action_cost`：ep_sum=-0.071, active_rate=5.2%

---

## 2. 骨架演进表

| iter | score | len | term | progress_shaping_ep_sum | progress_shaping_active | angle_penalty_ep_sum | angle_penalty_active | action_cost_ep_sum | action_cost_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active | success_landing_bonus_ep_sum | success_landing_bonus_active |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -155.975 | 154.4 | terminated=20/20, truncated=0/20 | 1.582 | 100.0% | -1.141 | 100.0% | -1.804 | 58.4% | -0.067 | 5.3% | — | — |
| 2 | -119.506 | 68.3 | terminated=20/20, truncated=0/20 | 5.583 | 100.0% | -0.205 | 100.0% | -0.071 | 5.2% | — | — | 5.000 | 0.1% |

---

## 3. 数字标记

### 修改变化（iter1 → iter2）

| 组件 | Δ ep_sum | Δ active_rate |
|---|---|---|
| progress_shaping | +4.001 | 0 pp |
| angle_penalty | +0.936 | 0 pp |
| action_cost | +1.733 | −53.2 pp |
| landing_velocity_penalty → success_landing_bonus | +5.067 | −5.2 pp |

### 死组件
- **`success_landing_bonus`**：active_rate=0.1%（< 2%）——该组件在绝大多数步中值为 0

### 淹没
- **`angle_penalty`**：active_rate=100.0%（> 90%）但 magnitude_share=1.9%（< 3%）

### 每步均值（ep_sum ÷ len）量级对比
- **最大两个**：`progress_shaping`=0.082, `success_landing_bonus`=0.073
- **最小两个**：`angle_penalty`=−0.003, `action_cost`=−0.001

### 无主导、无全覆盖惩罚
- 无组件 magnitude_share > 80%
- 无组件满足 active_rate > 95% 且 signed_share 为负

---

## 4. 语义演化

### 接近/进度信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 距离势能塑造 | v1: `progress_shaping` → v2: `progress_shaping` | v1: −155.975 → v2: −119.506 | 无 |
- v1→v2 未改变设计，为连续型 distance-to-goal shaping

### 着陆/接触信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 成功着陆反馈 | v1: `landing_velocity_penalty`(死) → v2: `success_landing_bonus`(死) | v1: −155.975 → v2: −119.506 | 1 次（连续惩罚 → 稀疏 bonus） |
- v1 使用 hinge 速度惩罚（高度门控），active_rate=5.3%（接近死组件）
- v2 替换为稀疏大额正面奖励（+100），active_rate=0.1%（死组件）
- 方向切换次数：1

### 稳定性/约束信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 姿态约束 | v1: `angle_penalty` → v2: `angle_penalty` | v1: −155.975 → v2: −119.506 | 无 |
- 两轮均使用二次角惩罚，权重 0.5，无设计变化
- v2 中该组件被淹没（active_rate=100%, magnitude_share=1.9%）

### 能效信号
| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|---------|---------|
| 动作代价 | v1: `action_cost` → v2: `action_cost` | v1: −155.975 → v2: −119.506 | 无 |
- 两轮均为 −0.02 per 非零动作，active_rate 从 58.4% 降至 5.2%