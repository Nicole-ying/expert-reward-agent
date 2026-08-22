# 诊断报告

## 1. 当前策略行为画像
agent 学会了接触着陆（13/20 终止），但回合过长（737 步）且得分低于历史最佳（103.63 vs 142.32）。contact_landing_reward 主导 90.7% 奖励份额，但仅 13.8% 激活率说明接触发生在末尾集中区间。

## 2. 组件健康度
- **contact_landing_reward**: 主导但错位（90.7% share，13.8% active，per-step 0.588）
- **goal_approach_progress**: 淹没（2.4% share，99.3% active，per-step 0.016）
- **velocity_brake_near_target**: 健康但被掩盖（-5.3% share，99.5% active，per-step -0.034）
- **orientation_stability**: 健康但被掩盖（-1.3% share，100% active，per-step -0.009）

## 3. 骨架演进表

```
| iter | score  | len    | term | contact_ep | contact_active | approach_ep | approach_active | velocity_ep | orient_ep |
|------|--------|--------|------|------------|----------------|-------------|-----------------|-------------|-----------|
| 2    | 142.32 | 721.10 | ?/20 | ~1517      | ?              | ~19.5       | ?               | ~-126       | ~-45      |
| 3    | 36.97  | 930.15 | 7/20 | 2031       | 52.3%          | 159         | 100%            | -221        | -61       |
| 4    | 103.63 | 736.95 | 13/20| 433        | 13.8%          | 12          | 99.3%           | -25         | -6        |
```

**演进解读**：iter3→4 将 contact 从加性权重和（weighted sum）改回几何平均（geometric mean^0.333），导致接触奖励暴跌 79%（2031→433），但终止率反升（7→13），说明质量门槛提高后反而促进了"正确接触"。然而 goal_approach_progress 同时崩塌 92%（159→12），distance shaping 被 contact 目标彻底压制，导致航行效率极低（len 仍高达 737）。

**交互效应**：contact 门槛收紧 + approach 信号失效 = agent 学会"磨蹭到目标附近才尝试接触"的低效策略，解释了高终止率但低分的矛盾现象。

## 4. 行为结论
**病态**。当前配置陷入"完成信号倒挂"陷阱：
- contact（稀疏完成信号）量级是 approach（稠密引导信号）的 **37 倍**（per-step 0.588 vs 0.016）
- approach 虽然 99.3% 全程激活，但单步贡献仅 0.016，在 contact 的 0.588 面前形同虚设

**最需干预组件**：**goal_approach_progress**。不是因为它"坏了"，而是它的 10.0 系数在 contact 5.0×quality 面前被结构性淹没。需要将 approach 系数提升至 50-100 量级，或将 contact 基础系数降至 1.0，才能让"航行→接近→着陆"的完整策略链条形成正确的梯度传递。