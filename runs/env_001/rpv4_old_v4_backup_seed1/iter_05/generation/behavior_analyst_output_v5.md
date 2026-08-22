# 数据报告（iter_04）

## 1. 行为事实
- **得分**：score = -10.11，range = [-43.15, 24.82]
- **终止方式**：terminated 0/20（成功着陆0次），truncated 20/20（全部达到1000步上限）
- **回合长度**：len = 1000.0
- **Checkpoint 得分轨迹**（每10%步数官方环境得分）：

| cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|
| -119.2 | -45.3 | -182.0 | -164.2 | -139.9 | -118.5 | -95.3 | -141.7 | 59.6 | -8.6 |

  走势：从 -119 先升后骤降至 -182（cp30%），随后波动上升至 cp90% 的 +59.6，最后骤降至 -8.6（cp100%）。  
  同期 **contact_bonus** 在 cp90% 突然从 0 跃至 158.68（active_rate 2.5%），cp100% 又归零。

## 2. 骨架演进表
| iter | score | len | term (终止/截断) | goal_proximity_ep_sum | goal_proximity_active | upright_penalty_ep_sum | upright_penalty_active | velocity_penalty_ep_sum | velocity_penalty_active | contact_bonus_ep_sum | contact_bonus_active |
|------|-------|-----|------------------|------------------------|------------------------|-------------------------|-------------------------|--------------------------|--------------------------|-----------------------|-----------------------|
| 1 | -76.56 | 975.15 | 1/20 终止，19/20 截断 | 0.635 | 100.0% | -3.714 | 100.0% | -12.561 | 100.0% | 0.0 | 0.0% |
| 2 |  63.72 | 668.05 | 17/20 终止，3/20 截断 | 1.835 | 98.8% | -7.301 | 100.0% | -9.517 | 99.0% | 1075.63 | 5.2% |
| 3 |  21.45 | 864.55 | 9/20 终止，11/20 截断 | 1.769 | 99.9% | -11.452 | 100.0% | -9.103 | 99.9% | 1011.47 | 8.2% |
| 4 | -10.11 | 1000.00 | 0/20 终止，20/20 截断 | 2.922 | 100.0% | -5.113 | 100.0% | -7.837 | 100.0% | 0.0 | 0.0% |

## 3. 数字标记
- **主导**（magnitude_share > 80%）：  
  iter2 与 iter3 中 contact_bonus 主导（98.2% / 97.7%）。  
  iter4 无组件主导（最大为 velocity_penalty 47.6%）。
- **死组件**（active_rate < 2%）：  
  iter1 与 iter4 中 contact_bonus 为死组件（0.0%）。  
  iter2、iter3 无死组件。
- **淹没**（active_rate > 90% 且 magnitude_share < 3%）：  
  iter2 与 iter3 中 goal_proximity 被淹没（active 98.8%/99.9%，magnitude_share 0.3%）。  
  iter4 无淹没。
- **全覆盖惩罚**（active_rate > 95% 且 signed_share 为负）：  
  所有迭代中 velocity_penalty 和 upright_penalty 均符合条件。
- **量级对比**（iter4 每步均值 ep_sum/len）：  
  最大两个：goal_proximity 0.00292，contact_bonus 0.0；  
  最小两个：velocity_penalty -0.00784，upright_penalty -0.00511。
- **修改变化**（iter4 相对 iter3 的 Δ）：  
  - goal_proximity：ep_sum +1.153，active +0.1%  
  - upright_penalty：ep_sum +6.339，active 不变  
  - velocity_penalty：ep_sum +1.266，active +0.1%  
  - contact_bonus：ep_sum -1011.47，active -8.2%

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮得分（ep_sum_mean） | 方向切换 |
|----------|------------|--------------------------|----------|
| 接近信号 | goal_proximity (iter1) → goal_proximity (iter2) → goal_proximity (iter3) → goal_proximity (iter4) | 0.635 → 1.835 → 1.769 → 2.922 | 线性距离delta → 逆距离delta：1次 |
| 着陆信号 | contact_bonus (iter1) → contact_bonus (iter2) → contact_bonus (iter3) → contact_bonus (iter4) | 0.0 → 1075.63 → 1011.47 → 0.0 | 无速度门控→有速度门控→无速度门控：2次切换 |
| 速度约束 | velocity_penalty (iter1) → velocity_penalty (iter2) → velocity_penalty (iter3) → velocity_penalty (iter4) | -12.56 → -9.52 → -9.10 → -7.84 | 全高度惩罚→高度门控惩罚：1次 |
| 姿态约束 | upright_penalty (iter1) → upright_penalty (iter2) → upright_penalty (iter3) → upright_penalty (iter4) | -3.71 → -7.30 → -11.45 → -5.11 | 无切换 |

*着陆信号的方向切换 ≥ 2 次：同一功能在“稀疏导弹（无额外门控）→带速度门控的稀疏导弹→退回去门控”之间反复，可能存在振荡。*