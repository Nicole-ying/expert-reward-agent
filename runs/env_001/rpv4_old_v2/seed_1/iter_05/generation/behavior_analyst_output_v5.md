# 数据报告

## 1. 行为事实
- **得分**：score=13.465799，range=[-41.158885, 171.621777]
- **终止方式**：terminated 3/20，truncated 17/20
- **回合长度**：len=989.500000
- **Checkpoint 轨迹**：本轮未提供 checkpoint 评估数据（表中仅有表头，无实际值）

## 2. 骨架演进表
| iter | score | len | term | contact\_bonus\_ep\_sum | contact\_bonus\_active | fuel\_penalty\_ep\_sum | fuel\_penalty\_active | progress\_ep\_sum | progress\_active | orientation\_penalty\_ep\_sum | orientation\_penalty\_active |
|------|-------|-----|------|------------------------|------------------------|-----------------------|-----------------------|-------------------|------------------|------------------------------|------------------------------|
| 1    | -85.93 | 71.1 | T20/Tr0 | 4.254 | 2.2% | -0.785 | 22.1% | 0.539 | 100.0% | -2.748 | 23.9% |
| 2    | -104.33 | 301.4 | T19/Tr1 | 3.128 | 0.3% | -11.463 | 76.1% | 2.207 | 100.0% | -1.587 | 5.1% |
| 3    | 91.82 | 900.7 | T5/Tr15 | 1029.163 | 94.4% | -45.035 | 100.0% | 6.361 | 100.0% | -0.634 | 85.2% |
| 4    | 13.47 | 989.5 | T3/Tr17 | 614.993 | 93.9% | -48.768 | 98.6% | 24.984 | 99.8% | -0.433 | 83.3% |

*注：term 栏格式为 “T<terminated\_count>/Tr<truncated\_count>”*

## 3. 数字标记
- **主导**：contact\_bonus 的 magnitude\_share = 88.6% > 80%，主导当前奖励分布。
- **死组件**：无（所有组件 active\_rate 均远高于 2%）。
- **淹没**：无。
- **全覆盖惩罚**：fuel\_penalty 的 active\_rate = 98.6% > 95%，且 signed\_share = -7.0%（负），符合全覆盖惩罚。
- **量级对比**（本轮每步均值 ep\_sum / len）：
  - 最大两个：contact\_bonus ≈ 0.6215，progress ≈ 0.0253
  - 最小两个：fuel\_penalty ≈ -0.0493，orientation\_penalty ≈ -0.00044
- **修改变化**（iter4 vs iter3，ep\_sum 和 active\_rate 的 Δ）：
  - contact\_bonus：ep\_sum Δ = -414.171，active Δ = -0.5%
  - fuel\_penalty：ep\_sum Δ = -3.733，active Δ = -1.4%
  - progress：ep\_sum Δ = +18.624，active Δ = -0.2%
  - orientation\_penalty：ep\_sum Δ = +0.201，active Δ = -1.9%