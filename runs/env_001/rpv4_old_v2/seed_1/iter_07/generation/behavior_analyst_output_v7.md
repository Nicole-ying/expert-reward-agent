# 行为事实
- 得分：score=-20.437，range=[-261.300, 212.847]
- 终止方式：terminated 14/20（着陆/坠毁），truncated 6/20（超时）
- 回合长度：len=611.100
- Checkpoint 轨迹：无 Checkpoint 评估数据

# 骨架演进表
| iter | score | len | term | contact_bonus_ep_sum | contact_bonus_active | fuel_penalty_ep_sum | fuel_penalty_active | progress_ep_sum | progress_active | orientation_penalty_ep_sum | orientation_penalty_active |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -85.93 | 71.1 | 20/0 | 4.254 | 2.2% | -0.785 | 22.1% | 0.539 | 100.0% | -2.748 | 23.9% |
| 2 | -104.33 | 301.4 | 19/1 | 3.128 | 0.3% | -11.463 | 76.1% | 2.207 | 100.0% | -1.587 | 5.1% |
| 3 | 91.82 | 900.7 | 5/15 | 1029.163 | 94.4% | -45.035 | 100.0% | 6.361 | 100.0% | -0.634 | 85.2% |
| 4 | 13.47 | 989.5 | 3/17 | 614.993 | 93.9% | -48.768 | 98.6% | 24.984 | 99.8% | -0.433 | 83.3% |
| 5 | -39.29 | 863.4 | 11/9 | 2866.351 | 91.8% | -39.860 | 92.3% | 4.049 | 99.8% | -2.026 | 76.7% |
| 6 | -20.44 | 611.1 | 14/6 | 3244.916 | 13.7% | -23.950 | 78.4% | 4.204 | 100.0% | -2.135 | 67.1% |

term 列为 terminated/truncated 的回合数（共20局）。

# 数字标记
- **主导**：contact_bonus magnitude_share=98.9% > 80%
- **死组件**：无（所有组件 active_rate ≥ 2%）
- **淹没**：progress active_rate=100.0% > 90% 且 magnitude_share=0.3% < 3%
- **全覆盖惩罚**：无（无组件满足 active_rate>95% 且 signed_share<0）
- **量级对比**（每步均值 = ep_sum / len）：
  - 最大两个组件：contact_bonus (5.310), progress (0.0069)
  - 最小两个组件：fuel_penalty (-0.0392), orientation_penalty (-0.0035)
- **修改变化**（iter6 vs iter5）：
  - contact_bonus: Δep_sum=+378.565, Δactive=-78.1pp
  - fuel_penalty: Δep_sum=+15.910, Δactive=-13.9pp
  - progress: Δep_sum=+0.155, Δactive=+0.2pp
  - orientation_penalty: Δep_sum=-0.109, Δactive=-9.6pp