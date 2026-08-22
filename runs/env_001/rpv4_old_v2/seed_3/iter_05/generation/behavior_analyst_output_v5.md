# 行为事实
- 得分：score=227.06，range=[190.92, 273.11]
- 终止方式：terminated 20/20（全部成功），truncated 0/20
- 回合长度：len=490.10
- Checkpoint 轨迹：无数据

# 骨架演进表
| iter | score | len | term | prog_ep_sum | prog_act | vel_ep_sum | vel_act | ori_ep_sum | ori_act | safe_ep_sum | safe_act | land_ep_sum | land_act | pen_ep_sum | pen_act |
|:---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 144.17 | 965.45 | T:1, Tr:19 | 1.99 | 100.0% | -1.29 | 100.0% | -3.95 | 100.0% | 312.46 | 65.2% | — | — | — | — |
| 2 | 124.27 | 604.20 | T:13, Tr:7 | 1.88 | 99.2% | -1.44 | 99.2% | -9.48 | 100.0% | 852.00 | 7.1% | — | — | — | — |
| 3 | 110.44 | 528.55 | T:18, Tr:2 | 3.64 | 99.8% | — | — | -15.55 | 100.0% | — | — | 359.97 | 24.5% | -15.68 | 98.9% |
| 4 | 227.06 | 490.10 | T:20, Tr:0 | 1.99 | 97.1% | -1.28 | 97.2% | -3.04 | 100.0% | 33.09 | 7.0% | — | — | — | — |

说明：`—` 表示该版本无此组件。`T` = terminated, `Tr` = truncated。

# 数字标记
- **主导**：safe_contact_reward magnitude_share 83.9% (>80%)
- **死组件**：无
- **淹没**：无
- **全覆盖惩罚**：
  - orientation_stability_reward：active_rate 100% (>95%)，signed_share -7.7%（负）
  - velocity_damping_reward：active_rate 97.2% (>95%)，signed_share -3.2%（负）
- **量级对比**（每步均值 = episode_sum_mean / len = 490.1）：
  - 最大两个：safe_contact_reward (+0.0675)，progress_reward (+0.0041)
  - 最小两个：orientation_stability_reward (-0.0062)，velocity_damping_reward (-0.0026)
- **修改变化**（iter 4 相对于 iter 3 有共同组件的 Δ）：
  - orientation_stability_reward：ep_sum Δ = +12.52（-3.04 vs -15.55），active_rate Δ = 0
  - progress_reward：ep_sum Δ = -1.66（1.99 vs 3.64），active_rate Δ = -2.7%
  - 其余组件在两轮间无可比数据（iter 3 无 safe_contact/velocity_damping，iter 4 无 landing_proxy/action_penalty）。