## 1. 行为事实  
- 得分：156.75，范围 [-53.73, 251.51]  
- 终止方式：terminated 16/20，truncated 4/20  
- 回合长度：577.95 步  
- Checkpoint 评估轨迹：无数据（提供的 checkpoint 表为空）  

## 2. 骨架演进表  

| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | velocity_penalty_ep_sum | velocity_penalty_active | angle_penalty_ep_sum | angle_penalty_active | success_proxy_ep_sum | success_proxy_active | contact_quality_ep_sum | contact_quality_active | thrust_penalty_ep_sum | thrust_penalty_active | soft_success_ep_sum | soft_success_active | goal_distance_progress_ep_sum | goal_distance_progress_active | soft_landing_ep_sum | soft_landing_active | orientation_penalty_ep_sum | orientation_penalty_active | landing_contact_success_ep_sum | landing_contact_success_active |  
|------|-------|-----|------|-----------------------|------------------------|--------------------------|---------------------------|------------------------|------------------------|-------------------------|--------------------------|--------------------------|--------------------------|--------------------------|---------------------------|-----------------------|----------------------|---------------------------|--------------------------|--------------------------|---------------------|----------------------|---------------------|--------------------------------|-----------------------------|  
| 1 | -120.99 | 68.35 | term=20/20, trunc=0/20 | -148.51 | 100.0% | -93.23 | 99.9% | -43.99 | 100.0% | 1.43 | 0.1% | 0.77 | 0.2% | -0.21 | 6.3% | — | — | — | — | — | — | — | — | — | — |  
| 2 | -183.32 | 947.35 | term=7/20, trunc=13/20 | -3768.39 | 100.0% | -24.15 | 100.0% | -9.95 | 100.0% | 5792.23 | 100.0% | 0.00 | 0.0% | -47.37 | 100.0% | — | — | — | — | — | — | — | — | — | — |  
| 3 | -120.25 | 68.40 | term=20/20, trunc=0/20 | -156.89 | 100.0% | -93.44 | 99.8% | -38.99 | 100.0% | 1.43 | 0.1% | 0.77 | 0.2% | -0.18 | 5.3% | — | — | — | — | — | — | — | — | — | — |  
| 4 | -1471.59 | 435.40 | term=20/20, trunc=0/20 | 163.93 | 100.0% | -376.90 | 100.0% | -3.79 | 100.0% | — | — | 0.00 | 0.0% | -21.77 | 100.0% | 12716.35 | 96.2% | — | — | — | — | — | — | — | — |  
| 5 | -63.97 | 963.75 | term=1/20, trunc=19/20 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 3.78 | 100.0% | -4.93 | 100.0% | -1.22 | 100.0% | 0.00 | 0.0% |  
| 6 | -37.58 | 947.65 | term=3/20, trunc=17/20 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 4.98 | 100.0% | -2.92 | 100.0% | -2.14 | 100.0% | 2.31 | 0.1% |  
| 7 | 156.75 | 577.95 | term=16/20, trunc=4/20 | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 5.83 | 99.1% | -5.57 | 99.2% | -2.56 | 100.0% | 110.22 | 17.8% |  

## 3. 数字标记  

- **主导**：landing_contact_success（magnitude_share = 87.8%，>80%）  
- **死组件**：无（最小 active_rate 为 17.8%，>2%）  
- **淹没**：orientation_penalty（active_rate = 100.0%，>90%；magnitude_share = 2.0%，<3%）  
- **全覆盖惩罚**：soft_landing（active_rate = 99.2%，>95%；signed_share = -4.4%）；orientation_penalty（active_rate = 100.0%，>95%；signed_share = -2.0%）  
- **量级对比**（每步均值 `ep_sum / len` 绝对值）：  
  - 最大两个：landing_contact_success（0.191）、goal_distance_progress（0.010）  
  - 最小两个：orientation_penalty（0.0044）、soft_landing（0.0096）  
- **修改变化**（iter 6 → iter 7，ep_sum 为差值，active 为百分点差）：  
  - goal_distance_progress：ep_sum +0.851，active -0.9pp  
  - soft_landing：ep_sum -2.649，active -0.8pp  
  - orientation_penalty：ep_sum -0.423，active 0.0pp  
  - landing_contact_success：ep_sum +107.911，active +17.7pp