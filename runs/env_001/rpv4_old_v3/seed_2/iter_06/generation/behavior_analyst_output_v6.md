# 数据报告

## 1. 行为事实
- **得分**：score=-46.51，range=[-274.01, 201.88]  
- **终止方式**：terminated 7/20（成功或接触终止），truncated 13/20（达到时间上限 1000 步后被截断），实际回合长度 len=740.15  
- **Checkpoint 轨迹**：  
  - 40% 步数时官方得分 0.00  
  - 80% 步数时官方得分 0.00  
  得分在评估过程中始终为 0，直至最终阶段出现稀疏成功信号。

## 2. 骨架演进表

| iter | score | len | term | progress_shaping_ep_sum | progress_shaping_active | action_cost_ep_sum | action_cost_active | angle_penalty_ep_sum | angle_penalty_active | landing_velocity_penalty_ep_sum | landing_velocity_penalty_active | success_landing_bonus_ep_sum | success_landing_bonus_active | crash_penalty_ep_sum | crash_penalty_active |
|------|-------|-----|------|------------------------|----------------------|------------------|-----------------|--------------------|-------------------|-----------------------------|------------------------------|-------------------------|-------------------------|-------------------|-------------------|
| 1 | -155.98 | 154.4 | term=20 | 1.582 | 100.0% | -1.804 | 58.4% | -1.141 | 100.0% | -0.067 | 5.3% | — | — | — | — |
| 2 | -119.51 | 68.3 | term=20 | 5.583 | 100.0% | -0.071 | 5.2% | -0.205 | 100.0% | — | — | 5.0 | 0.1% | — | — |
| 3 | -203.45 | 154.2 | term=20 | 1.488 | 100.0% | -1.472 | 47.7% | -2.460 | 100.0% | — | — | 0.0 | 0.0% | 0.0 | 0.0% |
| 4 | -114.38 | 68.5 | term=20 | 5.597 | 100.0% | -0.097 | 7.1% | -0.087 | 100.0% | — | — | 30.0 | 0.4% | — | — |
| 5 | -46.51 | 740.2 | term=7, trunc=13 | 5.105 | 99.9% | -13.469 | 91.0% | -6.828 | 100.0% | — | — | 1195.0 | 1.6% | — | — |

*说明：空白项表示该轮未使用该组件；ep_sum 为 episode_sum_mean，active_rate 为 active_rate。*

## 3. 数字标记（基于当前轮 iter=5）
- **主导**：`success_landing_bonus` magnitude_share=97.7% >80%  
- **死组件**：`success_landing_bonus` active_rate=1.6% <2%  
- **淹没**：  
  - `action_cost` active_rate=91.0%，magnitude_share=1.1% <3%  
  - `progress_shaping` active_rate=99.9%，magnitude_share=0.7% <3%  
  - `angle_penalty` active_rate=100.0%，magnitude_share=0.6% <3%  
- **全覆盖惩罚**：`angle_penalty` active_rate=100% >95%，signed_share=-0.6% （为负）  
- **量级对比**（每步均值，ep_sum ÷ len）：  
  - 最大两个：`success_landing_bonus` (1.614)，`action_cost` (0.0182)  
  - 最小两个：`angle_penalty` (0.00922)，`progress_shaping` (0.00690)  
- **修改变化**（iter=5 相对于 iter=4 的 Δ）：
  - `success_landing_bonus`：ep_sum Δ = +1165.0，active_rate Δ = +1.2%
  - `action_cost`：ep_sum Δ = −13.372，active_rate Δ = +83.9%
  - `progress_shaping`：ep_sum Δ = −0.492，active_rate Δ = −0.1%
  - `angle_penalty`：ep_sum Δ = −6.741，active_rate Δ = 0%

## 4. 语义演化（跨轮组件功能追踪）

**分组依据**  
- **接近/进度信号**：基于位置/距离的引导（delta 形式）  
- **着陆/接触信号**：基于着陆事件/接触条件的奖励/惩罚  
- **稳定性/约束信号**：基于速度/角度/动作的惩罚或约束  
- **完成/成功代理**：大额条件性正信号（本任务无独立代理，由着陆信号承担）

| 语义功能 | 各轮组件名 | 各轮得分 | 方向切换 |
|---------|----------|--------|---------|
| 接近/进度信号 | progress_shaping (v1-v5) | -156, -120, -203, -114, -46.5 | 无切换（持续势能塑造） |
| 着陆/接触信号 | v1: —<br>v2: success_landing_bonus<br>v3: success_landing_bonus, crash_penalty<br>v4,5: success_landing_bonus | -156, -120, -203, -114, -46.5 | 稀疏bonus → 稀疏bonus+稀疏惩罚 → 稀疏bonus；切换次数：2 |
| 稳定性/约束信号 | v1: landing_velocity_penalty, angle_penalty, action_cost<br>v2-v5: angle_penalty, action_cost | -156, -120, -203, -114, -46.5 | v1→v2 移除门控速度惩罚，其余持续连续二次型/线性惩罚；对立设计切换次数：1 |

**着陆/接触信号方向切换详情**：  
- v2 引入稀疏大正奖励 (`success_landing_bonus`)，为单一边界。  
- v3 增加稀疏大惩罚 (`crash_penalty`) 并与成功奖励互斥，形成正/负对称稀疏信号。  
- v4 删除惩罚，回到仅稀疏奖励，奖励形状再次变为单侧正激励。  
由此在“稀疏bonus ↔ 稀疏bonus+稀疏惩罚”之间经历了两次改动。

**稳定性/约束信号切换详情**：  
- 仅一次从 v1 的门控高度‑速度惩罚 (`landing_velocity_penalty`) 彻底移除，后续轮次未再恢复，未出现往复振荡。