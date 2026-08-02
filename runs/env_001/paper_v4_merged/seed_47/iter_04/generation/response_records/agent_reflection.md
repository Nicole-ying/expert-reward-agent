# Response Record

1. **evidence**  
   Iter 3 得分 -73.9，ep_len 815，terminated 30%；success_bonus 从未激活，landing_potential_diff 占据 magnitude_share 84.4%，contact_bonus 几乎不触发（0.1%）。Iter 2 曾以 simpler landing_reward + progress_reward 获得 -24.96（当前最佳）。这表明上一轮的 potential_diff 主导和死寂 success_bonus 是主要失败原因。

2. **behavior_diagnosis**  
   agent 在后期迭代中学习到以不安全的高速下降来获取 landing_potential_diff 的正向信号，导致 episode 提前终止（crash 或出界），得分急剧恶化。同时 dead success_bonus 无法提供成功指引。

3. **signal_completeness**  
   任务需要① 接近目标的密集进展信号 ② 姿态安全约束 ③ 着陆质量（低速、竖直、接触）的连续反馈 ④ 避免硬着陆的速度惩罚。当前缺失③和④的有效表达。

4. **selected_level**  
   Level 3 REBUILD：连续三轮未刷新 best，且
