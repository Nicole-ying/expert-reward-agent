# Subagent Research Signal

**训练过程**: Early→mid→late: avg_len 206→237→297, avg_score 941→1037→1729, gen_reward/step 6.71→6.62→8.05, orig_reward/step -90.3→-88.5→-82.1, crash_rate 65%→54%→44%. Agent learned to prolong episodes and maximize shaped reward, but crashes remain frequent.

**组件健康**: All components active; contact_bonus fires 32.9% of steps (mean=7.92 when active), making it the dominant positive contributor. total_reward per_step_mean=5.01 vs original_env_reward=-0.21. Large mismatch in scale and sign.

**奖励对齐**: Shaped reward (per-step ~5-8) far exceeds true task signal (original_env_reward -0.21). Evaluation mean=-46.7, all episodes truncated at 1000 steps, zero terminations. Rewards inflated but agent fails to land.

**异常检测**: None detected; training progression smooth but evaluation failure indicates silent exploitation.

**置信度**: `medium`
