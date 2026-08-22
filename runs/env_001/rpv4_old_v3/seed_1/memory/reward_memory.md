# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_avoidance + goal_approach + soft_landing | -214.16 | -214.16 | 0.00 | 182.50 | crash_avoidance=-0.001 goal_approach=0.001 soft_landing=-0.009 | new_best |
| 2 | crash_avoidance + goal_approach + soft_landing | -109.58 | -109.58 | 0.00 | 68.45 | crash_avoidance=-0.002 goal_approach=0.016 soft_landing=-0.282 | new_best |
| 3 | goal_approach + landing_success + soft_landing | -115.21 | -109.58 | -5.63 | 68.35 | goal_approach=0.016 landing_success=0.536 soft_landing=-0.290 | no_meaningful_improvement |
| 4 | contact_reward + goal_approach + soft_landing | -111.06 | -109.58 | -1.49 | 68.40 | contact_reward=0.017 goal_approach=0.016 soft_landing=-0.281 | no_meaningful_improvement |
| 5 | goal_approach + landing_success + soft_landing | -119.00 | -109.58 | -9.43 | 68.40 | goal_approach=0.016 landing_success=0.643 soft_landing=-0.288 | unsolved_stagnation_fresh_restart |
| 6 | goal_approach + landing_comfort + soft_landing | -108.18 | -108.18 | 0.00 | 68.45 | goal_approach=0.016 landing_comfort=0.006 soft_landing=-0.280 | unsolved_stagnation_fresh_restart |
| 7 | contact + progress | 255.28 | 255.28 | 0.00 | 351.65 | contact=0.111 progress=0.001 | target_solved_new_best |
| 8 | contact + progress | 155.50 | 255.28 | -99.78 | 955.40 | contact=0.122 progress=0.001 | stop_after_solved_drop_keep_best |
