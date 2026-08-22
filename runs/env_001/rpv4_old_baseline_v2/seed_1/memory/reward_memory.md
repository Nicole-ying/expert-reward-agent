# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | initial | -42.74 | -42.74 | 0.0 | 975.6 | initial | new_best |
| 2 | angle_penalty + angvel_penalty + contact_reward + progress + vel_penalty | -60.94 | -42.74 | -18.20 | 975.75 | angle_penalty=-0.014 angvel_penalty=-0.005 contact_reward=2.173 progress=0.008 vel_penalty=-0.048 | no_meaningful_improvement |
| 3 | angle_penalty + angvel_penalty + contact_reward + distance_penalty + vel_penalty | 52.52 | 52.52 | 0.00 | 507.20 | angle_penalty=-0.017 angvel_penalty=-0.014 contact_reward=10.234 distance_penalty=-0.384 vel_penalty=-0.016 | new_best |
| 4 | angle_penalty + angvel_penalty + contact_reward + dist_reward + distance_penalty + success_bonus | -80.35 | 52.52 | -132.87 | 813.75 | angle_penalty=-0.015 angvel_penalty=-0.001 contact_reward=1.873 dist_reward=0.003 distance_penalty=-0.056 | no_meaningful_improvement |
| 5 | angle_penalty + angvel_penalty + contact_reward + progress + proximity + success_bonus | -46.32 | 52.52 | -98.84 | 1000.00 | angle_penalty=-0.015 angvel_penalty=-0.001 contact_reward=0.817 progress=0.016 proximity=0.351 | no_meaningful_improvement |
| 6 | angle_penalty + angvel_penalty + contact_reward + distance_penalty + success_bonus + vel_penalty | -47.43 | 52.52 | -99.95 | 75.05 | angle_penalty=-0.009 angvel_penalty=-0.015 contact_reward=1.169 distance_penalty=-0.484 success_bonus=0.014 | unsolved_high_achievement_continue_from_best |
| 7 | angle_penalty + angvel_penalty + contact_reward + distance_penalty + success_bonus + vel_penalty | 204.38 | 204.38 | 0.00 | 296.45 | angle_penalty=-0.017 angvel_penalty=-0.004 contact_reward=7.641 distance_penalty=-0.363 success_bonus=0.183 | target_solved_new_best |
| 8 | angle_penalty + angvel_penalty + contact_reward + distance_penalty + success_bonus + vel_penalty | -16.94 | 204.38 | -221.32 | 434.65 | angle_penalty=-0.013 angvel_penalty=-0.005 contact_reward=2.634 distance_penalty=-0.377 success_bonus=0.185 | stop_after_solved_drop_keep_best |
