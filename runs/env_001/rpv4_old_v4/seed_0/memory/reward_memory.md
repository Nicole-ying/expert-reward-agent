# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_reward + orientation_penalty + progress_reward + vel_damping | -159.49 | -159.49 | 0.00 | 303.35 | contact_reward=0.000 orientation_penalty=-0.008 progress_reward=-0.002 vel_damping=-0.004 | new_best |
| 2 | landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -36.38 | -36.38 | 0.00 | 1000.00 | landing_proximity_reward=2.273 orientation_penalty=-0.005 progress_reward=0.001 vel_damping=-0.046 | new_best |
| 3 | landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -434.87 | -36.38 | -398.49 | 967.20 | landing_proximity_reward=0.013 orientation_penalty=-0.006 progress_reward=-0.001 vel_damping=-0.005 | no_meaningful_improvement |
| 4 | landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -59.19 | -36.38 | -22.81 | 977.90 | landing_proximity_reward=4.890 orientation_penalty=-0.008 progress_reward=0.002 vel_damping=-0.049 | no_meaningful_improvement |
| 5 | fuel_penalty + landing_proximity_reward + orientation_penalty + progress_reward + vel_damping | -124.22 | -36.38 | -87.84 | 68.35 | fuel_penalty=-0.010 landing_proximity_reward=0.040 orientation_penalty=-0.015 progress_reward=0.016 vel_damping=-0.172 | unsolved_stagnation_fresh_restart |
| 6 | approach_reward + fuel_penalty + orientation_penalty + vel_penalty | -119.14 | -36.38 | -82.76 | 68.45 | approach_reward=-0.970 fuel_penalty=-0.002 orientation_penalty=-0.004 vel_penalty=-0.154 | no_meaningful_improvement |
| 7 | fuel_penalty + orientation_penalty + progress_reward + vel_penalty | 252.78 | 252.78 | 0.00 | 308.20 | fuel_penalty=-0.010 orientation_penalty=-0.002 progress_reward=0.011 vel_penalty=-0.019 | target_solved_new_best |
| 8 | fuel_penalty + orientation_penalty + progress_reward + vel_penalty | -216.09 | 252.78 | -468.87 | 152.35 | fuel_penalty=-0.011 orientation_penalty=-0.002 progress_reward=-0.003 vel_penalty=-0.019 | stop_after_solved_drop_keep_best |
