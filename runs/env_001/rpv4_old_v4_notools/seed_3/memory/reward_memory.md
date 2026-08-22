# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | -32.35 | -32.35 | 0.00 | 1000.00 | contact_bonus=2.905 goal_proximity=0.013 orientation_penalty=-0.008 velocity_penalty=-0.005 | new_best |
| 2 | contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | -27.46 | -27.46 | 0.00 | 1000.00 | contact_bonus=2.253 goal_proximity=0.008 orientation_penalty=-0.014 velocity_penalty=-0.004 | new_best |
| 3 | contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | -27.46 | -27.46 | 0.00 | 1000.00 | contact_bonus=2.253 goal_proximity=0.008 orientation_penalty=-0.014 velocity_penalty=-0.004 | no_meaningful_improvement |
| 4 | goal_proximity + landing_progress + orientation_penalty + velocity_penalty | -111.62 | -27.46 | -84.16 | 68.45 | goal_proximity=0.104 landing_progress=0.752 orientation_penalty=-0.008 velocity_penalty=-0.035 | unsolved_stagnation_fresh_restart |
| 5 | crash_penalty + landing_reward + progress_reward | -307.97 | -27.46 | -280.52 | 956.80 | crash_penalty=-0.000 landing_reward=5.949 progress_reward=0.004 | no_meaningful_improvement |
| 6 | crash_penalty + landing_reward + progress_reward | -307.97 | -27.46 | -280.52 | 956.80 | crash_penalty=-0.000 landing_reward=5.949 progress_reward=0.004 | no_meaningful_improvement |
| 7 | crash_penalty + progress_reward + touchdown_bonus | -113.33 | -27.46 | -85.87 | 747.55 | crash_penalty=0.000 progress_reward=0.029 touchdown_bonus=62.435 | unsolved_stagnation_fresh_restart |
| 8 | approach_delta_gated + contact_alignment + energy_penalty + soft_landing_bonus + stability_penalty | -110.44 | -27.46 | -82.98 | 68.45 | approach_delta_gated=0.001 contact_alignment=0.002 energy_penalty=-0.000 soft_landing_bonus=0.005 stability_penalty=-0.005 | no_meaningful_improvement |
| 9 | attitude_penalty + contact_reward + energy_penalty + landing_bonus + progress_reward + speed_penalty | -119.09 | -27.46 | -91.63 | 68.45 | attitude_penalty=-0.019 contact_reward=0.003 energy_penalty=0.000 landing_bonus=0.611 progress_reward=0.016 | no_meaningful_improvement |
| 10 | attitude_penalty + contact_reward + energy_penalty + landing_bonus + progress_reward + speed_penalty | -119.09 | -27.46 | -91.63 | 68.45 | attitude_penalty=-0.019 contact_reward=0.003 energy_penalty=0.000 landing_bonus=0.611 progress_reward=0.016 | unsolved_stagnation_fresh_restart |
