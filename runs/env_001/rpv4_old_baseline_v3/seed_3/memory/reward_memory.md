# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + goal_proximity + orientation_penalty + velocity_penalty | -32.35 | -32.35 | 0.00 | 1000.00 | contact_bonus=2.905 goal_proximity=0.013 orientation_penalty=-0.008 velocity_penalty=-0.005 | new_best |
| 2 | contact_reward + fuel_cost + landing_reward + orient_cost + progress + proximity_bonus | -105.48 | -32.35 | -73.13 | 1000.00 | contact_reward=0.003 fuel_cost=-0.050 landing_reward=0.000 orient_cost=-0.013 progress=-0.002 | no_meaningful_improvement |
| 3 | contact_reward + descent_reward + fuel_cost + landing_bonus + orient_cost + progress | -121.57 | -32.35 | -89.22 | 546.45 | contact_reward=5.924 descent_reward=0.007 fuel_cost=-0.022 landing_bonus=4.246 orient_cost=-0.029 | no_meaningful_improvement |
| 4 | contact_reward + descent_reward + fuel_cost + landing_bonus + orient_cost + progress | -363.30 | -32.35 | -330.95 | 130.80 | contact_reward=0.004 descent_reward=0.010 fuel_cost=-0.008 landing_bonus=0.000 orient_cost=-0.048 | unsolved_stagnation_fresh_restart |
| 5 | progress_reward + touchdown_reward + velocity_penalty | 171.40 | 171.40 | 0.00 | 536.65 | progress_reward=0.027 touchdown_reward=2.522 velocity_penalty=-0.021 | new_best |
| 6 | fuel_penalty + orientation_penalty + progress_reward + stable_contact_reward + touchdown_bonus + velocity_penalty | -246.83 | 171.40 | -418.23 | 225.60 | fuel_penalty=-0.042 orientation_penalty=-0.064 progress_reward=-0.041 stable_contact_reward=0.002 touchdown_bonus=0.001 | no_meaningful_improvement |
| 7 | contact_reward + fuel_cost + orientation_penalty + progress_reward + proximity_reward + speed_penalty | 77.69 | 171.40 | -93.71 | 934.45 | contact_reward=1.303 fuel_cost=-0.016 orientation_penalty=-0.024 progress_reward=0.008 proximity_reward=1.692 | no_meaningful_improvement |
| 8 | contact_reward + fuel_cost + orientation_penalty + progress_reward + proximity_reward + speed_penalty | -119.62 | 171.40 | -291.02 | 68.50 | contact_reward=0.299 fuel_cost=-0.007 orientation_penalty=-0.012 progress_reward=0.032 proximity_reward=0.043 | unsolved_high_achievement_continue_from_best |
| 9 | contact_reward + fuel_cost + orientation_penalty + progress_reward + proximity_reward + speed_penalty | -222.79 | 171.40 | -394.19 | 891.15 | contact_reward=5.097 fuel_cost=-0.023 orientation_penalty=-0.078 progress_reward=0.007 proximity_reward=1.717 | no_meaningful_improvement |
| 10 | fuel_cost + landing_reward + orientation_penalty + progress_reward + speed_penalty + touchdown_bonus | -124.58 | 171.40 | -295.98 | 522.75 | fuel_cost=-0.022 landing_reward=18.678 orientation_penalty=-0.058 progress_reward=0.015 speed_penalty=-0.016 | no_meaningful_improvement |
