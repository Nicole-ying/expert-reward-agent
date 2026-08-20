# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | orientation_penalty + proximity_delta + velocity_penalty | -116.46 | -116.46 | 0.00 | 134.15 | orientation_penalty=-0.003 proximity_delta=0.005 velocity_penalty=-0.002 | new_best |
| 2 | orientation_penalty + proximity_delta + velocity_danger | -110.22 | -110.22 | 0.00 | 68.45 | orientation_penalty=-0.081 proximity_delta=0.776 velocity_danger=-0.117 | new_best |
| 3 | landing_bonus + orientation_penalty + proximity_delta + velocity_danger | -114.97 | -110.22 | -4.74 | 68.40 | landing_bonus=0.044 orientation_penalty=-0.078 proximity_delta=0.781 velocity_danger=-0.118 | no_meaningful_improvement |
| 4 | landing_bonus + orientation_penalty + proximity_delta + velocity_danger | -111.88 | -110.22 | -1.66 | 68.40 | landing_bonus=0.010 orientation_penalty=-0.112 proximity_delta=0.787 velocity_danger=-0.120 | no_meaningful_improvement |
| 5 | orientation_penalty + proximity_delta + soft_approach_bonus + velocity_danger | -115.17 | -110.22 | -4.94 | 68.40 | orientation_penalty=-0.074 proximity_delta=0.789 soft_approach_bonus=0.015 velocity_danger=-0.121 | unsolved_stagnation_fresh_restart |
| 6 | A_progress_gated + C_landing_steady | 135.20 | 135.20 | 0.00 | 917.45 | A_progress_gated=0.001 C_landing_steady=0.137 | new_best |
| 7 | A_progress_gated + C_landing_steady | 145.47 | 145.47 | 0.00 | 1000.00 | A_progress_gated=0.011 C_landing_steady=0.139 | new_best |
| 8 | A_progress_gated + C_landing_steady | 141.79 | 145.47 | -3.68 | 955.20 | A_progress_gated=0.010 C_landing_steady=0.087 | no_meaningful_improvement |
| 9 | A_progress_gated + C_landing_steady | 145.89 | 145.89 | 0.00 | 1000.00 | A_progress_gated=0.010 C_landing_steady=0.082 | new_best |
| 10 | A_progress_gated + C_landing_steady | 145.99 | 145.99 | 0.00 | 1000.00 | A_progress_gated=0.028 C_landing_steady=0.079 | unsolved_high_achievement_continue_from_best |
