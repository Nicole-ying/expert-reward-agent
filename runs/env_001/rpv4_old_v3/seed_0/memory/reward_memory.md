# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_prevention_x + crash_prevention_y + goal_proximity + living_penalty + soft_landing_penalty | -40.04 | -40.04 | 0.00 | 154.05 | crash_prevention_x=-0.000 crash_prevention_y=-0.007 goal_proximity=0.021 living_penalty=-0.002 soft_landing_penalty=-0.012 | new_best |
| 2 | crash_prevention_x + crash_prevention_y + goal_proximity + landing_success_reward + living_penalty | 217.21 | 217.21 | 0.00 | 484.10 | crash_prevention_x=-0.000 crash_prevention_y=-0.001 goal_proximity=0.013 landing_success_reward=0.312 living_penalty=-0.002 | target_solved_new_best |
| 3 | goal_proximity + landing_success_reward + living_penalty | 212.94 | 217.21 | -4.27 | 511.90 | goal_proximity=0.012 landing_success_reward=0.348 living_penalty=-0.002 | target_solved_no_improvement |
| 4 | goal_proximity + landing_success_reward + living_penalty | -56.07 | 217.21 | -273.28 | 109.20 | goal_proximity=0.032 landing_success_reward=0.005 living_penalty=-0.005 | stop_after_solved_drop_keep_best |
