# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_safety_penalty + progress_reward + x_boundary_penalty | 39.61 | 39.61 | 0.00 | 211.25 | landing_safety_penalty=0.003 progress_reward=0.007 x_boundary_penalty=0.000 | new_best |
| 2 | landing_contact_bonus + landing_safety_penalty + progress_reward + x_boundary_penalty | 97.70 | 97.70 | 0.00 | 872.65 | landing_contact_bonus=0.283 landing_safety_penalty=0.003 progress_reward=0.003 x_boundary_penalty=0.000 | new_best |
| 3 | landing_contact_bonus + landing_safety_penalty + progress_reward + x_boundary_penalty | 146.75 | 146.75 | 0.00 | 1000.00 | landing_contact_bonus=0.100 landing_safety_penalty=0.003 progress_reward=0.002 x_boundary_penalty=0.000 | new_best |
| 4 | landing_safety_penalty + precise_landing_bonus + progress_reward + x_boundary_penalty | -117.67 | 146.75 | -264.42 | 68.40 | landing_safety_penalty=0.016 precise_landing_bonus=0.066 progress_reward=0.016 x_boundary_penalty=0.000 | no_meaningful_improvement |
| 5 | landing_contact_bonus + landing_safety_penalty + progress_reward | -8.04 | 146.75 | -154.79 | 1000.00 | landing_contact_bonus=0.847 landing_safety_penalty=0.002 progress_reward=0.004 | no_meaningful_improvement |
| 6 | approach_bonus + landing_safety_penalty + progress_reward | 219.70 | 219.70 | 0.00 | 446.40 | approach_bonus=0.716 landing_safety_penalty=0.001 progress_reward=0.002 | target_solved_new_best |
| 7 | approach_bonus + landing_safety_penalty + progress_reward | 232.57 | 232.57 | 0.00 | 460.10 | approach_bonus=0.657 landing_safety_penalty=0.001 progress_reward=0.002 | target_solved_new_best |
| 8 | approach_bonus + landing_safety_penalty + progress_reward | 144.26 | 232.57 | -88.31 | 1000.00 | approach_bonus=0.400 landing_safety_penalty=0.002 progress_reward=0.002 | stop_after_solved_drop_keep_best |
