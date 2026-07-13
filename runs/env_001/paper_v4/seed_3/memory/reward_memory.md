# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + orientation_penalty + proximity + velocity_penalty | -19.59 | -19.59 | 0.00 | 78.25 | contact_bonus=0.003 orientation_penalty=-0.044 proximity=-2.196 velocity_penalty=-0.211 | new_best |
| 2 | contact_bonus + orientation_penalty + progress + velocity_penalty | 147.71 | 147.71 | 0.00 | 1000.00 | contact_bonus=0.241 orientation_penalty=-0.010 progress=0.013 velocity_penalty=-0.018 | new_best |
| 3 | orientation_penalty + progress + settling_quality + velocity_penalty | -15.42 | 147.71 | -163.13 | 1000.00 | orientation_penalty=-0.012 progress=0.014 settling_quality=0.614 velocity_penalty=-0.020 | no_meaningful_improvement |
| 4 | landing_reward + orientation_penalty + progress + velocity_penalty | 253.71 | 253.71 | 0.00 | 362.10 | landing_reward=0.407 orientation_penalty=-0.012 progress=0.015 velocity_penalty=-0.023 | target_solved_new_best |
| 5 | engine_penalty + landing_reward + orientation_penalty + progress + velocity_penalty | 249.34 | 253.71 | -4.36 | 374.85 | engine_penalty=-0.031 landing_reward=0.411 orientation_penalty=-0.013 progress=0.016 velocity_penalty=-0.027 | target_solved_no_improvement |
| 6 | engine_penalty + landing_reward + orientation_penalty + progress + velocity_penalty | 247.21 | 253.71 | -6.50 | 410.40 | engine_penalty=-0.003 landing_reward=0.440 orientation_penalty=-0.011 progress=0.013 velocity_penalty=-0.020 | stop_solved_no_improvement_keep_best |
