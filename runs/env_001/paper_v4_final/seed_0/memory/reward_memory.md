# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + progress_reward + soft_landing | -111.81 | -111.81 | 0.00 | 68.45 | angle_penalty=-0.000 progress_reward=0.016 soft_landing=0.001 | new_best |
| 2 | angle_penalty + angular_velocity_penalty + progress_reward + soft_landing | -109.32 | -109.32 | 0.00 | 68.45 | angle_penalty=-0.000 angular_velocity_penalty=-0.001 progress_reward=0.016 soft_landing=0.001 | new_best |
| 3 | angle_penalty + angular_velocity_penalty + progress_reward + soft_landing | -81.43 | -81.43 | 0.00 | 69.50 | angle_penalty=-0.000 angular_velocity_penalty=-0.001 progress_reward=0.016 soft_landing=0.001 | new_best |
| 4 | angle_penalty + angular_velocity_penalty + progress_reward + soft_landing | 141.95 | 141.95 | 0.00 | 956.30 | angle_penalty=-0.000 angular_velocity_penalty=-0.000 progress_reward=0.004 soft_landing=0.032 | new_best |
| 5 | angle_penalty + contact_stability + progress_reward + soft_landing + success_bonus | -111.32 | 141.95 | -253.27 | 68.45 | angle_penalty=-0.000 contact_stability=0.001 progress_reward=0.016 soft_landing=0.001 success_bonus=0.238 | no_meaningful_improvement |
| 6 | angle_penalty + contact_stability + progress_reward + soft_landing + success_bonus | 243.15 | 243.15 | 0.00 | 386.15 | angle_penalty=-0.000 contact_stability=0.033 progress_reward=0.007 soft_landing=0.020 success_bonus=0.275 | target_solved_new_best |
| 7 | angular_velocity_penalty + contact_stability + progress_reward + soft_landing + success_bonus | 237.90 | 243.15 | -5.25 | 438.45 | angular_velocity_penalty=-0.000 contact_stability=0.048 progress_reward=0.003 soft_landing=0.027 success_bonus=0.402 | target_solved_no_improvement |
| 8 | angle_hinge_penalty + contact_stability + progress_reward + soft_landing + success_bonus | 180.67 | 243.15 | -62.48 | 340.30 | angle_hinge_penalty=-0.000 contact_stability=0.020 progress_reward=0.008 soft_landing=0.013 success_bonus=0.138 | stop_after_solved_drop_keep_best |
