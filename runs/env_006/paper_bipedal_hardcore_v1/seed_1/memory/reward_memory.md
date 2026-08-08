# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_efficiency_penalty + progress_reward + stability_penalty | -18.01 | -18.01 | 0.00 | 411.40 | action_efficiency_penalty=-0.018 progress_reward=0.218 stability_penalty=-0.010 | new_best |
| 2 | action_efficiency_penalty + angular_velocity_penalty + progress_reward + stability_penalty | -50.60 | -18.01 | -32.59 | 222.20 | action_efficiency_penalty=-0.018 angular_velocity_penalty=-0.000 progress_reward=0.202 stability_penalty=-0.008 | no_meaningful_improvement |
| 3 | action_efficiency_penalty + angular_velocity_penalty + progress_reward | -36.96 | -18.01 | -18.95 | 255.70 | action_efficiency_penalty=-0.017 angular_velocity_penalty=-0.000 progress_reward=0.267 | no_meaningful_improvement |
| 4 | action_efficiency_penalty + angular_velocity_penalty + progress_reward + vertical_velocity_penalty | -36.96 | -18.01 | -18.95 | 255.70 | action_efficiency_penalty=-0.017 angular_velocity_penalty=-0.000 progress_reward=0.267 vertical_velocity_penalty=0.000 | unsolved_stagnation_fresh_restart |
| 5 | angle_gate + health_gate + progress_raw + vertical_gate | -52.26 | -18.01 | -34.25 | 297.45 | angle_gate=0.793 health_gate=0.766 progress_raw=0.235 vertical_gate=0.965 | no_meaningful_improvement |
| 6 | angle_gate + health_gate + progress_raw + vertical_gate | -61.63 | -18.01 | -43.62 | 240.35 | angle_gate=0.729 health_gate=0.816 progress_raw=0.220 vertical_gate=0.960 | no_meaningful_improvement |
| 7 | angle_gate + falling_risk_penalty + health_gate + progress_raw + vertical_gate | -55.66 | -18.01 | -37.65 | 278.15 | angle_gate=0.742 falling_risk_penalty=0.017 health_gate=0.821 progress_raw=0.225 vertical_gate=0.965 | unsolved_stagnation_fresh_restart |
| 8 | forward_progress + stability_angle_penalty + stability_angvel_penalty + vertical_speed_penalty | -61.41 | -18.01 | -43.40 | 326.70 | forward_progress=0.195 stability_angle_penalty=-0.011 stability_angvel_penalty=0.000 vertical_speed_penalty=-0.000 | no_meaningful_improvement |
| 9 | forward_progress + ground_penalty + stability_angle_penalty | -37.48 | -18.01 | -19.47 | 370.40 | forward_progress=0.247 ground_penalty=-0.002 stability_angle_penalty=-0.012 | no_meaningful_improvement |
| 10 | action_penalty + forward_progress + stability_angle_penalty | -58.44 | -18.01 | -40.43 | 197.75 | action_penalty=-0.035 forward_progress=0.218 stability_angle_penalty=-0.011 | unsolved_stagnation_fresh_restart |
