# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | gated_forward_speed + posture_hinge_penalty | -57.47 | -57.47 | 0.00 | 166.90 | gated_forward_speed=0.238 posture_hinge_penalty=-0.009 | new_best |
| 2 | gated_forward_speed + stability_quad_penalty | -62.54 | -57.47 | -5.07 | 167.40 | gated_forward_speed=0.164 stability_quad_penalty=-0.042 | no_meaningful_improvement |
| 3 | gated_forward_speed + stability_tilt_hinge_penalty | -49.54 | -49.54 | 0.00 | 163.65 | gated_forward_speed=0.232 stability_tilt_hinge_penalty=-0.007 | new_best |
| 4 | gated_forward_speed + stability_tilt_hinge_penalty | -49.54 | -49.54 | 0.00 | 163.65 | gated_forward_speed=0.232 stability_tilt_hinge_penalty=-0.007 | unsolved_stagnation_fresh_restart |
| 5 | action_cost + contact_transition_reward + forward_reward_gated | -42.42 | -42.42 | 0.00 | 406.50 | action_cost=-0.018 contact_transition_reward=0.097 forward_reward_gated=0.239 | new_best |
| 6 | action_cost + contact_transition_reward + forward_reward_gated | -62.76 | -42.42 | -20.34 | 359.70 | action_cost=-0.019 contact_transition_reward=0.049 forward_reward_gated=0.256 | no_meaningful_improvement |
| 7 | action_cost + contact_transition_reward + forward_reward_gated | -43.01 | -42.42 | -0.59 | 757.95 | action_cost=-0.018 contact_transition_reward=0.246 forward_reward_gated=0.155 | unsolved_stagnation_fresh_restart |
| 8 | energy_penalty + forward_reward + gated_forward + stability_gate | -50.62 | -42.42 | -8.20 | 230.80 | energy_penalty=-0.010 forward_reward=0.247 gated_forward=0.233 stability_gate=0.927 | no_meaningful_improvement |
| 9 | contact_transition_reward + energy_penalty + forward_reward + gated_forward + stability_gate | -67.14 | -42.42 | -24.73 | 256.40 | contact_transition_reward=0.067 energy_penalty=-0.011 forward_reward=0.150 gated_forward=0.144 stability_gate=0.951 | no_meaningful_improvement |
| 10 | contact_transition_reward + energy_penalty + forward_reward + gated_forward + roughness_penalty + stability_gate | -55.58 | -42.42 | -13.17 | 304.00 | contact_transition_reward=0.078 energy_penalty=-0.010 forward_reward=0.189 gated_forward=0.179 roughness_penalty=-0.001 | unsolved_stagnation_fresh_restart |
