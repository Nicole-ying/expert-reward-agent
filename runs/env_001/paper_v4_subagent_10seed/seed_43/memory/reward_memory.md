# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + progress + soft_landing_penalty | -115.30 | -115.30 | 0.00 | 775.10 | landing_bonus=7.699 progress=0.040 soft_landing_penalty=0.203 | new_best |
| 2 | landing_approach_reward + progress + soft_landing_penalty | -18.80 | -18.80 | 0.00 | 1000.00 | landing_approach_reward=2.424 progress=0.036 soft_landing_penalty=0.198 | new_best |
| 3 | contact_success_reward + progress + soft_landing_penalty | -112.84 | -18.80 | -94.04 | 501.05 | contact_success_reward=42.715 progress=0.051 soft_landing_penalty=0.281 | no_meaningful_improvement |
| 4 | contact_success_reward + landing_gate + progress | -115.49 | -18.80 | -96.69 | 131.15 | contact_success_reward=0.722 landing_gate=0.145 progress=0.117 | no_meaningful_improvement |
| 5 | contact_success_reward + landing_approach_reward + progress | -55.78 | -18.80 | -36.97 | 1000.00 | contact_success_reward=50.158 landing_approach_reward=0.424 progress=0.029 | unsolved_stagnation_fresh_restart |
| 6 | action_cost + angle_penalty + boundary_penalty + landing_soft_reward + progress | -117.78 | -18.80 | -98.98 | 68.30 | action_cost=-0.003 angle_penalty=-0.001 boundary_penalty=0.000 landing_soft_reward=0.013 progress=0.013 | no_meaningful_improvement |
| 7 | action_cost + angle_penalty + landing_soft_reward + progress + safety_penalty | -80.85 | -18.80 | -62.05 | 103.55 | action_cost=-0.006 angle_penalty=-0.002 landing_soft_reward=0.033 progress=0.010 safety_penalty=-0.019 | no_meaningful_improvement |
| 8 | action_cost + gate_factor + shaping + success_bonus | -95.67 | -18.80 | -76.87 | 71.60 | action_cost=-0.011 gate_factor=0.883 shaping=0.028 success_bonus=1.177 | unsolved_stagnation_fresh_restart |
| 9 | angvel_penalty + contact_success + progress | 174.56 | 174.56 | 0.00 | 710.05 | angvel_penalty=-0.002 contact_success=1.613 progress=0.008 | new_best |
| 10 | angvel_penalty + contact_success + progress | -33.83 | 174.56 | -208.39 | 1000.00 | angvel_penalty=-0.002 contact_success=1.769 progress=0.009 | no_meaningful_improvement |
