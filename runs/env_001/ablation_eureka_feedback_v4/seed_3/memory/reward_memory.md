# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + proximity + soft_landing + terminal_velocity_penalty | -16.95 | -16.95 | 0.00 | 1000.00 | energy_penalty=-0.007 proximity=0.695 soft_landing=0.402 terminal_velocity_penalty=-0.000 | new_best |
| 2 | energy_penalty + proximity + soft_landing + terminal_velocity_penalty | 115.51 | 115.51 | 0.00 | 725.70 | energy_penalty=-0.008 proximity=0.679 soft_landing=0.761 terminal_velocity_penalty=-0.000 | new_best |
| 3 | energy_penalty + landing_success + proximity + terminal_velocity_penalty | -17.09 | 115.51 | -132.61 | 1000.00 | energy_penalty=-0.008 landing_success=0.844 proximity=0.657 terminal_velocity_penalty=-0.000 | no_meaningful_improvement |
| 4 | energy_penalty + landing_quality + proximity + terminal_velocity_penalty | -24.93 | 115.51 | -140.44 | 1000.00 | energy_penalty=-0.008 landing_quality=1.205 proximity=0.649 terminal_velocity_penalty=-0.000 | no_meaningful_improvement |
| 5 | energy_penalty + progress_gate_reward + terminal_velocity_penalty | -113.59 | 115.51 | -229.10 | 105.25 | energy_penalty=-0.003 progress_gate_reward=0.008 terminal_velocity_penalty=-0.001 | unsolved_high_achievement_continue_from_best |
| 6 | energy_penalty + landing_improvement + proximity + terminal_velocity_penalty | -2471.44 | 115.51 | -2586.95 | 303.35 | energy_penalty=-0.008 landing_improvement=0.021 proximity=0.012 terminal_velocity_penalty=-0.006 | no_meaningful_improvement |
| 7 | energy_penalty + landing_quality + proximity | 1.56 | 115.51 | -113.95 | 954.35 | energy_penalty=-0.007 landing_quality=0.999 proximity=0.004 | no_meaningful_improvement |
| 8 | energy_penalty + landing_event + proximity | -123.40 | 115.51 | -238.92 | 68.40 | energy_penalty=-0.001 landing_event=0.011 proximity=0.017 | unsolved_high_achievement_continue_from_best |
| 9 | energy_penalty + landing_event + proximity + soft_landing | 113.98 | 115.51 | -1.53 | 786.95 | energy_penalty=-0.006 landing_event=0.282 proximity=0.007 soft_landing=0.092 | no_meaningful_improvement |
| 10 | energy_penalty + landing_event + proximity + soft_landing | 115.19 | 115.51 | -0.32 | 845.40 | energy_penalty=-0.008 landing_event=0.599 proximity=0.003 soft_landing=0.070 | no_meaningful_improvement |
