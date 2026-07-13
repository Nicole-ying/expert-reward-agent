# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_progress + posture_penalty + soft_landing_bonus | -107.52 | -107.52 | 0.00 | 91.20 | distance_progress=0.015 posture_penalty=-0.002 soft_landing_bonus=0.035 | new_best |
| 2 | angle_global + angle_near + angvel_global + angvel_near + contact_reward + distance_progress | -40.47 | -40.47 | 0.00 | 983.30 | angle_global=-0.001 angle_near=-0.009 angvel_global=-0.000 angvel_near=-0.001 contact_reward=2.532 | new_best |
| 3 | angle_penalty + angvel_penalty + contact_reward + distance_progress + landing_bonus + speed_penalty | -62.67 | -40.47 | -22.20 | 979.90 | angle_penalty=-0.004 angvel_penalty=-0.000 contact_reward=3.991 distance_progress=0.026 landing_bonus=34.783 | no_meaningful_improvement |
| 4 | angle_penalty + angvel_penalty + contact_reward + distance_progress + landing_bonus | 130.64 | 130.64 | 0.00 | 647.80 | angle_penalty=-0.003 angvel_penalty=-0.001 contact_reward=1.713 distance_progress=0.008 landing_bonus=20.247 | new_best |
| 5 | alive_penalty + angle_penalty + angvel_penalty + contact_reward + distance_progress + landing_bonus | -119.30 | 130.64 | -249.93 | 876.50 | alive_penalty=-0.005 angle_penalty=-0.003 angvel_penalty=-0.001 contact_reward=1.587 distance_progress=0.020 | no_meaningful_improvement |
| 6 | angle_penalty + angvel_penalty + contact_reward + distance_progress + landing_bonus | -63.66 | 130.64 | -194.30 | 1000.00 | angle_penalty=-0.002 angvel_penalty=-0.001 contact_reward=1.602 distance_progress=0.009 landing_bonus=28.509 | no_meaningful_improvement |
| 7 | altitude_penalty + angle_penalty + angvel_penalty + contact_reward + distance_progress + fuel_penalty | -141.96 | 130.64 | -272.60 | 68.65 | altitude_penalty=-1.691 angle_penalty=-0.013 angvel_penalty=-0.004 contact_reward=0.598 distance_progress=0.145 | unsolved_high_achievement_continue_from_best |
