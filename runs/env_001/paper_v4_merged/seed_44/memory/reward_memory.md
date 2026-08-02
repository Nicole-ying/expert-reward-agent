# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + contact_bonus + progress + speed_penalty | -18.89 | -18.89 | 0.00 | 1000.00 | angle_penalty=-0.027 contact_bonus=1.048 progress=0.029 speed_penalty=-0.009 | new_best |
| 2 | angle_penalty + landing_reward + progress + speed_penalty | 144.30 | 144.30 | 0.00 | 1000.00 | angle_penalty=-0.015 landing_reward=0.007 progress=0.025 speed_penalty=-0.008 | new_best |
| 3 | angle_penalty + contact_reward + landing_reward + progress | 2.30 | 144.30 | -142.00 | 506.35 | angle_penalty=-0.032 contact_reward=2.522 landing_reward=0.004 progress=0.031 | no_meaningful_improvement |
| 4 | angle_penalty + contact_reward + landing_reward + progress | 250.95 | 250.95 | 0.00 | 333.85 | angle_penalty=-0.046 contact_reward=0.000 landing_reward=0.001 progress=0.099 | target_solved_new_best |
| 5 | angle_penalty + contact_reward + landing_reward + progress | 222.88 | 250.95 | -28.08 | 552.50 | angle_penalty=-0.035 contact_reward=0.016 landing_reward=0.003 progress=0.078 | target_solved_no_improvement |
| 6 | angle_penalty + contact_landing_bonus + landing_reward + progress | 130.57 | 250.95 | -120.38 | 778.90 | angle_penalty=-0.028 contact_landing_bonus=0.778 landing_reward=0.005 progress=0.033 | stop_after_solved_drop_keep_best |
