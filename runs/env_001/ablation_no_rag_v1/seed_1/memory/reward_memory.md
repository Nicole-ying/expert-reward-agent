# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_progress + tilt_penalty | -100.51 | -100.51 | 0.00 | 68.55 | distance_progress=0.016 tilt_penalty=-0.002 | new_best |
| 2 | distance_progress + tilt_penalty + velocity_penalty | -110.36 | -100.51 | -9.85 | 68.45 | distance_progress=0.016 tilt_penalty=-0.002 velocity_penalty=-0.092 | no_meaningful_improvement |
| 3 | distance_progress + tilt_penalty + vertical_accel_reward | -107.65 | -100.51 | -7.14 | 68.40 | distance_progress=0.016 tilt_penalty=-0.001 vertical_accel_reward=0.002 | no_meaningful_improvement |
| 4 | distance_progress + landing_support + tilt_penalty | 18.39 | 18.39 | 0.00 | 321.60 | distance_progress=0.014 landing_support=0.043 tilt_penalty=-0.004 | new_best |
| 5 | distance_progress + landing_quality + tilt_penalty | -8.89 | 18.39 | -27.28 | 1000.00 | distance_progress=0.007 landing_quality=0.656 tilt_penalty=-0.006 | no_meaningful_improvement |
| 6 | distance_progress + landing_guidance + tilt_penalty | 32.83 | 32.83 | 0.00 | 704.20 | distance_progress=0.005 landing_guidance=0.905 tilt_penalty=-0.008 | new_best |
| 7 | distance_progress + landing_guidance + tilt_penalty | -79.48 | 32.83 | -112.31 | 69.65 | distance_progress=0.159 landing_guidance=0.005 tilt_penalty=-0.003 | no_meaningful_improvement |
| 8 | distance_progress + landing_guidance + tilt_penalty | 78.23 | 78.23 | 0.00 | 328.10 | distance_progress=0.023 landing_guidance=0.175 tilt_penalty=-0.005 | new_best |
| 9 | distance_progress + landing_guidance + tilt_penalty | 128.95 | 128.95 | 0.00 | 682.60 | distance_progress=0.014 landing_guidance=7.800 tilt_penalty=-0.010 | new_best |
| 10 | distance_progress + landing_guidance + tilt_penalty | -75.71 | 128.95 | -204.67 | 107.30 | distance_progress=0.027 landing_guidance=1.734 tilt_penalty=-0.013 | no_meaningful_improvement |
