# Subagent Research Signal

**训练过程**: Early episodes short (len=422) with high crash rate (33%). Mid-stage improved: length up to 567, crash down to 15%, generated reward rising. Late-stage length slightly dropped to 544, crash rose to 17%, but generated reward still increased, suggesting exploitation of shaped reward rather than genuine task progress.

**组件健康**: landing_bonus: nonzero_rate=27%, mean=200 when active, dominates total. speed_reward, angle_reward: always active with stable means. proximity_reward: small negative. All components active, no dead components.

**奖励对齐**: Major gap: generated_reward/step grew from 3.9 to 5.96 while original_env_reward remained negative per episode (-73.2 to -59.9). Landing bonus fires on 27% of steps, implying the agent learns to freeze in a posture that meets conditions rather than completing the task. Eval episodes all run to 1000 steps, no early termination, and true reward scores remain negative.

**异常检测**: Crash rate re-increased in late stage while generated reward kept rising, indicating possible overfitting to landing_bonus exploitation. No catastrophic divergence but evidence of stationary reward farming.

**置信度**: `high`
