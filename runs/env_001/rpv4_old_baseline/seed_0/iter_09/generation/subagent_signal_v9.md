# Subagent Research Signal

**训练过程**: Late avg_len=70, avg_score=91.1, gen_reward/step=6.255; early avg_score=83.9, gen_reward=5.601. Original_env_reward/step remained -99.98. Crash rate constant 100%. Agent exploited shaped reward, never learned to survive.

**组件健康**: Radial_reward active 100%, mean_when_active=1.91; vel_penalty 100%, -0.55; angle_penalty 100%, -0.11; descent_reward 93.4%, 0.044; engine_penalty 30.6%, -0.22; proximity_bonus 20.9%, 0.16; landing_bonus 3.4% nonzero, mean_when_active=118.3. All components fired but landing_bonus rare. Generated_reward=1.244/step.

**奖励对齐**: Mismatch: generated_reward/step positive (1.24) while original_env_reward/step negative (-1.59). Eval reward -122.8, 100% crashes. Shaped reward incentivizes radial velocity and altitude reduction, not survival.

**异常检测**: Early convergence to high shaped reward without decreasing crash rate. Crash rate 100% from start to end. Generated reward rose while original reward flat at -100. Reward exploitation evident.

**置信度**: `high`
