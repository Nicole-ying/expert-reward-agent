# Subagent Research Signal

**训练过程**: Episode length increased (130→171) and shaped score rose (992.6→1267.9), but crash rate remained high (91%→83%). Shaped reward per step turned positive (-0.71→1.28) while original_env_reward stayed highly negative (-97→-95).

**组件健康**: Dense components (near, radial, slow, angle, act_pen, time_pen) all have nonzero rates ≥74%. Crash_pen fires rarely (3.6%) and landing_reward extremely rarely (1.3%), indicating near-dead components.

**奖励对齐**: Severe misalignment: shaped reward (1.28/step) far above original_env_reward (-95.1/step). Dense shaping rewards slow, upright movement near pad, which agent exploits without learning to land, as shown by 0/20 evaluation terminations and negative mean eval reward.

**异常检测**: Landing reward fires only 1.3% of steps during training. Final evaluation episodes all survive 1000 steps (no crashes) but mean eval reward is -90.8, consistent with hovering exploitation. Early crash rate plateaued at 83%.

**置信度**: `high`
