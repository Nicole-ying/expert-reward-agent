# Subagent Research Signal

**训练过程**: Ep length flat at 74. Crash rate ~100% (early) → 99% (mid/late), no survival learned. Avg score fluctuated: -0.3→2.7→0.6, with no sustained improvement. Agent behavior stagnant.

**组件健康**: All components active (100% nonzero) except contact_reward (3.8%) and landing_bonus (0.4%). Dominant: descent_reward (mean 0.016), contact_reward (mean 0.113 when active 2.96). landing_bonus huge (200) but triggers 0.4%. Generated reward positive (0.013/step) vs. original -1.36/step.

**奖励对齐**: Generated reward per step rose (6.94→7.54) while original stayed ~-100. Shaped reward not aligned with task success; agent exploits positive components (e.g. descent) to get high generated reward while crashing 99% of the time. Clear reward hacking.

**异常检测**: Agent optimizes generated reward but fails task: crash rate >99%, generated reward climbs, original reward static. No divergence but severe reward-reality gap persists.

**置信度**: `high`
