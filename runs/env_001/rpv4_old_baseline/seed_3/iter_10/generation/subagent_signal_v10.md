# Subagent Research Signal

**训练过程**: Agent behavior improved: episode length rose from 176 to 227, score from 765.9 to 1593.0 across early→late periods. Crash rate declined from 70% to 62%. Generated reward per step increased from 5.785 to 7.728, original env reward improved from -94.113 to -87.854.

**组件健康**: All cost/progress components had 100% nonzero rate. landing_bonus nonzero 12.8% (mean=500 when active). contact_reward nonzero 32.7%, vy_cost 43.5%, engine_penalty 78.7%. No dead components.

**奖励对齐**: Generated reward per step strongly positive (5.59 avg) while original env reward per step heavily negative (-0.94). The gap persists; agent learns shaped reward but original reward only modestly improves. Possible exploitation: landing_bonus spikes reward despite 62% crash rate, indicating inconsistent landing success.

**异常检测**: Crash rate remains high (62% late) despite score gains. Might reflect early convergence to a suboptimal policy that exploits landing_bonus without consistently safe landings.

**置信度**: `medium`
