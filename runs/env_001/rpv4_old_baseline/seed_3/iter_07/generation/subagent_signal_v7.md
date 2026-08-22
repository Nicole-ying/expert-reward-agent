# Subagent Research Signal

**训练过程**: Agent improved: avg episode length 146→209, score 387→716, crash rate 86%→72%. Generated reward per step rose 2.92→3.59, original env reward slightly improved -89.4→-79.8. But crashes still dominate (72% late).

**组件健康**: All components non-zero. soft_landing (nonzero 28.7%, mean active 5.24) and contact_encouragement (28.7%, 4.56) dominate shaped reward. progress (0.10) and proximity (0.18) maintain constant but small presence. original_env_reward negligible (-0.37). No dead components.

**奖励对齐**: Large gap: generated_reward ~3.0/step positive vs original_env_reward ~-0.37. Shaped reward driven by soft_landing/contact components, not task success. Eval mean score -87.9, only 5/20 episodes terminated normally, indicating agent fails to survive or land despite reward gains. Likely exploitation of high-value intermittent triggers.

**异常检测**: Persistent high crash rate (72% late) despite training; original reward hardly improves; potential reward exploitation where agent occasionally triggers soft_landing/contact bonuses without stable landing, leading to suboptimal convergence.

**置信度**: `medium`
