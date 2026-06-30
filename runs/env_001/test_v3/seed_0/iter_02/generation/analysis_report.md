# Analysis Report

## Recommended Action: tune
外部得分-111.79远低于目标200，且两轮无改善。progress_reward均值0.242弱，stability_penalty均值-0.331主导，导致penalty dominance。soft_landing_bonus触发率仅0.5%，稀疏无效。建议调大progress_reward系数至50以上，减小stability_penalty系数至0.1以下，并提高soft_landing_bonus触发条件或改为连续shaping。

## Skeleton Status
- family: progress+stability+landing_proxy
- stagnant: True
- iterations_on_skeleton: 2

## Component Analysis
- progress_reward: role=progress dir=positive issue=mean 0.242 is too low relative to stability_penalty mean -0.331; coefficient 15.0 insufficient to drive learning
- stability_penalty: role=constraint dir=negative issue=mean -0.331 dominates total reward; coefficient too high, causing penalty dominance
- soft_landing_bonus: role=proxy dir=positive issue=nonzero_rate 0.0051 indicates extremely rare triggering; sparse and ineffective

## Detected Issues
- failure_modes: stability_penalty_dominance, early_failure_or_crash
- hacking_risks: stability_penalty_dominance
