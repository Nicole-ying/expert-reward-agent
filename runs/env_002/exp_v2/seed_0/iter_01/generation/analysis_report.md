# Analysis Report

## Recommended Action: tune
当前骨架仅运行1轮，得分-102.55远低于目标200。stability_penalty 出现极大负值（-334.8），主导总奖励，导致agent不敢动。progress_delta_reward 均值接近0，驱动力不足。soft_landing_bonus 从未触发，条件过严。best_reward.py与当前代码完全相同，无需revert。建议：降低stability_penalty权重（如从0.5/0.3降至0.1/0.05），增大progress_delta_reward权重（如从2.0增至5.0），放宽soft_landing_bonus条件（如降低速度下限至0.2）。

## Skeleton Status
- family: progress+stability+landing_proxy+energy
- stagnant: False
- iterations_on_skeleton: 1

## Component Analysis
- energy_penalty: role=efficiency dir=negative issue=none
- progress_delta_reward: role=progress dir=positive issue=mean near zero, indicating insufficient drive
- soft_landing_bonus: role=proxy dir=positive issue=never triggered (nonzero_rate=0), too strict conditions
- stability_penalty: role=constraint dir=negative issue=large negative spikes (min=-334.8) dominate total reward

## Detected Issues
- failure_modes: stability_penalty_dominance, generated_reward_negative_average
- hacking_risks: stability_penalty_dominance
