# Analysis Report

## Recommended Action: revert
当前得分-76.01远低于best得分11.95，差距87.96。对比代码，current将progress_delta_reward系数从5.0提升至15.0，但信号仍弱且出现大幅负值；stability_penalty系数和阈值收紧但惩罚过弱；soft_landing_bonus权重降低且条件收紧导致几乎不触发。这些修改均导致性能下降。建议恢复到best_reward的系数（progress_delta_reward=5.0, stability_penalty系数0.1/0.05, soft_landing_bonus=0.5且条件放宽），仅在此基础上小幅调整。

## Skeleton Status
- family: progress+stability+landing_proxy+anchor
- stagnant: True
- iterations_on_skeleton: 3

## Component Analysis
- progress_delta_reward: role=progress dir=positive issue=系数从5.0提升至15.0，但mean仅0.051，abs_mean 0.274，说明信号强度不足，且min=-10.29表明存在大幅负值，可能引起震荡
- stability_penalty: role=constraint dir=negative issue=系数和阈值收紧后，mean=-0.0056，nonzero_rate=0.142，惩罚过弱，无法有效约束姿态
- energy_penalty: role=efficiency dir=negative issue=none
- soft_landing_bonus: role=proxy dir=positive issue=权重从0.5降至0.2，条件收紧，nonzero_rate仅0.042，几乎不触发，无法提供有效引导

## Detected Issues
- failure_modes: goal_near_oscillation, high_reward_without_success
- hacking_risks: contact_reward_hacking
