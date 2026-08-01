# Compressed Failure and Reward-Misalignment Cards

Use these cards only after training feedback is available.
Select only the matched cards for the next iteration context.
Do not pass the whole file to the reward generator.

## speed_penalty_dominance

- signal: abs(speed_penalty_mean) > abs(progress_reward_mean)
- risk: agent may learn to avoid motion instead of solving the task
- fix: reduce speed penalty, delay it, or apply it mainly near landing-critical states

## stability_penalty_dominance

- signal: abs(stability_penalty_mean) > abs(progress_reward_mean)
- risk: agent may become overly conservative or afraid to move
- fix: reduce angle/angular-velocity penalty or make it conditional near target

## sparse_completion_proxy

- signal: completion/landing bonus trigger_rate < 1%
- risk: final bonus provides little early learning guidance
- fix: replace hard bonus with smoother landing-quality shaping

## early_failure_or_crash

- signal: negative external score and short episode length
- risk: reward does not guide stable control before termination
- fix: add smooth approach/landing signals; avoid relying on sparse terminal-like proxy

## high_reward_without_success

- signal: generated_reward improves but external score stays poor
- risk: policy optimizes the custom reward but not the real task
- fix: reduce exploitable terms; add constraints tied to actual task progress and stable outcome

## goal_near_oscillation

- signal: distance/progress improves but episode length is long and landing proxy rarely triggers
- risk: agent hovers or oscillates around the goal without completing the task
- fix: add smooth low-speed, low-angle, near-target shaping; avoid pure distance reward

## contact_reward_hacking

- signal: contact-related reward triggers without good external score
- risk: agent exploits contact flags without safe task completion
- fix: require near target, low speed, stable angle, and both supports before using contact

## generated_reward_negative_average

- signal: total generated reward mean is negative during most training steps
- risk: agent receives mostly punitive feedback and may not discover useful behavior
- fix: rebalance progress and penalty terms; avoid large always-on penalties in early versions
