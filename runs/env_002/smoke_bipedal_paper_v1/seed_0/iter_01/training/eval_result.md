# External Evaluation Result

Evaluation uses the original environment reward, not the generated training reward.
All evaluations use the same fixed seed set for reproducible paired comparison.

- eval_episodes: 5
- eval_seed_offset: 10000
- mean_eval_reward: -96.500279
- mean_episode_length: 80.200000
- min_eval_reward: -97.216587
- max_eval_reward: -93.963102
- termination: 5 terminated, 0 truncated

## Episodes

| episode | eval_seed | reward | length | end |
|---:|---:|---:|---:|---|
| 0 | 10000 | -97.216587 | 74 | terminated |
| 1 | 10001 | -97.158223 | 73 | terminated |
| 2 | 10002 | -93.963102 | 106 | terminated |
| 3 | 10003 | -97.066841 | 75 | terminated |
| 4 | 10004 | -97.096641 | 73 | terminated |
