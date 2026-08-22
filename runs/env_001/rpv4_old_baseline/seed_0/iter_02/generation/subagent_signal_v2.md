# Subagent Research Signal

**训练过程**: Early->mid: crash rate 44%→24%, score -2.7→-2.3, len 158→178. Mid->late: plateau, score -2.3, crash 24-25%, len 179. Per-step shaped reward improved -0.070→-0.042.

**组件健康**: progress_reward active 99.9% (mean +0.0071). stability_penalty 100% (-0.0148). action_penalty 64.5% (-0.0100 when active). original_env_reward 100% (-0.0029). No dead components.

**奖励对齐**: Shaped reward (mean -0.0142/step) far more negative than original (-0.0029) due to high stability penalties. Eval episodes achieved positive scores (mean +25.37). Gap exists but agent still learned task.

**异常检测**: Not reported.

**置信度**: `high`
