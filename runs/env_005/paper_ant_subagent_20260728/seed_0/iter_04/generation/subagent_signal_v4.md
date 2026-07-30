# Subagent Research Signal

**Key Findings**: Score=1839.7 across ~981-step episodes; only 2/20 terminated (10%), 18/20 truncated. Per-step generated_reward=1.14 vs original_env_reward=-0.11 — the native env considers behavior suboptimal despite high generated scores. The agent survives long but rarely reaches terminal state.

**Component Anomalies**: gated_forward dominates at 90.3% signed share (episode_sum=3421). upright_bonus (+4.9%) and lateral_penalty (-4.5%) nearly cancel. height_penalty is dead: 2.7% active rate, 0.0% share — safe zone [0.3, 0.8] is never breached. action_penalty negligible at -0.1%.

**Training Dynamics**: No temporal snapshots available (component_dynamics returned empty). Cannot assess scaffold→final drift or checkpoint-level trends. All evidence is from final-policy fixed-seed evaluation.

**Signal Quality**: height_penalty dead (safe zone too wide — thresholds never triggered). original_env_reward is negative (-0.11/step) despite high generated reward — possible reward hacking where policy exploits gated_forward without truly solving the locomotion task. Low termination rate (10%) suggests the agent isn't reaching episode goals.

**Evidence Confidence**: `medium`
