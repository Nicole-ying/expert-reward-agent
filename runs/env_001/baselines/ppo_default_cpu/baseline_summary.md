# LunarLander-v3 PPO Baseline (SB3 Defaults, CPU)

- **env**: LunarLander-v3
- **seed**: 0
- **device**: cpu
- **total_timesteps**: 1,000,000
- **start**: 2026-07-08T18:50:17.150374
- **end**: 2026-07-08T19:15:02.587764
- **train_time**: 24.3 min

## PPO Parameters (SB3 defaults)

| param | value |
|-------|-------|
| n_steps | 2048 |
| batch_size | 64 |
| n_epochs | 10 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |
| ent_coef | 0.0 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| lr | 3e-4 |
| net_arch | pi=[64,64], vf=[64,64] |
| activation | Tanh |
| ortho_init | True |

## Final Evaluation (100 episodes, deterministic)

| metric | value |
|--------|-------|
| **mean_reward** | **188.02** |
| **std_reward** | **66.06** |
| min | -55.34 |
| max | 251.02 |
| mean_ep_len | 509.3 |
| solved (≥200) | **❌ NO** |

## Model

- `runs/env_001/baselines/ppo_default_cpu/ppo_lunarlander_baseline.zip` (0.1 MB)
