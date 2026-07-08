# ⚠️ Restart Context

以下骨架在前序迭代中已尝试但未成功：
- angle_penalty + angular_vel_penalty + distance_reward + energy_penalty + time_penalty
- landing_bonus + progress_delta + stability_penalty

请选择一个**完全不同的主信号骨架**。例如如果上述列表都是 progress_delta 系列，请尝试 potential_based_shaping 或 bounded_proximity。不要重复已失败的骨架。
