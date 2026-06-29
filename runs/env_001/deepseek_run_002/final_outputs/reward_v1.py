import numpy as np

def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for Env_001: 2D lunar lander trajectory optimization.
    
    Args:
        obs: current observation [x_pos, y_pos, x_vel, y_vel, body_angle, angular_vel, left_contact, right_contact]
        action: discrete action (0: no_engine, 1: left_orientation, 2: main_engine, 3: right_orientation)
        next_obs: next observation after taking action
        original_reward: original environment reward (not used)
        info: info dict (empty for this environment)
        training_progress: training progress from 0.0 to 1.0 (not used in v1)
    
    Returns:
        total_reward: scalar reward value
        info: updated info dict with reward_terms
    """
    # Extract relevant signals from observations
    # Position relative to target (target is at origin)
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    # Next state signals
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # ========== 1. Progress-based distance reward (main learning signal) ==========
    # Calculate Euclidean distance to target (origin)
    current_dist = np.sqrt(x_pos**2 + y_pos**2)
    next_dist = np.sqrt(next_x_pos**2 + next_y_pos**2)
    
    # Progress reward: positive when moving closer to target
    progress_reward = current_dist - next_dist
    
    # ========== 2. Lightweight stability penalty ==========
    # Penalize high velocity and large angular deviation
    # These are important for stable landing approach
    speed = np.sqrt(next_x_vel**2 + next_y_vel**2)
    angle_penalty = abs(next_body_angle)  # Penalize deviation from upright (angle=0)
    angular_vel_penalty = abs(next_angular_vel)  # Penalize rotation
    
    # Combine stability penalties with small weights
    # Weight for velocity: 0.01 (gentle penalty for high speed)
    # Weight for angle: 0.02 (moderate penalty for tilting)
    # Weight for angular velocity: 0.005 (light penalty for spinning)
    stability_penalty = -0.01 * speed - 0.02 * angle_penalty - 0.005 * angular_vel_penalty
    
    # ========== 3. Contact bonus (encouraging landing) ==========
    # Reward when both supports are in contact (landed)
    contact_bonus = 0.0
    if next_left_contact > 0.5 and next_right_contact > 0.5:
        contact_bonus = 1.0  # Small bonus for being on the ground
    
    # ========== Combine rewards ==========
    total_reward = progress_reward + stability_penalty + contact_bonus
    
    # ========== Store reward terms for diagnostics ==========
    info["reward_terms"] = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "contact_bonus": contact_bonus,
        "total_reward": total_reward
    }
    
    return total_reward, info