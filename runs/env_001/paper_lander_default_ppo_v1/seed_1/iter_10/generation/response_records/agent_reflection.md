# Response Record

1. `evidence`：当前代码（iter 9）score=-108.87，所有20条评估轨迹均在68步内提前终止；`landing_improvement`（state-to-improvement）均值11.71但无法挽救早期崩溃，`speed_tracking_reward`均值-42.51占负向支配；历史最佳（iter 5）使用`soft_landing_proxy`（state value）获162.94分、ep_len=463.7且speed仅-0.374，形成鲜明对照。
2. `behavior_diagnosis`：agent从未学会靠近着陆垫，68步内即坠毁；`landing_improvement`在远离垫面时质量差分为零，整条轨迹仅靠微弱的`progress_reward`和支配性的负`speed_tracking_reward`，信用分配崩溃。
3. `signal_completeness`：缺少可从中远距离获得的连续正向引导信号。`progress_reward`步均量级过小，`speed_tracking_reward`在无引导时退化为纯惩罚；着陆质量信号必须从近垫区扩展到更早阶段才可达。
4. `selected_level`：Level 2 — `landing_improvement`是state_to_improvement变换，iter 5（state value）与iter 6/9（state_to_improvement）的直接对照证明该数学形态在此任务中失败，必须替换回state value结构。
5. `selected_intervention`：唯一目标组件`landing_improvement`→替换为`soft_landing_proxy`（state value形态），同步将`proximity_threshold`从0.5扩大到1.5，使正向引导从中远距离即可激活；`progress_reward`和`speed_tracking_reward`保持与best完全一致。
6. `falsifiable_hypothesis`：state value形态的soft_landing_proxy配合更宽的邻近阈值，会在agent尚在远处时即提供可感知的正向梯度（proximity_score从dist≈1.5开始非零），使speed_tracking_reward不再孤立为支配性惩罚，agent应能复现并超越best的受控接近行为。
7. `expected_next_round`：episode_length应回升至300+，score由负转正并接近或超过162.94，`speed_tracking_reward`的episode_sum_mean绝对值应大幅下降（从-42→个位数），`soft_landing_proxy`应持续激活且magnitude_share显著上升。
8. `main_risk`：扩大proximity_threshold可能让agent在阈值边界附近徘徊获取中等proxy而非果断着陆；若下一轮出现中等分平台（ep_len很长但score停滞在best以下），需追加completion对齐信号。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    观测:
    obs[0]: x_position_relative_to_target
    obs[1]: y_position_relative_to_pad_height
    obs[2]: x_velocity
    obs[3]: y_velocity
    obs[4]: body_angle
    obs[5]: angular_velocity
    obs[6]: left_contact_flag (0.0/1.0)
    obs[7]: right_contact_flag (0.0/1.0)
    """

    def distance(obs_arr):
        return (obs_arr[0] ** 2 + obs_arr[1] ** 2 + 1e-8) ** 0.5

    # ---- 1. Progress reward: 距离减少量 ----
    dist_old = distance(obs)
    dist_new = distance(next_obs)
    progress_reward = dist_old - dist_new

    # ---- 2. Speed tracking reward: 期望速度引导 ----
    max_speed = 5.0
    d_ref = 1.0
    desired_speed = max_speed * min(dist_new / d_ref, 1.0)

    cur_speed = (next_obs[2] ** 2 + next_obs[3] ** 2 + 1e-8) ** 0.5
    speed_error = abs(cur_speed - desired_speed)
    lambda_speed = 0.2
    speed_tracking_reward = -lambda_speed * speed_error

    # ---- 3. Soft landing proxy: state value, 扩大邻近阈值以提供早期引导 ----
    proximity_threshold = 1.5   # 从best的0.5扩大，使中远距离即获得正向信号
    speed_threshold = 0.25
    angle_threshold = 0.2

    proximity_score = max(0.0, 1.0 - dist_new / proximity_threshold)
    speed_low_score = max(0.0, 1.0 - cur_speed / speed_threshold)
    angle_score = max(0.0, 1.0 - abs(next_obs[4]) / angle_threshold)
    contact_score = (next_obs[6] + next_obs[7]) * 0.5

    soft_landing_proxy = (
        proximity_score +
        speed_low_score +
        angle_score +
        2.0 * contact_score
    )

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + soft_landing_proxy

    reward_components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'soft_landing_proxy': soft_landing_proxy
    }
    return float(total_reward), reward_components
```
