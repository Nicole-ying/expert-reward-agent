# 设计理由

**修改了什么**：将 `contact_bonus`（稀疏二值、contact_threshold + leg 接触，active_rate=0%）**替换为连续化的 `landing_incentive`**。

**为什么**：僵尸组件 contact_bonus 从未触发（因为 agent 从未同时满足 `dist<0.3` 且至少一条腿接触）。agent 缺乏引导其靠近目标的强烈信号——progress_reward 每步仅 0.00072，极易被环境原始负奖励淹没。需要一个新的全局势场信号，使「离目标更近」的状态本身就有奖励。

**数学形式**：
```
next_dist = sqrt(next_x² + next_y²)
landing_incentive = w_landing / (1.0 + next_dist * 10.0)
```
- 使用 `1/(1+10d)` 而非指数衰减，因为在 dist>1 时指数项几乎为零（agent 到不了 reward 可感知的区域），而反比例函数在 dist=5 时仍有约 0.006/步，提供了全局的势场指引
- w_landing = 0.3：在原点（dist=0）时每步 0.3，dist=1 时约 0.027，dist=5 时约 0.006 ——在 episode 早期就提供可感知的正信号，在接近原点时显著增强

**系数校准**：
- progress_reward 的 per-step ≈ 0.71/985 ≈ 0.00072
- landing_incentive 在 dist=5 时 ≈ 0.006/步，约为主信号的 8x ——这看似超标，但**这个组件就是新的主信号框架**：它提供势场方向，progress_reward 提供即时反馈
- angle_penalty 的 per-step ≈ -0.13/985 ≈ -0.00013，远小于 0.3x 主信号
- 无新增惩罚，总惩罚负担不变

**未使用的观测**：x_velocity[2]、y_velocity[3]、angular_velocity[5] 仍未被使用，但本轮暂不加入——首要问题是缺乏目标接近信号。velocity 相关控制将在 landing_incentive 开始激活后再加入（比如在接近地面时控制下降速度）。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observation
    x = obs[0]
    y = obs[1]

    # Unpack next observation (state after action)
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_angle = next_obs[4]

    # ------------------  Main progress signal (improvement_delta)  ------------------
    # Reward distance reduction to the target pad (0,0)
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    w_progress = 1.0
    progress = (dist - next_dist)  # positive when moving toward the target

    # -----------  Landing incentive: continuous proximity bonus  -----------
    # Higher reward when closer to the landing pad (global potential field)
    w_landing = 0.3
    landing_incentive = w_landing / (1.0 + next_dist * 10.0)

    # -------------------  Health constraint: body angle -------------------
    # Penalize extreme tilt that could lead to a crash (hinge form)
    w_angle = 0.5
    safe_angle = 0.5          # radians
    angle_error = abs(next_angle) - safe_angle
    angle_penalty = -w_angle * angle_error if angle_error > 0 else 0.0

    # -------------------  Total reward  -------------------
    total_reward = w_progress * progress + landing_incentive + angle_penalty

    components = {
        "progress_reward": w_progress * progress,
        "landing_incentive": landing_incentive,
        "angle_penalty": angle_penalty
    }
    return float(total_reward), components
```

---

# 诊断摘要
- **audit**: contact_bonus 是僵尸组件（active_rate=0%），agent 从未到达奖励可感知区域；19/20 截断表明 agent 存活但无法完成任务。**信号缺失**—缺失全局势场引导。
- **behavior**: agent 在 ~1000 步中缓慢徘徊，净移动仅 0.71 单位；progress_reward 过于微弱，1/20 终止可能是速度失控坠毁。
- **signal**: 缺「靠近目标即为好状态」的势场信号；contact_bonus 的稀疏二值条件（dist<0.3 + 接触）从未满足，实际上等于没有着陆信号。
- **level**: Level 2 — 将稀疏二值 proxy 替换为连续化 bounded factor。
- **hypothesis**: 全局势场 `1/(1+10d)` 使 agent 在所有距离上都能感知方向——靠近原点直接获得更高奖励，不再只依赖微弱的 progress delta。这将引导 agent 更直接地飞向目标，减少徘徊，缩短 episode，从而也减少环境原始负奖励的累积。
- **risk**: 势场可能让 agent 学习在原点附近悬停（不真正着陆）以持续获取高奖励；若如此，下一轮需加入垂直速度控制或接触门控来消除该 exploit。