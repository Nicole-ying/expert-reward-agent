---

## 1. 诊断报告

### evidence

**跨 8 轮 contact_bonus 语义的生死循环（核心证据）：**

| iter | CB value | CB active | CB mag_share | score | term | 关键配置 |
|------|----------|-----------|-------------|-------|------|---------|
| 1 | 50 | 0% dead | 0% | -76.56 | 1/20 | 基线 |
| 2 | 50 | 5.2% | **98.2% 主导** | **63.72** | 17/20 | VP gate exp(-h/2), GP=2.0 |
| 3 | 50+speed gate | 8.2% | **97.7% 主导** | 21.45 | 9/20 | 尝试限制 CB，失败 |
| 4 | 8 | 0% dead | 0% | -10.11 | 0/20 | CB 太小，死亡 |
| 5 | 回归 iter2 配置 | 0% dead | 0% | -33.04 | 0/20 | 无法复现 iter2 |
| 6 | →landing_approach | 98.3% | 22.1% | -81.80 | 0/20 | 连续但仅位置感知，0 着陆 |
| 7 | →CB=20 | 0.7% | **86.1% 主导** | -5.37 | 2/20 | VP=0.3, gate=exp(-h/2) |
| 8 | CB=20, VP gate 收窄 | 0% dead | 0% | **-76.20** | 0/20 | VP=0.5, gate=exp(-h/1) |

**iter 8 checkpoint 证据（训练过程中 agent 曾发现着陆但丢失）：**
- contact_bonus cp10% = 75.195 (active 2.9%)，cp60% = 88.965 (active 1.9%)，但 cp100% = 0
- 得分曲线剧烈震荡：-92.3 → -177.2 → -133.5 → … → -71.4
- 这是典型的稀疏奖励不稳定：agent 偶尔撞到着陆状态获得巨奖，但连续信号无法提供可重复的梯度路径

**iter 7→8 的倒退机制：**
- iter 7：score=-5.37, 2/20 着陆, CB 86.1% mag_share（虽主导但至少活着）
- iter 8：VP gate 从 exp(-h/2.0) 收窄到 exp(-h/1.0)，权重 0.3→0.5
- 结果：在 h=2 处 penalty 从 -0.110v² 降到 -0.068v²（太弱），但在 h=0.5 处从 -0.234v² 升到 -0.303v²（太强且过渡太陡）
- 造成「高空自由→近地陡墙」的梯度断层，agent 在接近地面时遇到突然的惩罚墙，学会了回避近地区域

### behavior_diagnosis

策略当前的行为：**agent 在 1000 步内全程悬停/漂浮，从未尝试着陆**（0/20 terminated，全部 truncated）。持续信号每步净出血约 -0.0068（goal_proximity +0.0027, vel_penalty -0.0053, upright_penalty -0.0042），但总量太小（~ -6.8/episode），不足以驱动 agent 寻求终止。agent 学到的策略是「尽量少动以最小化惩罚」，而非「飞向目标垫着陆」。

### signal_completeness

对照环境卡片的 mandatory_roles：
- `goal_proximity` ✅ 存在，但仅提供方向引导，无着陆梯度
- `soft_landing_velocity_penalty` ✅ 存在，但 iter 8 的陡峭门控制造了梯度断层
- `upright_orientation` ✅ 存在且正常
- `contact_success` ⚠️ 存在但为稀疏二值 bonus——8 轮中有 5 轮完全死亡，3 轮 magnitude_share > 86% 主导

**缺失信号：连续着陆包络。** agent 需要一个在着陆状态附近连续上升的奖励场，让它能从「接近目标垫」平滑过渡到「着陆」。当前稀疏 contact_bonus 在着陆瞬间爆发巨奖，但在此之前毫无梯度；landing_approach（iter 6）考虑了位置但忽略了速度和姿态。

### selected_level

**L2** — 更换着陆信号的数学形态：从稀疏二值 bonus 变为连续乘法包络（multiplicative envelope）。这不是调参（L1），也不是换主框架（L3），而是将 contact_success 角色的实现从「`if contact: bonus`」变为「连续状态空间中的平滑峰值函数」。

### selected_intervention

**目标组件：** 将 `contact_bonus`（稀疏二值）替换为 `soft_landing_envelope`（连续乘法包络）。

**具体公式：**
```
soft_landing_envelope = 3.0 * exp(-h / 0.3) * exp(-5.0 * x²) * exp(-(vx² + vy²))
```

- `exp(-h/0.3)`：高度敏感——h=0.3 时衰减到 0.368，确保近地才有显著奖励
- `exp(-5x²)`：水平居中——继承 iter2/7/8 中验证有效的 centering gate
- `exp(-(vx²+vy²))`：低速奖励——速度越大奖励越低，自然引导减速
- 峰值 3.0/step：在理想着陆状态（h≈0, x≈0, v≈0）给出明确正信号，但远小于 iter2 的 contact_bonus=50（稀疏爆发），不会造成 magnitude_share 爆炸

**联动调整（同语义）：** velocity_penalty 门控从 exp(-h/1.0) 调整到 exp(-h/1.5)，权重保持 0.5。iter 8 的 exp(-h/1.0) 过渡太陡（在 h=2→0.5 之间 penalty 从 0.068v² 跳到 0.303v²，4.5 倍变化），exp(-h/1.5) 提供更平滑的过渡（2.6 倍变化），同时仍比 iter 7 的 exp(-h/2.0) 更集中近地。

### falsifiable_hypothesis

连续乘法包络在着陆状态处形成平滑的奖励峰，agent 在任何位置都能感知到「往着陆状态靠近会增加奖励」的梯度。这应该消除稀疏 bonus 的「要么爆发要么死亡」的振荡模式，使 agent 能够稳定地学会着陆。具体来说：
- `soft_landing_envelope` 的 active_rate 应该接近 100%（每步都有非零值）
- 其 magnitude_share 应该在中低范围（20-50%），不会像 contact_bonus 那样垄断
- checkpoint 得分曲线应该单调改善，不再剧烈震荡

### expected_next_round

- `soft_landing_envelope`：active_rate ≈ 100%，ep_sum 在 10-40 范围（取决于 agent 多快学会接近着陆状态），magnitude_share < 50%
- `goal_proximity`：ep_sum 应略有提升（agent 被 envelope 引导更积极接近目标）
- `velocity_penalty`：ep_sum 略微更负（agent 花更多时间在近地区域），但 magnitude_share 保持健康
- `upright_penalty`：变化不大
- score：应回到 iter 7 水平以上（> -5），且 terminated 应 > 2/20
- Checkpoint 趋势应更平滑，不再出现 -177 的剧烈低谷

### main_risk

**悬停奖励陷阱：** 如果 soft_landing_envelope 在 h≈0.5-1.0 区域给出中等奖励（~0.1-0.4/step），agent 可能学会在这个高度悬停而非真正着陆。因为悬停时速度接近零，envelope 仍有一定值，且不会触发 episode 终止。缓解：envelope 的 exp(-h/0.3) 衰减足够快——h=1.0 时高度因子仅 0.036，乘以速度和居中因子后奖励很小（< 0.05/step），不足以支撑悬停策略。同时 velocity_penalty + upright_penalty 持续施加负压，使悬停的净回报为负。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D rigid-body lander with two support legs.
    Designed for goal_approach_and_soft_contact: reach center pad,
    land softly with near-zero velocities, upright, and both legs in contact.

    Key change (iter 9): contact_bonus (sparse binary) replaced by
    soft_landing_envelope — a continuous multiplicative reward that peaks
    at the ideal landing state (h=0, x=0, v=0). Unlike the sparse bonus
    which oscillated between dead (0%) and dominating (98% magnitude share)
    across 7 of 8 iterations, the envelope provides gradient everywhere:
      - exp(-h/0.3): sharp height decay — negligible above h=1, ramps near ground
      - exp(-5x²): centering — same gate proven effective in iters 2/7/8
      - exp(-(vx²+vy²)): slows the agent as it approaches the pad

    Velocity penalty gate softened from exp(-h/1.0) to exp(-h/1.5):
    the iter 8 gate was too sharp, creating a "penalty cliff" near the
    surface that the agent learned to avoid entirely. exp(-h/1.5) provides
    a smoother transition while keeping the penalty concentrated near ground.

    The multiplicative envelope ensures all landing criteria must be
    simultaneously satisfied for high reward — partial satisfaction
    (e.g., centered but fast) yields proportionally less.
    """
    # --- goal proximity: linear delta-distance for uniform gradients ---
    prev_dist = (obs[0]**2 + obs[1]**2) ** 0.5
    next_dist = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    goal_proximity = 4.0 * (prev_dist - next_dist)

    # --- soft landing velocity penalty: height-gated ---
    # gate = 1.0 at ground, decays smoothly with height (k=1.5)
    # At h=1.5: gate=0.368, h=1.0: gate=0.513, h=0.5: gate=0.717, h=0: gate=1.0
    height = max(next_obs[1], 0.0)
    gate = 2.718281828 ** (-height / 1.5)
    vel_penalty = -0.5 * (next_obs[2]**2 + next_obs[3]**2) * gate

    # --- upright orientation: discourage tilt and angular velocity ---
    upright_penalty = -1.0 * (next_obs[4]**2 + next_obs[5]**2)

    # --- soft landing envelope: continuous reward peaking at ideal landing ---
    # Three multiplicative factors, each in (0, 1], product peaks at 1.0
    # at the perfect landing state (h=0, x=0, vx=vy=0).
    # Peak value = 3.0 per step at ideal state.
    # At h=0.5, x=0.1, v²=0.1: ~3.0 * 0.189 * 0.951 * 0.905 ≈ 0.49
    # At h=1.0, x=0.2, v²=0.3: ~3.0 * 0.036 * 0.819 * 0.741 ≈ 0.065
    # At h=2.0, x=0.5, v²=1.0: ~3.0 * 0.0013 * 0.287 * 0.368 ≈ 0.0004
    h = max(next_obs[1], 0.0)
    h_factor = 2.718281828 ** (-h / 0.3)
    x_factor = 2.718281828 ** (-5.0 * next_obs[0]**2)
    v_factor = 2.718281828 ** (-(next_obs[2]**2 + next_obs[3]**2))
    soft_landing_envelope = 3.0 * h_factor * x_factor * v_factor

    total_reward = goal_proximity + vel_penalty + upright_penalty + soft_landing_envelope

    components = {
        "goal_proximity": goal_proximity,
        "velocity_penalty": vel_penalty,
        "upright_penalty": upright_penalty,
        "soft_landing_envelope": soft_landing_envelope
    }
    return float(total_reward), components
```