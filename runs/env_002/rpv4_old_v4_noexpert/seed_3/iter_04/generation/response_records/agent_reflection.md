# Response Record

## 诊断报告

### evidence
- 历史得分：iter1=278.88 → iter2=285.19（c=1.5，历史最佳）→ iter3=281.28（c=2.0 回退）。目标 300，当前差距约 4.9%，属于「接近目标」区间。
- **iter3 关键新证据**：`score_range=[-51.748, 300.544]`，且 `early_terminal(<150 steps 且 score<-50)=0/20`。这说明存在「走得很好但**后期摔倒**」的回合（摔倒时官方任务给 -100 terminal penalty，把整轮拉到负）。同时存在接近满分 300.5 的成功回合。
- 反推分布：1 个摔倒回合（-51.7）拖低均值，其余回合均值约 298.8——即**非摔倒回合已经逼近 300**，缺口主要来自摔倒，而不是速度不足。
- 前向速度已够：iter2 均速 ≈ 756.2/(1.5×1110.5)=0.454 m/s，iter3 ≈ 967.0/(2.0×1160.15)=0.417 m/s。
- 当前 `balance_penalty` 只有 `1.0 * next_obs[0]**2` 的平滑弱惩罚，magnitude_share 仅 2.1%。而 info 为空，agent 在训练中**完全感知不到官方 -100 的摔倒惩罚**——所以它缺少「临近摔倒」的强信号。

### behavior_diagnosis
策略已能稳定向前行走（非摔倒回合 ~298.8 分），但偶尔在后期失去平衡摔倒（-51.7 分），把均值从 ~298 拖到 281。这个行为模式**不健康**：问题不是「不够快」，而是「缺少摔倒规避」。现有的弱平方 tilt 惩罚无法在倾斜增大时提供足够强的梯度，导致策略学不会在接近摔倒时及时纠正。

### signal_completeness
- forward_progress（前向速度）：✅ 已有且有效。
- balance/fall_prevention（防摔倒）：⚠️ 已有但**太弱**，缺一个「死区铰链式」的强摔倒风险信号。
- energy_efficiency（力矩/能耗）：缺失，但属次级目标，本轮不引入（避免多语义干预）。

### selected_level
**L2**（改 balance 组件的数学形态）。理由：
- L1 系数方向已穷尽（1.0/1.5/2.0 都试过），且回退 1.5 是已拒绝的重复干预。
- 新证据指向结构性缺口（无强摔倒信号），单纯调系数（如 tilt 系数 1.0→2.0）仍是平滑平方，无法在倾斜临界区产生陡峭梯度。
- 虽然 gap<10% 通常优先 L1，但 L1 已无有效空间，且本 L2 针对的是**新发现的摔倒瓶颈**，不是重调已接近最优的组件。

### selected_intervention
把 `tilt_penalty = 1.0 * next_obs[0]**2`（全量程平滑平方）替换为**死区铰链平方**：
`fall_risk_penalty = 40.0 * max(0.0, abs(next_obs[0]) - 0.4) ** 2`
- 正常行走躯干倾角约 0.1~0.15 rad（由 iter3 balance_penalty per-step ≈ 0.019 反推），阈值 0.4 rad 不会干扰正常步态。
- 一旦倾角超过 0.4 rad 进入「临近摔倒」区，惩罚急剧上升（0.7 rad→3.6/步，1.0 rad→14.4/步，1.3 rad→32.4/步），等价于用观测重构官方 -100 摔倒惩罚的梯度。
- 保留 `forward_reward = 1.5 * next_obs[2]`（已验证最优速度系数）与角速度/垂直速度弱正则。

### falsifiable_hypothesis
如果 281→300 的缺口主要由后期摔倒造成，那么死区铰链会迫使策略避开「倾斜>0.4 rad」的摔倒前状态 → 摔倒率下降 → score 最低分从 -51.7 升向正值 → 均值逼近 ~295-300。若下一轮均值仍停在 ~281 或下降，则说明摔倒并非瓶颈（或惩罚过强导致步态过保守）。

### expected_next_round
策略保持 ~0.45 m/s 的前进速度，但在倾斜临界区更早收紧躯干控制；`fall_risk_penalty` 组件在正常步态下大部分为 0（active_rate 显著 <100%），仅在摔倒回合的末段骤升；score 最低分抬升，均值向 300 靠拢。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v4 reward: 主信号仍为前向速度，但把"稳定性"从全局弱平方惩罚
    # 改为"摔倒风险"死区铰链惩罚。
    #
    # 证据：iter3 最终评估 score_range=[-51.748, 300.544] 且
    # early_terminal(<150 且 score<-50)=0/20，说明存在"走得很好但后期摔倒"
    # 的回合（官方摔倒给 -100 terminal penalty，把整轮拉到负），
    # 非摔倒回合均值已约 298.8。缺口在摔倒，不在速度。
    # 此前奖励只有 1.0*hull_angle^2 的弱平滑惩罚，且 info 为空，
    # agent 训练时感知不到官方 -100，缺少临近摔倒的强信号。
    #
    # 本轮 L2：把 tilt 惩罚从"全量程平方"改为"死区铰链平方"，
    # 仅在 |hull_angle| > 0.4 rad 后急剧增强，用观测重构摔倒惩罚梯度。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    # 保留已验证的最优速度系数 1.5（iter2 最佳 285.19）。
    forward_reward = 1.5 * next_obs[2]

    # 摔倒风险：死区铰链，只惩罚超过 0.4 rad 的躯干倾斜。
    # 正常步态倾角约 0.1~0.15 rad，不会被误伤；临近摔倒时惩罚急剧上升。
    tilt = abs(next_obs[0])
    fall_risk_penalty = 40.0 * max(0.0, tilt - 0.4) ** 2

    # 其余稳定正则：角速度与垂直速度（弱）。
    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    stability_penalty = angular_velocity_penalty + vertical_velocity_penalty

    total_reward = forward_reward - fall_risk_penalty - stability_penalty

    components = {
        "forward_reward": forward_reward,
        "fall_risk_penalty": -fall_risk_penalty,
        "stability_penalty": -stability_penalty,
    }

    return float(total_reward), components
```
