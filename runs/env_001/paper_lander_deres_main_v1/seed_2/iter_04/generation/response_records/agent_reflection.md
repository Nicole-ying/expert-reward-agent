# Response Record

## 诊断

### 0. 环境理解
- 2D 飞行器从顶部出发，需着陆到画面中央的目标垫
- obs[0:2] 位置，obs[2:4] 速度，obs[4:5] 姿态，obs[6:7] 腿触地
- 动作：无推力 / 左姿态 / 主引擎 / 右姿态

### 1. Agent 发生了什么？
v3（-153.05，85% crash）比 v2（-120.87）更差。核心问题：
- **landing_bonus 可被盘旋 exploit**：v3 去掉 contact_factor 后，agent 在目标附近低速盘旋时每步获得约 +2.5 奖励（dist≈0.3, speed≈0.3），episode_sum_mean 达 119，max 极端值 8001。但 external score=-153，说明 agent 在"刷分"而非真正着陆。
- **progress_reward = -distance 是状态值而非动作效果**：盘旋时 distance 不变，-distance 恒定负值，但 landing_bonus 的正向奖励覆盖了它，总奖励仍为正 → agent 没有动力真正触地。
- v2 的 contact_factor 阻止了盘旋 exploit（触地前 bonus=0），但也导致 nonzero_rate 仅 4.38%，缺乏着陆梯度。

### 2. 哪个组件最值得干预？
**progress_reward**。它的数学形式（-distance，状态值）是盘旋 exploit 的共犯——在稳态盘旋时持续贡献恒定值，无法区分"靠近"和"停留"。改为 delta 形式后，盘旋时 delta≈0，切断了 exploit 的正反馈循环。

### 3. 我之前改了什么？
v2→v3 去掉了 contact_factor，得分从 -120.87 降到 -153.05。根据 revert 规则，回到 best（v2）代码，在此基础上做一个**新修改**：将 progress_reward 从 `-distance` 改为 `progress_delta`（距离减少量）。

这是层次 2 修改（改数学形式），只动一个组件。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v5: revert 到 v2 (best) 的 landing_bon
