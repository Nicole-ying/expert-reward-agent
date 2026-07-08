# Response Record

## 诊断

### 1. 这个 agent 发生了什么？

**核心症状：接触转移事件被反复刷分，但从未真正着陆。**

- 所有 20 个评估 episode 都是 truncated（1000 步截断），**0 个 terminated**——没有任何一次成功着陆（settled）。
- `contact_bonus` episode_sum_mean = 544.92，magnitude_share = 63.3%，active_rate = 11.7%。平均每 episode ~117 次接触转移事件，每次获得显著奖励。
- `landing_progress` sum = 4.45（仅 0.5% signed share），势能塑形信号在总量上几乎可以忽略。
- agent 学到了通过**反复弹跳**触发"未接触→双脚接触"转移来刷分，但从未在垫上稳定下来。这就是为什么所有 episode 都跑满 1000 步却不终止。

### 2. 哪个组件最值得干预？

**contact_bonus**。它是总奖励的主导成分（63.3%），但它的数学形态（转移事件 + phi 缩放）允许 agent 通过弹跳反复触发。上一轮（iter 8）试图加速率门控来修复，但得分从 91.87 降到 76.84——速度门控没有解决根本问题，反而破坏了有效的探索路径。

### 3. 我之前改了什么？

- iter 7（best, 91.87）：`persistent_to_transition_event`——将接触奖励从每步发放改为转移事件。这解决了驻留刷分，但引入了弹跳刷分。
- iter 8（current, 76.84）：在 iter 7 基础上加 `softness` 速度门控。**失败**——得分下降 15 分。按 revert 规则，应回到 iter 7 代码，并换一个方向修改。

---

## 修改方案

**诊断维度：`proxy_to_completion_alignment`**

接触转移事件是"着陆完成"的代理信号，但它与真正任务
