# Response Record

## 诊断

### 1. Agent 发生了什么？
Episode length 仅 71 步，100% 提前终止，score ≈ -111。**Agent 在 crash**——它学会了冲向原点 (0,0) 并坠毁，因为 `progress_delta` 对"靠近原点"给予正向奖励，完全不管靠近的方式。

### 2. 哪个组件是主要原因？
- `progress_delta`：均值 0.016，是唯一有效的正向信号。但它鼓励"冲向原点"而非"软着陆"——这是 crash 的根本原因。
- `soft_landing_proxy`：nonzero_rate 仅 0.38%，**几乎从未触发**。四个硬性二值条件（near AND slow AND upright AND both_contact）同时满足的概率极低，等于没有这个信号。
- `stability_penalty`：已从 iter1 的 ratio -3.56 降到 iter2 的
