# 环境理解与初始奖励 Prompt 诊断

## 总体判断

现有 Prompt 的主要问题不是约束不足，而是**决策主线被分类学和模板信息淹没**。模型需要填写 task family、dynamics subtype、expert task profile、mandatory/conditional/avoid roles、role-to-signal mapping、formula operator 等多层中间产物，容易把注意力花在“把表填全”，而不是回答奖励设计真正依赖的语义问题。

## 现有 Environment Analyzer 的问题

1. **输出过重。** 十二节环境卡同时承担接口审计、任务分类、专家画像、奖励职责拆分和失败预测，信息重复。
2. **先分类、后理解。** 强制从七类路线和多个 dynamics subtype 中选标签，会诱导模型用标签反推奖励，而不是从 success semantics 和 termination logic 推出安全的进展信号。
3. **成功与失败没有成为主轴。** Terminated、truncated 和 failure modes 虽被提及，却只是大表中的一部分，没有成为奖励骨架选择的首要证据。
4. **专家画像难以形成可验证输入。** `expert_task_profile` 等字段包含大量抽象描述，但其中许多不会直接改变 reward code，反而增加 Prompt 长度。
5. **可用信号审计是有效部分。** observation index、action semantics、允许的 `info` 字段以及禁止信号必须保留，否则生成器会发明变量。

## 现有 Initial Reward Generator 的问题

1. **设计流程太长。** `role → signal → formula operator → code` 本身合理，但又叠加 skeleton、operator library、component budget、task family 和大量反例，容易产生机械拼装。
2. **组件职责没有围绕成功语义组织。** 初始奖励最先要保证“优化方向与任务成功方向一致”，再补必要安全/稳定约束；现有 Prompt 对 operator 和 role taxonomy 的强调更强。
3. **正负信号尺度关系不够突出。** 如果惩罚总量长期压过正向进展，策略可能学会不行动、拖延或寻找 proxy 漏洞。主目标信号应占主导，惩罚应有条件、有限幅并只约束明确坏行为。
4. **终止奖励需要更精确。** 应奖励可合法识别的成功、惩罚可合法识别的失败；若接口不能区分，就不能仅凭 `terminated=True` 猜测，也不能发明 success flag。
5. **代码约束应保留。** 固定函数签名、禁止 `original_reward`、只用声明信号、第一段代码可执行、返回具名 components，这些约束直接服务于安全执行和后续诊断。

## 新的核心决策链

```text
任务成功语义
  → 终止/截断语义
  → 合法可用信号
  → 安全的主进展骨架
  → 必要的安全/稳定约束
  → 2–4 个可诊断组件
  → 代码验证与训练
```

骨架选择只保留最小必要规则：

- 有明确目标状态或目标位置：优先 potential difference / distance delta，使每步优化方向与接近目标一致。
- 有持续前进方向：优先 displacement delta 或 directional velocity。
- 核心是存活、站稳或保持平衡：优先 survival/health 信号，并辅以必要稳定项。
- 成功事件能从合法信号可靠识别：加入明确成功 bonus。
- 失败事件能从合法信号可靠识别：加入有界失败 penalty。
- `terminated` 本身不等于成功，也不等于失败；必须根据 step source 分解原因。
- `truncated` 通常表示时间上限等外部截断，不应自动当作成功或失败；以源码事实为准。

## 推荐的奖励职责

概念形式：

\[
R_{\text{total}}
=w_gR_{\text{goal}}
+w_sR_{\text{safety}}
+w_bR_{\text{stability}}
-w_pP_{\text{bad}}.
\]

不是每项都必须出现。初始版本优先 2–4 个具名组件：

1. `goal/progress`：主要正向引导，决定策略向哪里优化。
2. `success`：仅在成功可可靠识别时加入，可与 goal 合并。
3. `safety/stability`：只保留完成任务所必需的约束，尺度低于主进展信号。
4. `failure/bad_behavior_penalty`：仅对明确失败或危险行为触发，最好有界、稀疏或门控。

“2–4 个”应是强默认而非绝对数学定律。若两个职责能用一个组件表达就合并；只有缺少某个组件会使成功不可达或产生明显 reward hacking 时才增加第五项，并必须解释原因。
