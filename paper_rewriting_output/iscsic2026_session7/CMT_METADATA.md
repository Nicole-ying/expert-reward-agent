# ISCSIC 2026 CMT metadata

**Target special session:** Session VII - Frontiers of Intelligent Systems: AI-Driven Control, Autonomy, and Emerging Challenges

**Title**  
CREATE: Training-Feedback-Guided Reward Evolution for Learning-Based Control

**Abstract**  
Large language models (LLMs) can synthesize executable reward functions for reinforcement learning, but an initially plausible reward often fails after a full policy-training run. Discarding that run wastes information about the induced behavior and the reward terms that produced it. We present CREATE, a persistent reward-design agent that treats each completed training run as data for the next reward-design decision. CREATE separates the generated reward used by proximal policy optimization from the environment-native objective used for evaluation, summarizes task performance and reward-component statistics, revises the active reward program, records the lineage of revisions, and retains the best reward-policy pair. This forms an autonomous supervisory loop for learning-based control rather than a sequence of unrelated one-shot generations. Across five independent seeds on LunarLander-v3, CREATE improves every initially sub-threshold reward beyond the 200-point criterion within ten reward evaluations. On BipedalWalker-v3, all five lineages cross 300, including a representative progression from 103.03 to 264.39 and then 304.92. An unconstrained multi-component revision ablation solves 0/5 LunarLander runs. The results show that unsuccessful LLM-generated rewards can be reused as revisable intermediate programs and improved through training-feedback-guided evolution.

**Keywords**  
reinforcement learning; reward design; large language models; autonomous agents; learning-based control
