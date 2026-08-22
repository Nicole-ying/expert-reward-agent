# Subagent Research Signal

**训练过程**: Episode length grew 153→218, crash rate fell 86%→74%, generated reward rose 0.677→3.076, original env reward improved only marginally -92.5→-85.7; agent never solved the task.

**组件健康**: All components active: soft_contact_reward nonzero 35.6% (mean when active 9.64), progress_reward and orientation_penalty always active (means -0.22, -0.04). No dead components.

**奖励对齐**: Shaped reward (late 3.076) severely mismatched with original env reward (late -85.7); agent exploits progress/contact rewards without learning to land, evidenced by 74% crash rate and mediocre eval scores.

**异常检测**: Not reported.

**置信度**: `high`
