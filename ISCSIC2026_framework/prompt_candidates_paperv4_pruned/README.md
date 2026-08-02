# paper-v4-pruned prompt candidate

This candidate starts from the frozen paper-v4 environment analyzer and initial reward generator, then removes the parts that were not needed for the paper framework:

- closed task-route catalogues and dynamics subtypes;
- expert task profiles and morphology descriptions;
- fixed reward templates and mandatory formula-library routing;
- repeated rewriting of the original task description.

It keeps the paper-v4 mechanism that produced repairable initial rewards:

- dynamic reward-role decomposition;
- role-to-signal mapping;
- mandatory, conditional and excluded responsibilities;
- one independently diagnosable component per selected role;
- a 2–4 component v1 budget based on high-level behavioral roles rather than one component per observation group;
- post-training failure evidence for later reflection.

The vNext A/B studies additionally contributed runtime-contract propagation, component contribution semantics, terminal uncertainty, reward clipping, potential telescoping bounds, per-step cost accumulation and explicit reward-hacking audits.

The frozen files under `prompts/` and the simplified candidates under `prompt_candidates_vnext/` remain unchanged for provenance and comparison.
