import argparse
import json
from pathlib import Path
from .common import load_config, read_json, write_json
from rag.stage_retriever import retrieve_reflection_knowledge, save_reflection_knowledge


def make_mock_training_summary(run_dir):
    return {
        "observed_failure_modes": ["fast_crash_near_goal", "goal_near_oscillation"],
        "suspected_reward_hacking": ["progress reward can be obtained before unsafe crash"],
        "behavior_symptoms": ["agent approaches target but velocity remains high", "external reward does not improve as much as generated reward"],
        "reward_term_anomalies": ["position_progress may dominate total reward"],
        "candidate_next_actions": ["consider terminal_success_reward", "consider gated_reward", "consider stronger stability_penalty"],
        "low_level_metrics": {"episode_length_mean": 120, "termination_rate": 0.85, "external_eval_mean": -120.0},
        "free_text": "Use 04 failure modes and 05 reward hacking checks only."
    }


def run(config_path, run_name, training_summary_path=None, mock=False):
    cfg = load_config(config_path)
    run_dir = Path(cfg["experiment"]["run_root"]) / run_name
    rag_cfg = cfg["rag"]

    if training_summary_path:
        reflection_query = read_json(training_summary_path)
    elif mock:
        reflection_query = make_mock_training_summary(run_dir)
        write_json(str(run_dir / "mock_training_summary.json"), reflection_query)
    else:
        default_path = run_dir / "training_summary.json"
        if not default_path.exists():
            raise FileNotFoundError("Missing training_summary.json. Provide --training-summary or use --mock.")
        reflection_query = read_json(str(default_path))

    result = retrieve_reflection_knowledge(
        reflection_query=reflection_query,
        chunks_path=rag_cfg["reflection_chunks_path"],
        table_chunks_path=rag_cfg["reflection_table_chunks_path"],
        top_k=rag_cfg.get("reflection_top_k", 8),
        table_top_k=rag_cfg.get("reflection_table_top_k", 6),
    )

    save_reflection_knowledge(
        result,
        str(run_dir / "retrieved_reflection_knowledge.json"),
        str(run_dir / "retrieved_reflection_knowledge.md"),
    )
    print(run_dir / "retrieved_reflection_knowledge.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--training-summary", default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    run(args.config, args.run_name, args.training_summary, args.mock)
