import argparse
from .common import load_config, make_run_dir, read_text, write_json, record_prompt, record_response
from .common_context import write_json as write_json_file, write_text as write_text_file, markdown_block
from llm_clients.deepseek_client import DeepSeekClient


def build_interface_contract(card):
    obs = card.get("observation_space_analysis", {})
    act = card.get("action_space_analysis", {})
    rfi = card.get("reward_function_interface", {})
    termination = card.get("termination_analysis", {})
    step = card.get("step_function_analysis", {})
    indices = {}
    for item in obs.get("index_semantics", []):
        idx = str(item.get("index"))
        indices[idx] = {"name": item.get("name", ""), "meaning": item.get("meaning", ""), "reward_usable": item.get("reward_usable", False), "range": item.get("range", ""), "notes": item.get("notes", "")}
    actions = {}
    for item in act.get("action_meanings", []):
        actions[str(item.get("value"))] = {"name": item.get("name", ""), "meaning": item.get("meaning", "")}
    return {
        "env_id": card.get("env_id", "Env_001"),
        "obs": {"type": obs.get("type", ""), "shape": obs.get("shape", []), "dtype": obs.get("dtype", ""), "indices": indices, "allowed_slices": []},
        "action": {"type": act.get("type", ""), "n": act.get("n", None), "values": actions},
        "info": {"allowed_fields": rfi.get("allowed_info_fields", []), "uncertain_fields": rfi.get("uncertain_info_fields", []), "forbidden_fields": rfi.get("forbidden_info_fields", [])},
        "termination": {"termination_conditions": step.get("termination_conditions", []), "explicit_success_flag_available": termination.get("explicit_success_flag_available", False), "explicit_failure_flag_available": termination.get("explicit_failure_flag_available", False), "success_signal": termination.get("success_signal", {}), "failure_signal": termination.get("failure_signal", {})},
        "reward_function": {"signature": rfi.get("signature", "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):"), "allowed_inputs": rfi.get("allowed_inputs", []), "forbidden_inputs": rfi.get("forbidden_inputs", ["original_reward"])}
    }


def build_coder_environment_context(card, contract):
    return {"env_id": card.get("env_id", "Env_001"), "task_summary": card.get("task_summary", {}), "obs_index_semantics": contract.get("obs", {}).get("indices", {}), "action_meanings": contract.get("action", {}).get("values", {}), "termination_signal_map": contract.get("termination", {}), "allowed_info_fields": contract.get("info", {}).get("allowed_fields", []), "uncertain_info_fields": contract.get("info", {}).get("uncertain_fields", []), "forbidden_info_fields": contract.get("info", {}).get("forbidden_fields", [])}


def mock_environment_card():
    return {
        "env_id": "Env_001",
        "task_summary": {"one_sentence": "A 2D vehicle-like agent should reach and settle near a central target pad.", "main_objective": ["reach target region", "settle safely"], "secondary_objective": ["reduce engine use", "finish efficiently"], "safety_constraints": ["avoid crash", "avoid leaving bounds", "avoid unstable contact"]},
        "task_type_selection": {"selected_route_id": "navigation_goal_reaching", "selected_route_reason": "The primary objective is reaching a target region; landing, stability, and fuel are additional constraints.", "rejected_route_ids": [{"route_id": "survival_balance_task", "reason": "Task is not merely staying alive or balancing."}, {"route_id": "locomotion_continuous_control", "reason": "No walking/running forward locomotion objective."}, {"route_id": "manipulation_grasping", "reason": "No object manipulation or grasping."}, {"route_id": "autonomous_driving_safety", "reason": "No road, lane, traffic, or driving rules."}, {"route_id": "sparse_exploration", "reason": "Dense position/velocity/orientation/contact signals exist."}, {"route_id": "multi_objective_task", "reason": "Secondary objectives exist but are not the primary task type."}]},
        "observation_space_analysis": {"type": "Box", "shape": [8], "dtype": "float32", "range": "normalized continuous values; exact bounds from task spec if available", "index_semantics": [{"index": 0, "name": "x_position", "meaning": "horizontal position relative to target", "range": "approx normalized", "reward_usable": True, "notes": ""}, {"index": 1, "name": "y_position", "meaning": "vertical position relative to pad height", "range": "approx normalized", "reward_usable": True, "notes": ""}, {"index": 2, "name": "x_velocity", "meaning": "horizontal velocity", "range": "continuous", "reward_usable": True, "notes": ""}, {"index": 3, "name": "y_velocity", "meaning": "vertical velocity", "range": "continuous", "reward_usable": True, "notes": ""}, {"index": 4, "name": "body_angle", "meaning": "body orientation angle", "range": "continuous", "reward_usable": True, "notes": ""}, {"index": 5, "name": "angular_velocity", "meaning": "angular velocity", "range": "continuous", "reward_usable": True, "notes": ""}, {"index": 6, "name": "left_contact", "meaning": "left contact flag", "range": "0 or 1", "reward_usable": True, "notes": ""}, {"index": 7, "name": "right_contact", "meaning": "right contact flag", "range": "0 or 1", "reward_usable": True, "notes": ""}], "uncertain_items": []},
        "action_space_analysis": {"type": "Discrete", "n": 4, "shape": [], "action_meanings": [{"value": 0, "name": "no_engine", "meaning": "do nothing"}, {"value": 1, "name": "left_orientation_engine", "meaning": "fire left orientation engine"}, {"value": 2, "name": "main_engine", "meaning": "fire main engine"}, {"value": 3, "name": "right_orientation_engine", "meaning": "fire right orientation engine"}], "reward_usable_action_signals": {"engine_use_available_from_action": True, "action_smoothness_available": False, "notes": "engine usage can be derived from action, but action cost is a later objective"}},
        "step_function_analysis": {"state_transition_summary": "The step updates a physical 2D body based on discrete engine actions and returns next observation, reward, done/termination, and info.", "termination_conditions": [{"condition_id": "crash_or_body_contact", "semantic_type": "failure", "source_in_step": "body/crash contact condition", "observable_from_obs": False, "observable_from_info": False, "available_to_compute_reward": False, "notes": "not usable unless wrapper exposes reason"}, {"condition_id": "out_of_bounds", "semantic_type": "failure", "source_in_step": "horizontal position leaves viewport", "observable_from_obs": True, "observable_from_info": False, "available_to_compute_reward": "derived_possible", "notes": "requires threshold; not used in v1 core"}, {"condition_id": "settled_or_not_awake", "semantic_type": "ambiguous", "source_in_step": "body not awake / settled", "observable_from_obs": False, "observable_from_info": False, "available_to_compute_reward": False, "notes": "could be success or neutral; do not assume success"}], "info_fields": [], "official_reward_masked": True, "forbidden_reward_sources": ["original_reward", "official_reward"]},
        "termination_analysis": {"explicit_success_flag_available": False, "explicit_failure_flag_available": False, "success_signal": {"available_info_field": None, "derived_success_possible": True, "derived_success_signals": ["x_position", "y_position", "x_velocity", "y_velocity", "body_angle", "left_contact", "right_contact"], "use_as_v1_core": False, "reason": "No explicit success flag is available to compute_reward; derived condition is heuristic and should not be v1 core."}, "failure_signal": {"available_info_field": None, "derived_failure_possible": True, "derived_failure_signals": ["x_position threshold if known", "velocity/angle/contact heuristics"], "use_as_v1_core": False, "reason": "No explicit failure flag or termination reason is available to compute_reward."}},
        "reward_function_interface": {"signature": "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):", "allowed_inputs": ["obs", "action", "next_obs", "info", "training_progress"], "forbidden_inputs": ["original_reward"], "allowed_info_fields": [], "uncertain_info_fields": ["success", "failure", "termination_reason"], "forbidden_info_fields": ["official_reward", "fitness_score", "individual_reward", "original_reward"], "parameter_meanings": {"obs": "state before action", "action": "action taken at obs", "next_obs": "state after action", "original_reward": "environment reward, forbidden", "info": "metadata dictionary; only explicitly allowed fields may be used", "training_progress": "optional curriculum scalar, disabled unless plan explicitly uses it"}},
        "primary_environment_signals": ["position_to_target", "velocity", "orientation", "contact_flags"],
        "secondary_environment_signals": ["engine_usage_from_action", "episode_time"],
        "uncertain_or_unavailable_signals": ["explicit_success_flag", "explicit_failure_flag", "termination_reason"]
    }


def run(config_path, run_name=None, mock=False):
    cfg = load_config(config_path)
    run_dir = make_run_dir(cfg, run_name)
    system_prompt = read_text(cfg["prompts"]["environment_analyzer"])
    task_spec = read_text(cfg["inputs"]["task_spec_path"])
    masked_step = read_text(cfg["inputs"]["masked_step_path"])
    user_prompt = f"ANONYMIZED_TASK_SPEC:\n{task_spec}\n\nMASKED_STEP_SOURCE:\n{masked_step}"
    record_prompt(run_dir, "01_environment_analyzer", system_prompt, user_prompt)
    if mock:
        card = mock_environment_card()
    else:
        llm_cfg = cfg["llm"]
        client = DeepSeekClient(api_key_env=llm_cfg["api_key_env"], base_url=llm_cfg["base_url"])
        card = client.chat_json(model=llm_cfg["model_json"], system_prompt=system_prompt, user_prompt=user_prompt, temperature=llm_cfg["temperature_environment_analyzer"], max_tokens=llm_cfg["max_tokens_json"])
    contract = build_interface_contract(card)
    coder_context = build_coder_environment_context(card, contract)
    write_json(str(run_dir / "environment_card.json"), card)
    write_json(str(run_dir / "environment_interface_contract.json"), contract)
    write_json(str(run_dir / "coder_environment_context.json"), coder_context)
    write_json_file(run_dir / "raw_outputs/environment_card.raw.json", card)
    write_json_file(run_dir / "raw_outputs/environment_interface_contract.raw.json", contract)
    write_json_file(run_dir / "final_outputs/environment_card.json", card)
    write_json_file(run_dir / "final_outputs/environment_interface_contract.json", contract)
    write_text_file(run_dir / "human_review/01_environment_analyzer.output.md", markdown_block("environment_card", card) + "\n" + markdown_block("environment_interface_contract", contract))
    record_response(run_dir, "01_environment_analyzer", {"environment_card": card, "environment_interface_contract": contract})
    print(run_dir / "environment_card.json")
    print(run_dir / "environment_interface_contract.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    run(args.config, args.run_name, args.mock)
