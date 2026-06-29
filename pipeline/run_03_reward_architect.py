import argparse, json
from pathlib import Path
from .common import load_config, read_text, read_json, write_json, record_prompt, record_response
from .common_context import write_json as write_json_file, write_text as write_text_file, markdown_block
from llm_clients.deepseek_client import DeepSeekClient


def mock_reward_design_plan():
    return {"reward_version":"reward_v1","iteration_stage":"stage_1_minimal","selected_route_id":"navigation_goal_reaching","ideal_full_reward_skeleton":{"formula":"r = task_objective + learning_guidance - safety_cost - action_cost + optional_exploration_or_curriculum","components":["terminal_success_reward","progress_delta_reward","stability_penalty","terminal_failure_penalty","gated_reward","energy_penalty","time_penalty"],"note":"This is the mature target skeleton, not all implemented in v1."},"skeleton_decision_table":[{"skeleton_id":"terminal_success_reward","decision":"defer","role":"task_objective","required_signals":["success_flag"],"interface_status":"missing","reason":"No explicit success flag is available in info or contract.","risk":"sparse_reward_no_learning"},{"skeleton_id":"progress_delta_reward","decision":"select_v1","role":"learning_guidance","required_signals":["obs[0]","obs[1]","next_obs[0]","next_obs[1]"],"interface_status":"available","reason":"Provides dense step-wise signal using explicit position indices.","risk":"goal_near_oscillation"},{"skeleton_id":"stability_penalty","decision":"select_v1","role":"safety_cost","required_signals":["next_obs[2]","next_obs[3]","next_obs[4]"],"interface_status":"available","reason":"Light constraint reduces high-speed/unstable approach.","risk":"over_conservative_policy"},{"skeleton_id":"energy_penalty","decision":"defer","role":"action_cost","required_signals":["action"],"interface_status":"available","reason":"Secondary efficiency objective can cause no-op if added too early.","risk":"agent_afraid_to_move"}],"reward_v1_minimal_design":{"design_type":"two_term_combination","selected_skeletons":["progress_delta_reward","stability_penalty"],"reason":"Use one dense learning guidance term plus one light safety term.","excluded_due_to_interface":["terminal_success_reward","terminal_failure_penalty"],"excluded_due_to_minimal_first":["energy_penalty","time_penalty","gated_reward"]},"formula_plan":{"core_terms":[{"term_name":"position_progress","skeleton_id":"progress_delta_reward","variables":[{"symbol":"old_x","source":"obs[0]","meaning":"previous horizontal position relative to target"},{"symbol":"old_y","source":"obs[1]","meaning":"previous vertical position relative to target/pad"},{"symbol":"new_x","source":"next_obs[0]","meaning":"next horizontal position relative to target"},{"symbol":"new_y","source":"next_obs[1]","meaning":"next vertical position relative to target/pad"}],"design":"distance_old - distance_new","initial_weight_guidance":"main term, weight 1.0"},{"term_name":"light_stability_cost","skeleton_id":"stability_penalty","variables":[{"symbol":"new_vx","source":"next_obs[2]","meaning":"horizontal velocity"},{"symbol":"new_vy","source":"next_obs[3]","meaning":"vertical velocity"},{"symbol":"new_angle","source":"next_obs[4]","meaning":"body angle"}],"design":"-0.05*(abs(vx)+abs(vy))-0.05*abs(angle)","initial_weight_guidance":"small auxiliary cost"}],"do_not_implement_yet":["terminal_success_reward","terminal_failure_penalty","gated_reward","energy_penalty","time_penalty","dynamic_curriculum_reward"]},"coder_constraints":{"allowed_obs_indices":[0,1,2,3,4,5,6,7],"allowed_next_obs_indices":[0,1,2,3,4,5,6,7],"allowed_action_values":[0,1,2,3],"allowed_info_fields":[],"forbidden_info_fields":["success","failure","termination_reason","official_reward","fitness_score","individual_reward","original_reward"],"forbidden_patterns":["info.get('success')","obs[0:3]","next_obs[0:3]"]},"later_iteration_plan":[{"iteration_stage":"v2","condition":"agent approaches but does not settle","diagnosis_needed":"goal_near_oscillation","add_or_modify":"consider derived terminal success or gated contact after verifying signal availability","related_skeleton":"terminal_success_reward/gated_reward","reason":"Need final settling signal after approach works."},{"iteration_stage":"v3","condition":"agent lands but uses excessive engine","diagnosis_needed":"excessive_action_cost","add_or_modify":"add small energy_penalty","related_skeleton":"energy_penalty","reason":"Efficiency is secondary."}],"training_observation_plan":{"reward_terms_to_log":["position_progress","light_stability_cost"],"behavior_metrics_to_check":["distance_to_target","velocity_near_target","episode_length","external_eval_mean"],"failure_indicators":["goal_near_oscillation","fast_crash_near_goal","hover_without_landing"]}}

def validate_plan(plan, contract):
    errors=[]; obs_indices=set(int(k) for k in contract.get("obs",{}).get("indices",{}).keys()); allowed_info=set(contract.get("info",{}).get("allowed_fields",[])); forbidden_info=set(contract.get("info",{}).get("forbidden_fields",[]))|set(contract.get("info",{}).get("uncertain_fields",[]))
    for term in plan.get("formula_plan",{}).get("core_terms",[]):
        for var in term.get("variables",[]):
            src=var.get("source","")
            if src.startswith("obs[") or src.startswith("next_obs["):
                if ":" in src: errors.append(f"Slice not allowed in source: {src}")
                else:
                    try:
                        idx=int(src.split("[")[1].split("]")[0])
                        if idx not in obs_indices: errors.append(f"Obs index {idx} not in contract.")
                    except Exception: errors.append(f"Cannot parse obs source: {src}")
            if src.startswith("info."):
                field=src.split(".",1)[1]
                if field not in allowed_info: errors.append(f"Info field {field} is not explicitly allowed.")
                if field in forbidden_info: errors.append(f"Info field {field} is forbidden or uncertain.")
    for row in plan.get("skeleton_decision_table",[]):
        if row.get("decision")=="select_v1" and row.get("interface_status")!="available": errors.append(f"Selected skeleton {row.get('skeleton_id')} has interface_status={row.get('interface_status')}.")
    return {"valid":len(errors)==0,"errors":errors}

def run(config_path, run_name, mock=False):
    cfg=load_config(config_path); run_dir=Path(cfg["experiment"]["run_root"])/run_name
    system_prompt=read_text(cfg["prompts"]["reward_architect"]); architect_input=read_json(str(run_dir/"llm_inputs/02_reward_architect.input.json")); user_prompt=json.dumps(architect_input,ensure_ascii=False,indent=2); record_prompt(run_dir,"02_reward_architect",system_prompt,user_prompt)
    if mock: plan=mock_reward_design_plan()
    else:
        llm_cfg=cfg["llm"]; client=DeepSeekClient(api_key_env=llm_cfg["api_key_env"],base_url=llm_cfg["base_url"]); plan=client.chat_json(model=llm_cfg["model_json"],system_prompt=system_prompt,user_prompt=user_prompt,temperature=llm_cfg["temperature_reward_architect"],max_tokens=llm_cfg["max_tokens_json"])
    contract=architect_input["environment_interface_contract"]; validation=validate_plan(plan,contract)
    write_json(str(run_dir/"reward_design_plan.json"),plan); write_json_file(run_dir/"raw_outputs/reward_design_plan.raw.json",plan); write_json_file(run_dir/"final_outputs/reward_design_plan_v1.json",plan); write_json_file(run_dir/"validations/02_reward_architect.validation.json",validation); write_text_file(run_dir/"human_review/02_reward_architect.output.md",markdown_block("reward_design_plan_v1",plan)+"\n"+markdown_block("validation",validation))
    if not validation["valid"]:
        print("WARNING: reward_design_plan validation failed."); [print(" -",e) for e in validation["errors"]]
    record_response(run_dir,"02_reward_architect",plan); print(run_dir/"reward_design_plan.json")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/env001_deepseek_rag.yaml"); ap.add_argument("--run-name",required=True); ap.add_argument("--mock",action="store_true"); args=ap.parse_args(); run(args.config,args.run_name,args.mock)
