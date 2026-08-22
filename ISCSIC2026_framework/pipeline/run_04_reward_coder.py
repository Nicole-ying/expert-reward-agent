import argparse, json, re
from pathlib import Path
from .common import load_config, read_text, read_json, write_text, record_prompt, record_response
from .common_context import write_json as write_json_file, write_text as write_text_file, markdown_block
from llm_clients.deepseek_client import DeepSeekClient

MOCK_REWARD_CODE="""
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    old_x = obs[0]
    old_y = obs[1]
    new_x = next_obs[0]
    new_y = next_obs[1]
    new_vx = next_obs[2]
    new_vy = next_obs[3]
    new_angle = next_obs[4]

    old_distance = (old_x * old_x + old_y * old_y) ** 0.5
    new_distance = (new_x * new_x + new_y * new_y) ** 0.5

    position_progress = old_distance - new_distance
    light_stability_cost = -0.05 * (abs(new_vx) + abs(new_vy)) - 0.05 * abs(new_angle)

    reward = position_progress + light_stability_cost

    if isinstance(info, dict):
        info["reward_terms"] = {
            "position_progress": position_progress,
            "light_stability_cost": light_stability_cost
        }

    return float(reward)
""".strip()

def strip_code_fences(text):
    t=text.strip()
    if t.startswith("```"):
        lines=t.splitlines();
        if lines and lines[0].startswith("```"): lines=lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines=lines[:-1]
        t="\n".join(lines)
    return t.strip()

def build_coder_input(cfg, run_dir):
    plan=read_json(str(run_dir/"reward_design_plan.json")); contract=read_json(str(run_dir/"environment_interface_contract.json")); coder_context=read_json(str(run_dir/"coder_environment_context.json")); implementation_constraints={"only_implement_core_terms":True,"do_not_use_raw_step_source":not cfg.get("context",{}).get("optional_debug_include_masked_step_for_coder",False),"do_not_use_original_reward":True,"do_not_invent_info_fields":True,"do_not_use_undeclared_obs_indices_or_slices":True,"allowed_obs_indices":list(contract.get("obs",{}).get("indices",{}).keys()),"allowed_info_fields":contract.get("info",{}).get("allowed_fields",[]),"forbidden_info_fields":contract.get("info",{}).get("forbidden_fields",[])+contract.get("info",{}).get("uncertain_fields",[])}; payload={"reward_design_plan":plan,"environment_interface_contract":contract,"coder_environment_context":coder_context,"implementation_constraints":implementation_constraints}
    if cfg.get("context",{}).get("optional_debug_include_masked_step_for_coder",False): payload["masked_step_source_for_debug_only"]=read_text(cfg["inputs"]["masked_step_path"])
    return payload

def validate_code(code, contract, plan):
    errors=[]
    if "def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):" not in code: errors.append("Missing exact compute_reward signature.")
    for token in ["import ","try:","except ","class ","lambda ","eval(","exec(","open("]:
        if token in code: errors.append(f"Forbidden token found: {token}")
    body=code.replace("def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):","")
    if "original_reward" in body: errors.append("original_reward is used inside function body.")
    obs_allowed=set(int(k) for k in contract.get("obs",{}).get("indices",{}).keys()); allowed_info=set(contract.get("info",{}).get("allowed_fields",[])); forbidden_info=set(contract.get("info",{}).get("forbidden_fields",[]))|set(contract.get("info",{}).get("uncertain_fields",[]))
    for match in re.finditer(r"\b(obs|next_obs)\s*\[\s*([^\]]+)\s*\]",code):
        obj,idx_txt=match.group(1),match.group(2)
        if ":" in idx_txt: errors.append(f"Slice not allowed: {obj}[{idx_txt}]")
        else:
            try:
                idx=int(idx_txt)
                if idx not in obs_allowed: errors.append(f"{obj}[{idx}] not declared in interface contract.")
            except Exception: errors.append(f"Non-literal index not allowed unless declared: {obj}[{idx_txt}]")
    for pat in [r"info\.get\(\s*['\"]([^'\"]+)['\"]",r"info\s*\[\s*['\"]([^'\"]+)['\"]\s*\]"]:
        for match in re.finditer(pat,code):
            field=match.group(1)
            if field=="reward_terms": continue
            if field not in allowed_info: errors.append(f"info field '{field}' is not explicitly allowed.")
            if field in forbidden_info: errors.append(f"info field '{field}' is forbidden or uncertain.")
    for sid in set(plan.get("formula_plan",{}).get("do_not_implement_yet",[])):
        if sid in code: errors.append(f"Deferred skeleton id appears in code: {sid}")
    try: compile(code,"<reward_v1.py>","exec")
    except Exception as e: errors.append(f"Compile failed: {e}")
    return {"valid":len(errors)==0,"errors":errors}

def run(config_path, run_name, mock=False):
    cfg=load_config(config_path); run_dir=Path(cfg["experiment"]["run_root"])/run_name; system_prompt=read_text(cfg["prompts"]["reward_coder"]); coder_input=build_coder_input(cfg,run_dir)
    write_json_file(run_dir/"llm_inputs/03_reward_coder.input.json",coder_input); write_text_file(run_dir/"human_review/03_reward_coder.input.md",markdown_block("reward_design_plan",coder_input["reward_design_plan"])+"\n"+markdown_block("environment_interface_contract",coder_input["environment_interface_contract"])+"\n"+markdown_block("coder_environment_context",coder_input["coder_environment_context"])+"\n"+markdown_block("implementation_constraints",coder_input["implementation_constraints"]))
    user_prompt=json.dumps(coder_input,ensure_ascii=False,indent=2); record_prompt(run_dir,"03_reward_coder",system_prompt,user_prompt)
    if mock: code=MOCK_REWARD_CODE
    else:
        llm_cfg=cfg["llm"]; client=DeepSeekClient(api_key_env=llm_cfg["api_key_env"],base_url=llm_cfg["base_url"]); raw=client.chat(model=llm_cfg["model_code"],system_prompt=system_prompt,user_prompt=user_prompt,temperature=llm_cfg["temperature_reward_coder"],max_tokens=llm_cfg["max_tokens_code"],json_mode=False); code=strip_code_fences(raw)
    validation=validate_code(code,coder_input["environment_interface_contract"],coder_input["reward_design_plan"])
    write_text(str(run_dir/"reward_v1.py"),code); write_text_file(run_dir/"raw_outputs/reward_v1.raw.py",code); write_text_file(run_dir/"final_outputs/reward_v1.py",code); write_json_file(run_dir/"validations/03_reward_coder.validation.json",validation); write_text_file(run_dir/"human_review/03_reward_coder.output.md","```python\n"+code+"\n```\n\n"+markdown_block("validation",validation))
    if not validation["valid"]:
        print("WARNING: reward code validation failed."); [print(" -",e) for e in validation["errors"]]
    record_response(run_dir,"03_reward_coder",code); print(run_dir/"reward_v1.py")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/env001_deepseek_rag.yaml"); ap.add_argument("--run-name",required=True); ap.add_argument("--mock",action="store_true"); args=ap.parse_args(); run(args.config,args.run_name,args.mock)
