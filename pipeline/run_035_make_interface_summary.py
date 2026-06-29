import argparse
from pathlib import Path
from .common import load_config, read_json, write_json

def run(config_path, run_name):
    cfg=load_config(config_path); run_dir=Path(cfg["experiment"]["run_root"])/run_name; contract=read_json(str(run_dir/"environment_interface_contract.json")); obs_indices=contract.get("obs",{}).get("indices",{}); action_values=contract.get("action",{}).get("values",{})
    summary={"env_id":contract.get("env_id","Env_001"),"observation_interface":[f"obs[{idx}] = {item.get('name')}: {item.get('meaning')}" for idx,item in sorted(obs_indices.items(),key=lambda x:int(x[0]))],"action_interface":[f"action {value} = {item.get('name')}: {item.get('meaning')}" for value,item in sorted(action_values.items(),key=lambda x:int(x[0]))],"termination_interface":contract.get("termination",{}),"allowed_inputs":contract.get("reward_function",{}).get("allowed_inputs",[]),"forbidden_inputs":contract.get("reward_function",{}).get("forbidden_inputs",[]),"allowed_info_fields":contract.get("info",{}).get("allowed_fields",[]),"uncertain_info_fields":contract.get("info",{}).get("uncertain_fields",[]),"forbidden_info_fields":contract.get("info",{}).get("forbidden_fields",[]),"function_signature":contract.get("reward_function",{}).get("signature","def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):"),"notes_for_coder":["Do not use original_reward even though it is present in the function signature.","Do not invent info fields.","Do not use obs slices unless allowed_slices declares them.","Use only formula_plan.core_terms from reward_design_plan."]}
    write_json(str(run_dir/"environment_interface_summary.json"),summary); print(run_dir/"environment_interface_summary.json")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/env001_deepseek_rag.yaml"); ap.add_argument("--run-name",required=True); args=ap.parse_args(); run(args.config,args.run_name)
