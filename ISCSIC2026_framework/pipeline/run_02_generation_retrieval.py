import argparse
from pathlib import Path
from .common import load_config, read_json
from .common_context import write_json, write_text, markdown_block
from rag.stage_retriever import retrieve_generation_knowledge, save_generation_knowledge, build_architect_generation_context


def run(config_path, run_name):
    cfg = load_config(config_path)
    run_dir = Path(cfg["experiment"]["run_root"]) / run_name
    environment_card = read_json(str(run_dir / "environment_card.json"))
    interface_contract = read_json(str(run_dir / "environment_interface_contract.json"))
    rag_cfg = cfg["rag"]
    result = retrieve_generation_knowledge(environment_card=environment_card, chunks_path=rag_cfg["generation_chunks_path"], table_chunks_path=rag_cfg["generation_table_chunks_path"], route_catalog_path=rag_cfg["route_catalog_path"], skeleton_catalog_path=rag_cfg["skeleton_catalog_path"])
    save_generation_knowledge(result, str(run_dir / "retrieved_generation_knowledge.raw.json"), str(run_dir / "retrieved_generation_knowledge.raw.md"))
    save_generation_knowledge(result, str(run_dir / "retrieved_generation_knowledge.json"), str(run_dir / "retrieved_generation_knowledge.md"))
    architect_context = build_architect_generation_context(environment_card, interface_contract, result)
    architect_input = {"environment_card": environment_card, "environment_interface_contract": interface_contract, "architect_generation_context": architect_context, "optional_memory_hits": []}
    write_json(run_dir / "llm_inputs/02_reward_architect.input.json", architect_input)
    write_text(run_dir / "human_review/02_reward_architect.input.md", markdown_block("environment_card", environment_card) + "\n" + markdown_block("environment_interface_contract", interface_contract) + "\n" + markdown_block("architect_generation_context", architect_context))
    write_json(run_dir / "raw_outputs/retrieved_generation_knowledge.raw.json", result)
    write_json(run_dir / "raw_outputs/architect_generation_context.raw.json", architect_context)
    print(run_dir / "retrieved_generation_knowledge.json")
    print(run_dir / "llm_inputs/02_reward_architect.input.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/env001_deepseek_rag.yaml"); ap.add_argument("--run-name", required=True); args = ap.parse_args(); run(args.config, args.run_name)
