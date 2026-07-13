from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from .agent.controller import RewardDesignAgent
from .agent.planner import RuleBasedExpertPlanner
from .agent.stop_policy import StopPolicy
from .execution.mock_backend import MockBackend
from .execution.production_backend import ProductionBackend
from .llm import DeepSeekExpertPlanner
from llm_clients.deepseek_client import DeepSeekClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Expert Reward Agent v2.")
    parser.add_argument("--mode", choices=["mock", "api-check", "run"], default="mock")
    parser.add_argument("--config", type=Path, default=Path("expert_reward_agent_v2/configs/env001_api_smoke.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs_v2/mock_run"))
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=None, help="Override the production training seed.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.mode == "api-check":
        _check_api(args.config)
        return
    if args.mode == "run":
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        _preflight(config)
        llm = config["llm"]
        backend = ProductionBackend(args.config, args.run_dir, seed=args.seed)
        planner = DeepSeekExpertPlanner(
            model=llm["model"],
            api_key_env=llm["api_key_env"],
            base_url=llm["base_url"],
            temperature=float(llm.get("planner_temperature", 0.1)),
            max_tokens=int(llm.get("planner_max_tokens", 4096)),
            record_dir=args.run_dir / "llm_calls",
        )
        max_iterations = int(config["agent"]["max_scientific_iterations"])
    else:
        backend = MockBackend()
        planner = RuleBasedExpertPlanner()
        max_iterations = args.max_iterations
    agent = RewardDesignAgent(
        backend=backend,
        planner=planner,
        run_dir=args.run_dir,
        knowledge_dir=root / "knowledge" / "cards",
        stop_policy=StopPolicy(max_scientific_iterations=max_iterations),
    )
    state = agent.run()
    print(json.dumps(state.to_dict(), indent=2))


def _check_api(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _preflight(config)
    llm = config["llm"]
    client = DeepSeekClient(api_key_env=llm["api_key_env"], base_url=llm["base_url"])
    result = client.chat_json(
        model=llm["model"],
        system_prompt="Return only a JSON object.",
        user_prompt='Return {"status":"ok","purpose":"reward-agent-api-check"}.',
        temperature=0.0,
        max_tokens=128,
    )
    if result.get("status") != "ok":
        raise RuntimeError(f"Unexpected API check response: {result}")
    print(json.dumps(result, ensure_ascii=False))


def _preflight(config: dict) -> None:
    required = ["llm", "inputs", "agent", "training"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    env_name = config["llm"].get("api_key_env", "DEEPSEEK_API_KEY")
    if not os.environ.get(env_name):
        raise RuntimeError(f"Missing API key environment variable: {env_name}")


if __name__ == "__main__":
    main()
