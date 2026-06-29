import argparse
from .run_01_environment_analyzer import run as run_env
from .run_02_generation_retrieval import run as run_ret
from .run_03_reward_architect import run as run_arch
from .run_035_make_interface_summary import run as run_iface
from .run_04_reward_coder import run as run_code

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/env001_deepseek_rag.yaml"); ap.add_argument("--run-name",default="deepseek_run_001"); ap.add_argument("--mock",action="store_true"); args=ap.parse_args()
    run_env(args.config,args.run_name,mock=args.mock); run_ret(args.config,args.run_name); run_arch(args.config,args.run_name,mock=args.mock); run_iface(args.config,args.run_name); run_code(args.config,args.run_name,mock=args.mock)
if __name__=="__main__": main()
