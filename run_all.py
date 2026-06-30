"""Single driver: build the model once, run all experiments (amortizes vLLM load)."""
import argparse
import time

from models.factory import get_model
import run_main
import run_rollout
import run_stoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--backend", default="vllm", choices=["hf", "vllm"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--tag", default="7b")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--roll_games", type=int, default=24)
    ap.add_argument("--roll_horizon", type=int, default=20)
    ap.add_argument("--stoch_games", type=int, default=40)
    args = ap.parse_args()

    t0 = time.time()
    llm = get_model(args.model, args.backend, args.tp)
    print(f"== model {llm.name} loaded in {time.time()-t0:.0f}s ==", flush=True)

    print("\n##### EXP1: one-step + counterfactual #####", flush=True)
    run_main.run(llm, args.games, args.horizon, f"results/exp1_onestep_{args.tag}.json")

    print("\n##### EXP2: rollout drift #####", flush=True)
    run_rollout.run(llm, args.roll_games, args.roll_horizon, f"results/exp2_rollout_{args.tag}.json")

    print("\n##### EXP3: stochasticity #####", flush=True)
    run_stoch.run(llm, args.stoch_games, args.roll_horizon, 6, f"results/exp3_stoch_{args.tag}.json")

    print(f"\n== ALL DONE in {time.time()-t0:.0f}s ==")


if __name__ == "__main__":
    main()
