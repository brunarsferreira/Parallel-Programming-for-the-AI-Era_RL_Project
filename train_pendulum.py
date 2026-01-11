import sys
sys.stdout.reconfigure(line_buffering=True)  # Force unbuffered output to be able to see stdout in real-time

import ray
from ray.rllib.algorithms.ppo import PPOConfig
import time
import argparse
import json
import csv
import os
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--env-runners', type=int, default=2)
    parser.add_argument('--output', type=str, default='results_pendulum')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Ray cluster
    ray.init(address="auto", ignore_reinit_error=True)
    
    nodes = ray.nodes()
    num_nodes = len([n for n in nodes if n.get("Alive", False)])
    total_cpus = int(ray.available_resources().get("CPU", 1))
    num_env_runners = args.env_runners
    
    print(f"Cluster: {num_nodes} nodes, {total_cpus} CPUs, {num_env_runners} env runners", flush=True)

    config = (
        PPOConfig()
        .environment("Pendulum-v1")
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=4,  # 4 envs per worker = more samples faster
            observation_filter="MeanStdFilter",
            rollout_fragment_length="auto",  # RLlibs calculate optimal value
        )
        .training(
            lr=0.0001,              
            gamma=0.9,              
            train_batch_size=8000,  # Bigger batch = more stable gradients
            minibatch_size=64,      # Smaller minibatch
            num_epochs=5,           
            lambda_=0.9,            
            clip_param=0.2,
            vf_loss_coeff=1.0,
            entropy_coeff=0.001,    # a bit of entropy for exploration
            grad_clip=0.5,          
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_interval=None,
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            }
        )
    )
    
    algo = config.build_algo()
    print(f"Training for {args.iterations} iterations...", flush=True)
    
    start_time = time.time()
    history = []
    
    for i in range(args.iterations):
        t0 = time.time()
        print(f"  Starting iteration {i+1}...", flush=True)
        result = algo.train()
        dt = time.time() - t0
        
        # Extract reward from result
        reward = result.get("env_runners", {}).get("episode_return_mean")
        if reward is None:
            reward = result.get("episode_return_mean")
        
        history.append({
            'iteration': i + 1,
            'episode_reward_mean': reward,
            'episode_len_mean': 200,
            'iteration_time': dt,
            'elapsed_time': time.time() - start_time
        })
        
        reward_str = f"{reward:.1f}" if reward is not None else "N/A"
        if (i + 1) % 10 == 0 or i < 5:
            print(f"  [{i+1:3d}] reward={reward_str}, time={dt:.1f}s", flush=True)
    
    total_time = time.time() - start_time
    final_reward = next((h['episode_reward_mean'] for h in reversed(history) if h['episode_reward_mean']), None)
    
    final_str = f"{final_reward:.1f}" if final_reward is not None else "N/A"
    print(f"Done: {total_time:.1f}s total, {total_time/args.iterations:.2f}s/iter, final={final_str}", flush=True)
    
    # Evaluate
    eval_reward = None
    try:
        print("Running evaluation...", flush=True)
        eval_result = algo.evaluate()
        eval_reward = eval_result.get("env_runners", {}).get("episode_return_mean")
        eval_str = f"{eval_reward:.1f}" if eval_reward is not None else "N/A"
        print(f"Eval: {eval_str}", flush=True)
    except Exception as e:
        print(f"Eval failed: {e}", flush=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"nodes{num_nodes}_runners{num_env_runners}_{timestamp}"
    # Training history CSV
    csv_path = os.path.join(args.output, f"training_history_{run_id}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iteration', 'episode_reward_mean', 
                                                'episode_len_mean', 'iteration_time', 'elapsed_time'])
        writer.writeheader()
        writer.writerows(history)
    
    # Summary
    summary = {
        'run_id': run_id, 'timestamp': timestamp,
        'num_nodes': num_nodes, 'total_cpus': total_cpus,
        'num_env_runners': num_env_runners, 'num_iterations': args.iterations,
        'total_time_seconds': total_time, 'time_per_iteration': total_time / args.iterations,
        'final_reward': final_reward, 'eval_reward': eval_reward
    }
    
    with open(os.path.join(args.output, f"summary_{run_id}.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # write to all_runs.csv
    combined_csv = os.path.join(args.output, "all_runs.csv")
    file_exists = os.path.exists(combined_csv)
    with open(combined_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)
    
    print(f"Saved to {args.output}/", flush=True)

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()