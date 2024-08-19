import subprocess
import multiprocessing
import os
import time
import argparse

def run_gpu_task(task_id):
    stdout_file = open(f"results/yolo_logs/output_task_{task_id}.log", "w")
    stderr_file = open(f"results/yolo_logs/error_task_{task_id}.log", "w")

    # Run the GPU task using subprocess
    subprocess.run(['python', 'src/attacks.py', "--file", "settings.yaml", "--task", str(task_id)], stdout=stdout_file, stderr=stderr_file)

    stdout_file.close()
    stderr_file.close()

def main():
    task_id = 44
    # Number of times to run the task
    while True:
        num_runs = 3  # Adjust this number as needed

        # Use multiprocessing to run tasks in parallel
        processes = []
        for _ in range(num_runs):
            p = multiprocessing.Process(target=run_gpu_task, args=(task_id,))
            p.start()
            processes.append(p)
            task_id += 1

        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        time.sleep(1)

if __name__ == "__main__":
    main()
