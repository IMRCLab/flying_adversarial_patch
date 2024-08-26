import subprocess
import multiprocessing
import os
import time
import argparse

OUTPUT_FOLDER = 'results/yolo_logs'

def run_gpu_task(task_id):
    stdout_file = open(f"{OUTPUT_FOLDER}/output_task_{task_id}.log", "w")
    stderr_file = open(f"{OUTPUT_FOLDER}/error_task_{task_id}.log", "w")

    # Run the GPU task using subprocess
    subprocess.run(['python', 'src/attacks.py', "--file", "settings.yaml", "--task", str(task_id)], stdout=stdout_file, stderr=stderr_file)

    stdout_file.close()
    stderr_file.close()

def main():

    # starting task_id
    start_task_id = 456
    end_task_id = 1200

    # number of proceses that can run in parallel
    num_runs = 2 

    processes = []

    # ending task_id
    while start_task_id < end_task_id:

        # Use multiprocessing to run tasks in parallel
        while len(processes) < num_runs:
            p = multiprocessing.Process(target=run_gpu_task, args=(start_task_id,))
            p.start()
            processes.append(p)
            start_task_id += 1

        # Wait for all processes to complete
        for p in processes:
            if not p.is_alive():
                processes.remove(p)
        
        time.sleep(1)

if __name__ == "__main__":
    main()
