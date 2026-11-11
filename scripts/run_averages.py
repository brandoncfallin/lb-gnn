import logging
import os
import subprocess
import sys
import time
from datetime import datetime

TOPOLOGY = "star"
NUM_TRIALS = 50
LAST_TRIAL_NUM = 0

# set up logging
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logs_dir = os.path.join(base_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/average_runs_{TOPOLOGY}.log"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)


def run_simulation(topology: str, trial_num: int):
    """
    Runs simulation for a specific topology and trial number

    Args:
        topology (str): Communication topology (ring, star, path, complete, or acyclic)
        trial_number (int): Current trial number

    Returns:
        None
    """

    try:
        logging.info(f"Starting trial {trial_num} for topology: {topology}")
        start_time = time.time()

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scripts_dir = os.path.join(base_dir, "scripts")
        result = subprocess.run(
            [sys.executable, os.path.join(scripts_dir, "single_run.py")],
            env={
                "TOPOLOGY": topology,
                "TRIAL_NUMBER": str(trial_num),
                **dict(os.environ),
            },
            capture_output=True,
            text=True,
        )

        # check if run was successful
        if result.returncode == 0:
            duration = time.time() - start_time
            logging.info(
                f"Successfully completed {topology} topology trial number {trial_num} in {duration:.2f} seconds"
            )
            return True
        else:
            logging.error(f"Failed to run {topology} topology trial num {trial_num}")
            logging.error(f"Error output: {result.stderr}")
            return False

    except Exception as e:
        logging.error(
            f"Error running topology {topology} trial number {trial_num}: {str(e)}"
        )
        return False


def main():
    start_time = datetime.now()
    logging.info(f"Starting averages batch run at {start_time}")

    results = {}

    for trial_idx in range(NUM_TRIALS):
        success = run_simulation(TOPOLOGY, trial_idx + LAST_TRIAL_NUM + 1)
        results[trial_idx] = "Success" if success else "Failed"

        # add a small delay between runs
        time.sleep(1)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logging.info("\nSimulation Batch Summary:")
    logging.info("-" * 30)
    for trial, status in results.items():
        logging.info(f"{trial}: {status}")
    logging.info(f"\nTotal execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    main()
