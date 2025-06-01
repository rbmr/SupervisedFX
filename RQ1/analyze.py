import logging

from RQ1.constants import EXPERIMENTS_DIR
from common.models.train_eval import analyse_results
from common.scripts import most_recent_modified

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Finished imports")

if __name__ == "__main__":

    experiment_dir = most_recent_modified(EXPERIMENTS_DIR)
    results_dir = experiment_dir / "results"
    analyse_results(results_dir)

