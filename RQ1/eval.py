import logging

from RQ1.constants import EXPERIMENTS_DIR
from RQ1.environment import get_environments
from common.models.train_eval import evaluate_models, analyse_results
from common.scripts import most_recent_modified

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Finished imports")

if __name__ == "__main__":

    train_env, eval_env = get_environments()

    experiment_dir = most_recent_modified(EXPERIMENTS_DIR)
    models_dir = experiment_dir / "models"
    results_dir = experiment_dir / "results"
    eval_envs = {
        "train": train_env,
        "eval": eval_env,
    }
    evaluate_models(models_dir, results_dir, eval_envs, eval_episodes=1)
    analyse_results(results_dir)

