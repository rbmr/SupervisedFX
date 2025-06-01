import logging
from datetime import datetime

from stable_baselines3 import A2C

from RQ1.constants import EXPERIMENT_NAME_FORMAT, EXPERIMENTS_DIR
from RQ1.environment import get_environments
from common.constants import SEED, DEVICE
from common.envs.callbacks import SaveOnEpisodeEndCallback, ActionHistogramCallback
from common.models.train_eval import train_model
from common.models.utils import save_model_with_metadata

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Finished imports")

if __name__ == "__main__":

    train_env, _ = get_environments()

    logging.info("Creating model...")

    model = A2C(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=1e-4,
        n_steps=5_000,
        gamma=0.995,
        gae_lambda=0.9,
        ent_coef=0.005,
        vf_coef=0.4,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=SEED,
        verbose=1,
        device="cpu"
    )

    logging.info("Model created.")

    experiment_name = datetime.now().strftime(EXPERIMENT_NAME_FORMAT)
    experiment_dir = EXPERIMENTS_DIR / experiment_name
    models_dir = experiment_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    callback = [SaveOnEpisodeEndCallback(models_dir),
                ActionHistogramCallback(train_env, log_freq=model.n_steps)]
    train_model(model, train_env, train_episodes=20, callback=callback)
    save_model_with_metadata(model, models_dir / "model_final.zip")
