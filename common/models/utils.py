import json
import zipfile
from pathlib import Path
from typing import Generator

import logging
from stable_baselines3 import PPO, DQN, SAC, TD3, DDPG, A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib import RecurrentPPO

ALGORITHM_MAP = {
    'PPO': PPO,
    'A2C': A2C,
    'SAC': SAC,
    'TD3': TD3,
    'DQN': DQN,
    'DDPG': DDPG,
    'RECURRENTPPO': RecurrentPPO
}
METADATA_FILE = "custom_metadata.json"

def get_device(model: BaseAlgorithm) -> str:
    if hasattr(model, 'device'):
        return str(model.device)
    if hasattr(model, 'policy') and hasattr(model.policy, 'device'):
        return str(model.policy.device)
    return "auto"

def save_model_with_metadata(model: BaseAlgorithm, path: Path, **metadata) -> None:
    """
    Save model with additional metadata that can be used to dynamically load the model.
    """
    # Validate input
    if path.suffix != ".zip":
        raise ValueError(f"{path} is not a zip file")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the model
    model.save(path)

    # Create metadata
    custom_metadata = {
        "algorithm": model.__class__.__name__,
        "device": get_device(model),
        **metadata
    }

    # Add to zip
    with zipfile.ZipFile(f"{path}", 'a') as zip_file:
        zip_file.writestr(METADATA_FILE, json.dumps(custom_metadata, indent=4))

def load_model_with_metadata(path: Path) -> BaseAlgorithm:
    """
    Dynamically load any model using additional metadata.
    """
    # Validat input
    if not path.exists():
        raise ValueError(f"{path} does not exist")
    if not path.is_file():
        raise ValueError(f"{path} is not a file")
    if path.suffix != ".zip":
        raise ValueError(f"{path} is not a zip file")

    # Load metadata
    with zipfile.ZipFile(f"{path}", 'r') as zip_file:
        if not METADATA_FILE in zip_file.namelist():
            raise ValueError(f"Zip file at {path} is missing metadata")
        with zip_file.open(METADATA_FILE) as f:
            metadata = json.load(f)

    algorithm = metadata.get("algorithm", None)
    if algorithm is None:
        raise ValueError(f"{metadata} does not contain algorithm metadata")

    algorithm_class = ALGORITHM_MAP.get(algorithm.upper(), None)
    if algorithm_class is None:
        raise ValueError(f"{algorithm} is not a recognized algorithm")

    device = metadata.get("device", "auto")
    model = algorithm_class.load(path, device=device)
    return model

def load_models(models_dir: Path) -> Generator[tuple[str, BaseAlgorithm], None, None]:
    """
    Generator that yields all the models in a directory.
    Expects models to have metadata.
    """
    if not models_dir.is_dir():
        raise ValueError(f"{models_dir} is not a directory")

    i = 0
    model_queue = list(f for f in models_dir.glob("*.zip") if f.is_file())
    model_queue.sort(key=lambda x: x.stat().st_mtime) # sort on last modified

    logging.info(f"Found {len(model_queue)} model zips in '{models_dir}'.")

    while i < len(model_queue):

        # Get current model_zip, skip to next model.
        model_zip = model_queue[i]
        i += 1

        # Load and yield model
        logging.info(f"Loading model from {model_zip}...")
        model_name = model_zip.stem
        model = load_model_with_metadata(model_zip)
        logging.info(f"{model.__class__.__name__} model loaded from {model_zip}.")
        yield model_name, model

        # Add newly created zips to the queue.
        model_zips: set[Path] = set(f for f in models_dir.glob("*.zip") if f.is_file())
        new_model_zips = list(model_zips - set(model_queue))
        new_model_zips.sort(key=lambda x: x.stat().st_mtime)
        if len(new_model_zips) > 0:
            new_models = ", ".join(str(f.name) for f in new_model_zips)
            logging.info(f"Found {len(new_model_zips)} new model zips ({new_models}). Adding them to the queue.")
        model_queue.extend(new_model_zips)

