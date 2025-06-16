import logging

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from RQ2.parameters import base_dqn_kwargs


# #################################
# # -- HyperParameters Phase 1 -- #
# #################################

def apply_patient_parameters(dqn_kwargs: dict) -> dict:
    """
    Apply parameters for a 'patient' DQN model.
    """
    dqn_kwargs['learning_rate'] = 0.00001
    dqn_kwargs['buffer_size'] = 100_000
    dqn_kwargs['batch_size'] = 512
    dqn_kwargs['tau'] = 0.001
    dqn_kwargs['train_freq'] = 512
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 5_000
    dqn_kwargs['exploration_fraction'] = 0.5
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.01
    return dqn_kwargs

def apply_cautious_parameters(dqn_kwargs: dict) -> dict:
    """
    Apply parameters for a 'cautious' DQN model.
    """
    dqn_kwargs['learning_rate'] = 0.00005
    dqn_kwargs['buffer_size'] = 60_000
    dqn_kwargs['batch_size'] = 512
    dqn_kwargs['tau'] = 0.0025
    dqn_kwargs['train_freq'] = 64
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 2500
    dqn_kwargs['exploration_fraction'] = 0.4
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.02
    return dqn_kwargs

def apply_balanced_parameters(dqn_kwargs: dict) -> dict:
    """
    Apply parameters for a 'balanced' DQN model.
    """
    dqn_kwargs['learning_rate'] = 0.0001
    dqn_kwargs['buffer_size'] = 30_000
    dqn_kwargs['batch_size'] = 256
    dqn_kwargs['tau'] = 0.005
    dqn_kwargs['train_freq'] = 16
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 1_000
    dqn_kwargs['exploration_fraction'] = 0.33
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.05
    return dqn_kwargs

def apply_aggressive_parameters(dqn_kwargs: dict) -> dict:
    """
    Apply parameters for an 'aggressive' DQN model.
    """
    dqn_kwargs['learning_rate'] = 0.001
    dqn_kwargs['buffer_size'] = 5_000
    dqn_kwargs['batch_size'] = 64
    dqn_kwargs['tau'] = 0.01
    dqn_kwargs['train_freq'] = 4
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 500
    dqn_kwargs['exploration_fraction'] = 0.25
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.1
    return dqn_kwargs

def HP_P1_patient(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_patient_parameters(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P1_cautious(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'cautious' DQN model.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P1_balanced(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'balanced' DQN model.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_balanced_parameters(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P1_aggressive(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes an 'aggressive' DQN model.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_aggressive_parameters(dqn_kwargs)
    return DQN(**dqn_kwargs)

# #################################
# # -- HyperParameters Phase 2 -- #
# #################################
def apply_reduced_capacity_network(dqn_kwargs: dict) -> dict:
    dqn_kwargs['policy_kwargs']['net_arch'] = [16,8]
    return dqn_kwargs

def apply_minimalist_capacity_network(dqn_kwargs: dict) -> dict:
    dqn_kwargs['policy_kwargs']['net_arch'] = [16]
    return dqn_kwargs

def apply_medium_symmetric_capacity_network(dqn_kwargs: dict) -> dict:
    dqn_kwargs['policy_kwargs']['net_arch'] = [32, 32]
    return dqn_kwargs

def apply_increased_capacity_network(dqn_kwargs: dict) -> dict:
    dqn_kwargs['policy_kwargs']['net_arch'] = [64, 32]
    return dqn_kwargs

def apply_symmetric_capacity_network(dqn_kwargs: dict) -> dict:
    dqn_kwargs['policy_kwargs']['net_arch'] = [64,64]
    return dqn_kwargs

def apply_high_capacity_network(dqn_kwargs: dict) -> dict:
    dqn_kwargs['policy_kwargs']['net_arch'] = [128, 64]
    return dqn_kwargs

def HP_P2_cautious_reduced(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    dqn_kwargs = apply_reduced_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_cautious_minimalist(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    dqn_kwargs = apply_minimalist_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_cautious_baseline(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_cautious_medium_symmetric(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    dqn_kwargs = apply_medium_symmetric_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_cautious_increased(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    dqn_kwargs = apply_increased_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_cautious_symmetric(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    dqn_kwargs = apply_symmetric_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_cautious_high(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_cautious_parameters(dqn_kwargs)
    dqn_kwargs = apply_high_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_balanced_reduced(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_balanced_parameters(dqn_kwargs)
    dqn_kwargs = apply_reduced_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_balanced_minimalist(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_balanced_parameters(dqn_kwargs)
    dqn_kwargs = apply_minimalist_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_balanced_baseline(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_balanced_parameters(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_balanced_medium_symmetric(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_balanced_parameters(dqn_kwargs)
    dqn_kwargs = apply_medium_symmetric_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_balanced_increased(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_balanced_parameters(dqn_kwargs)
    dqn_kwargs = apply_increased_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_balanced_symmetric(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_balanced_parameters(dqn_kwargs)
    dqn_kwargs = apply_symmetric_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P2_balanced_high(temp_env: DummyVecEnv) -> DQN:
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_balanced_parameters(dqn_kwargs)
    dqn_kwargs = apply_high_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

# #################################
# # -- HyperParameters Phase 3 -- #
# #################################

def apply_hybrid_parameters(dqn_kwargs: dict) -> dict:
    """
    Apply hybrid parameters that combine cautious and balanced approaches.
    """
    dqn_kwargs['learning_rate'] = 0.00075
    dqn_kwargs['buffer_size'] = 60_000
    dqn_kwargs['batch_size'] = 512
    dqn_kwargs['tau'] = 0.005
    dqn_kwargs['train_freq'] = 32
    dqn_kwargs['gradient_steps'] = 1
    dqn_kwargs['target_update_interval'] = 2250
    dqn_kwargs['exploration_fraction'] = 0.35
    dqn_kwargs['exploration_initial_eps'] = 1.0
    dqn_kwargs['exploration_final_eps'] = 0.05
    return dqn_kwargs

def HP_P3_hybdrid_minimalist(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'hybrid' DQN model with minimalist capacity.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_hybrid_parameters(dqn_kwargs)
    dqn_kwargs = apply_minimalist_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P3_hybrid_reduced(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'hybrid' DQN model with reduced capacity.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_hybrid_parameters(dqn_kwargs)
    dqn_kwargs = apply_reduced_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P3_hybrid_baseline(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'hybrid' DQN model with baseline capacity.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_hybrid_parameters(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P3_hybrid_increased(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'hybrid' DQN model with increased capacity.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_hybrid_parameters(dqn_kwargs)
    dqn_kwargs = apply_increased_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P3_hybrid_symmetric(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'hybrid' DQN model with symmetric capacity.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_hybrid_parameters(dqn_kwargs)
    dqn_kwargs = apply_symmetric_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P3_hybrid_medium_symmetric(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'hybrid' DQN model with medium symmetric capacity.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_hybrid_parameters(dqn_kwargs)
    dqn_kwargs = apply_medium_symmetric_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)

def HP_P3_hybrid_high(temp_env: DummyVecEnv) -> DQN:
    """
    Initializes a 'hybrid' DQN model with high capacity.
    """
    dqn_kwargs = base_dqn_kwargs(temp_env)
    dqn_kwargs = apply_hybrid_parameters(dqn_kwargs)
    dqn_kwargs = apply_high_capacity_network(dqn_kwargs)
    return DQN(**dqn_kwargs)