import torch
import torch.nn as nn
import torch.optim as optim

class CuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Improved inverse model with more capacity
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Improved forward model
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
        )

        self.inverse_loss_fn = nn.CrossEntropyLoss()
        self.forward_loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Add running statistics for normalization
        self.register_buffer('intrinsic_reward_mean', torch.tensor(0.0))
        self.register_buffer('intrinsic_reward_std', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))

    def forward(self, state, next_state, action_one_hot):
        inverse_input = torch.cat((state, next_state), dim=-1)
        pred_action_logits = self.inverse_model(inverse_input)

        forward_input = torch.cat((state, action_one_hot), dim=-1)
        pred_next_state = self.forward_model(forward_input)

        return pred_action_logits, pred_next_state

    def compute_intrinsic_reward(self, state, next_state, action_one_hot):
        with torch.no_grad():
            forward_input = torch.cat((state, action_one_hot), dim=-1)
            pred_next_state = self.forward_model(forward_input)

            # Compute prediction error as intrinsic reward
            prediction_error = torch.mean((pred_next_state - next_state) ** 2, dim=-1)
            
            # Update running statistics for normalization
            if self.training:
                self.update_count += 1
                if self.update_count == 1:
                    self.intrinsic_reward_mean = prediction_error.mean()
                    self.intrinsic_reward_std = prediction_error.std() + 1e-8
                else:
                    # Exponential moving average
                    alpha = 0.001  # Slower adaptation
                    self.intrinsic_reward_mean = (1 - alpha) * self.intrinsic_reward_mean + alpha * prediction_error.mean()
                    self.intrinsic_reward_std = (1 - alpha) * self.intrinsic_reward_std + alpha * prediction_error.std()

            # More conservative normalization to avoid extreme values
            if self.intrinsic_reward_std > 1e-6:
                normalized_reward = (prediction_error - self.intrinsic_reward_mean) / (self.intrinsic_reward_std + 1e-6)
                # Clip to reasonable range before sigmoid
                normalized_reward = torch.clamp(normalized_reward, -5, 5)
                intrinsic_reward = torch.sigmoid(normalized_reward)
            else:
                # If std is too small, use raw prediction error with scaling
                intrinsic_reward = torch.clamp(prediction_error * 0.1, 0, 1)
            
            return intrinsic_reward.cpu().numpy()

    def update_batch(self, states, next_states, action_idxs, action_one_hots):
        states = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)
        action_idxs = torch.cat(action_idxs, dim=0)
        action_one_hots = torch.cat(action_one_hots, dim=0)

        pred_action_logits, pred_next_state = self.forward(states, next_states, action_one_hots)
        
        inverse_loss = self.inverse_loss_fn(pred_action_logits, action_idxs)
        forward_loss = self.forward_loss_fn(pred_next_state, next_states)
        
        # Weighted combination - give more weight to forward model initially
        total_loss = 0.2 * inverse_loss + 0.8 * forward_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        return total_loss.item(), inverse_loss.item(), forward_loss.item()