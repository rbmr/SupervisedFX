import torch
import torch.nn as nn
import torch.optim as optim

class CuriosityModule(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.inverse_loss_fn = nn.CrossEntropyLoss()
        self.forward_loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, next_state, action_one_hot):
        inverse_input = torch.cat((state, next_state), dim=-1)
        pred_action_logits = self.inverse_model(inverse_input)

        forward_input = torch.cat((state, action_one_hot), dim=-1)
        pred_next_state = self.forward_model(forward_input)

        return pred_action_logits, pred_next_state

    def compute_intrinsic_reward(self, state, next_state, action_one_hot):
        _, pred_next_state = self.forward(state, next_state, action_one_hot)
        intrinsic_reward = torch.mean((pred_next_state - next_state) ** 2, dim=-1)
        return intrinsic_reward.detach().cpu().numpy()

    def update_batch(self, states, next_states, action_idxs, action_one_hots):
        states = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)
        action_idxs = torch.cat(action_idxs, dim=0)
        action_one_hots = torch.cat(action_one_hots, dim=0)

        pred_action_logits, pred_next_state = self.forward(states, next_states, action_one_hots)
        inverse_loss = self.inverse_loss_fn(pred_action_logits, action_idxs)
        forward_loss = self.forward_loss_fn(pred_next_state, next_states)
        loss = inverse_loss + forward_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
