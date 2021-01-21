from collections import namedtuple, deque
import copy
import numpy as np
import random
import torch
import torch.nn.functional as F

from model import Actor, Critic

BUFFER_SIZE = 1_000_000  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-3               # for soft update of target parameters
LR_ACTOR = 1e-4          # learning rate of the actor
LR_CRITIC = 3e-4         # learning rate of the critic
SIGMA_DECAY = 0.99
SIGMA_MIN = 0.005

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """
    Interacts with and learns from the environment.

    Parameters
    ----------
    action_dim
    seed

    Attributes
    ----------
    actor_local
    actor_target
    actor_optimizer
    critic_local
    critic_target
    critic_optimizer
    noise
    noise_epsilon
    memory
    """
    def __init__(self, action_dim, seed=42):
        self.action_dim = action_dim
        self.seed = seed

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(action_dim=action_dim, seed=seed).to(DEVICE)
        self.actor_target = Actor(action_dim=action_dim, seed=seed).to(DEVICE)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor_local.parameters(), lr=LR_ACTOR, amsgrad=True,
        )

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(action_dim=action_dim, seed=seed).to(DEVICE)
        self.critic_target = Critic(action_dim=action_dim, seed=seed).to(DEVICE)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic_local.parameters(), lr=LR_CRITIC, amsgrad=True,
        )

        # Noise process
        self.noise = OUNoise(action_dim, seed)
        self.noise_epsilon = 1

        # Replay memory
        self.memory = ReplayBuffer(action_dim, BUFFER_SIZE, BATCH_SIZE, seed)
    
    def step(self, state, action, reward, next_state, is_done):
        """
        Save experience in replay memory, and use random sample from buffer to
        learn.
        """
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, is_done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(
            states.copy().transpose((2, 0, 1))
        ).float().to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()

        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1).flatten()

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience
        tuples.

        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters
        ----------
        experiences : Tuple[torch.Tensor]
            Tuple of (s, a, r, s', done) tensors.
        gamma : float
            Discount factor.
        """
        states, actions, rewards, next_states, is_dones = experiences

        # --------------------------- Update Critic ---------------------------
        # Get predicted next-state actions and Q values from target models.
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next).detach()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - is_dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --------------------------- Update Actor ----------------------------
        # Compute actor loss.
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- Update target networks -----------------------
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
        local_model : PyTorch model
            Model to copy weight from.
        target_model : PyTorch model
            Model to copy weight to.
        tau : PyTorch float
            Interpolation parameter.
        """
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data
            )


class OUNoise:
    """
    Ornstein-Uhlenbeck process.

    Parameters
    ----------
    action_dim
    seed
    mu
    theta
    sigma
    """
    def __init__(self,
                 action_dim,
                 seed,
                 mu=0.,
                 theta=0.15,
                 sigma=0.2):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.seed = seed
        self.state = self.mu.copy()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        if self.sigma > SIGMA_MIN:
            self.sigma *= SIGMA_DECAY

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = (self.theta * (self.mu - x) +
              self.sigma * (np.random.rand(*x.shape) - 0.5))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.

    Parameters
    ----------
    action_dim
    buffer_size
    batch_size
    seed
    """

    def __init__(self, action_dim, buffer_size, batch_size, seed):
        self.action_dim = action_dim
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )
        self.seed = seed
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack(
            [e.state.transpose((2, 0, 1))[np.newaxis, :] for e in
             experiences if e is not None])
        ).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])
        ).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])
        ).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state.transpose((2, 0, 1))[np.newaxis, :] for e in
             experiences if e is not None])
        ).float().to(DEVICE)
        is_dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]
        ).astype(np.uint8)).float().to(DEVICE)

        return states, actions, rewards, next_states, is_dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
