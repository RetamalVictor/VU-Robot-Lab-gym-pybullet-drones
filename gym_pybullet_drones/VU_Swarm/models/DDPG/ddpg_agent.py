import numpy as np

import torch as T
import torch.nn.functional as F

from gym_pybullet_drones.VU_Swarm.models.DDPG.utils import ReplayBuffer, OUActionNoise
from gym_pybullet_drones.VU_Swarm.models.DDPG.DDPG_network import (
    ActorNetwork,
    CriticNetwork,
)


class Agent(object):
    def __init__(
        self,
        alpha,
        beta,
        input_dims,
        tau,
        checkpoint_dir,
        gamma=0.99,
        n_actions=2,
        max_size=1000000,
        layer1_size=400,
        layer2_size=300,
        batch_size=64,
    ):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size

        self.actor = ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="Actor",
            chkpt_dir=checkpoint_dir,
        )
        self.critic = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="Critic",
            chkpt_dir=checkpoint_dir,
        )

        self.target_actor = ActorNetwork(
            alpha,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="TargetActor",
            chkpt_dir=checkpoint_dir,
        )
        self.target_critic = CriticNetwork(
            beta,
            input_dims,
            layer1_size,
            layer2_size,
            n_actions=n_actions,
            name="TargetCritic",
            chkpt_dir=checkpoint_dir,
        )

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        """
        Observation: Matrix (num_Drones + 3,)
        """
        self.actor.eval()
        with T.no_grad():
            observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(observation).to(self.actor.device)
            mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return 0, 0
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return actor_loss, critic_loss

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_dict[name].clone()
            )
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print("Checking Actor parameters")

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print("Checking critic parameters")
        for param in current_critic_dict:
            print(
                param, T.equal(original_critic_dict[param], current_critic_dict[param])
            )
        input()
