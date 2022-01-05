import gym
import numpy as np
import seaborn as sns
import torch
from torch.functional import Tensor
from torch.optim.optimizer import Optimizer
from torch.distributions import categorical

import model


def save_heatmap(policy_attn):
    attn = policy_attn.detach().numpy()
    attn = attn.reshape(-1, 1)
    heatmap = sns.heatmap(
        attn,
        annot=True,
        yticklabels=[
            'position of cart', 
            'velocity of cart', 
            'angle of pole', 
            'rotation rate of pole'
        ],
        xticklabels=False
    )
    figure = heatmap.get_figure()
    figure.tight_layout()
    figure.savefig('policy_attention.png', dpi=400)


class CartPoleTrainer():

    def __init__(self):
        obs_size = 4
        self.policies = model.CartPolePolicies(model.CartPolePoliciesParams(
            obs_size=obs_size,
            num_actions=2
        ))
        self.policies_selector = model.CartPolePolicySelector(
            num_policies=obs_size)
        self._obs_size = obs_size

    def get_policy_prob_dist(self):
        return categorical.Categorical(logits=self.policies_selector.policy_attn)

    def train(self, env: gym.Env):
        print('Training policies...')
        opt = torch.optim.AdamW(list(self.policies.parameters()))
        for i in range(self._obs_size):
            for _ in range(1000):
                self.train_one_eps(
                    opt, env, torch.tensor(i), 
                    train_policy=True, train_policy_selector=False)

        print('Training policy selector...')
        opt = torch.optim.AdamW(list(self.policies_selector.parameters()))
        for i in range(1000):
            policy_id = self.get_policy_prob_dist().sample()
            self.train_one_eps(
                opt, env, policy_id, train_policy=False, train_policy_selector=True)
        
        save_heatmap(self.policies_selector.policy_attn)

        policy_id = torch.argmax(self.policies_selector.policy_attn).item()
        self.demo(env, policy_id)

    def demo(self, env: gym.Env, policy_id: Tensor):
        env = gym.wrappers.RecordVideo(env, './videos', name_prefix=f'policy-{policy_id}')
        # env = Monitor(env, f'./videos/{policy_id}', force=True)
        obs = env.reset()
        done = False
        eps_len = 0
        while not done:
            env.render()
            eps_len += 1
            policies = self.policies(torch.tensor(obs))
            policy_logits = self.policies_selector(policies)
            action = torch.argmax(policy_logits[policy_id]).item()
            obs, _, done, _ = env.step(action)

    def train_one_eps(
        self, 
        opt: Optimizer, 
        env: gym.Env, 
        policy_id: Tensor, 
        train_policy: bool,
        train_policy_selector: bool
    ):
        obs = env.reset()
        observations = []
        actions = []
        rewards = []
        done = False
        eps_len = 0

        while not done:
            eps_len += 1
            policies = self.policies(torch.tensor(obs))
            policy_logits = self.policies_selector(policies)
            if train_policy:
                action_prob_dist = categorical.Categorical(
                    logits=policy_logits[policy_id])
                action = action_prob_dist.sample()
            else:
                action = torch.argmax(policy_logits[policy_id])
            obs, reward, done, _ = env.step(action.item())
            actions.append(action)
            rewards.append(reward)
            observations.append(obs)
        observations = torch.as_tensor(np.array(observations))
        actions = torch.as_tensor(actions)

        # Reward-to-go
        rewards_acc = [0]
        for r in rewards:
            rewards_acc.append(rewards_acc[-1] + r)
        total_reward = sum(rewards)
        weights = torch.as_tensor([
            total_reward - acc for acc in rewards_acc[:-1]
        ])

        # optimize
        opt.zero_grad()
        policies = self.policies(observations)
        policy_logits = self.policies_selector(policies)

        if train_policy_selector:
            policy_logp = self.get_policy_prob_dist().log_prob(policy_id)
            loss = -(policy_logp * weights[0])
        else:
            loss = torch.tensor(0.0)

        if train_policy:
            for step, a in enumerate(actions):
                action_prob_dist = categorical.Categorical(
                    logits=policy_logits[policy_id][step])
                actions_logp = action_prob_dist.log_prob(a)
                loss += -(actions_logp * weights[step])

        if train_policy or train_policy_selector:
            loss.backward()
            opt.step()

        return loss.item(), eps_len


def main():
    cart_pole_env = gym.make('CartPole-v1')
    trainer = CartPoleTrainer()
    trainer.train(cart_pole_env)


if __name__ == '__main__':
    main()
