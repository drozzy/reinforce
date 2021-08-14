# %% - Andriy Drozdyuk

import torch
import gym

max_steps = 50_000
step = 0
lr = 0.005
γ = 0.9999

env = gym.make('CartPole-v0')

nn = torch.nn.Sequential(
    torch.nn.Linear(4, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, env.action_space.n),
    torch.nn.Softmax(dim=-1)
)
optim = torch.optim.Adam(nn.parameters(), lr=lr)

while step <= max_steps:
    obs = torch.tensor(env.reset(), dtype=torch.float)    
    done = False
    Actions, States, Rewards = [], [], []

    while not done:
        probs = nn(obs)
        dist = torch.distributions.Categorical(probs=probs)        
        action = dist.sample().item()
        obs_, rew, done, _ = env.step(action)
        
        Actions.append(torch.tensor(action, dtype=torch.int))
        States.append(obs)
        Rewards.append(rew)

        obs = torch.tensor(obs_, dtype=torch.float)

        step += 1
        
    DiscountedReturns = []
    for t in range(len(Rewards)):
        G = 0.0
        for k, r in enumerate(Rewards[t:]):
            G += (γ**k)*r
        DiscountedReturns.append(G)
    
    for State, Action, G in zip(States, Actions, DiscountedReturns):
        probs = nn(State)
        dist = torch.distributions.Categorical(probs=probs)    
        log_prob = dist.log_prob(Action)
        
        loss = - log_prob*G
        
        optim.zero_grad()
        loss.backward()
        optim.step()


# %% Play

for _ in range(5):
    Rewards = []
    
    obs = torch.tensor(env.reset(), dtype=torch.float)    
    done = False
    env.render()
    
    while not done:
        probs = nn(obs)
        c = torch.distributions.Categorical(probs=probs)        
        action = c.sample().item()
        
        obs_, rew, done, _info = env.step(action)
        env.render()

        obs = torch.tensor(obs_, dtype=torch.float)

        Rewards.append(rew)
    

    print(f'Reward: {sum(Rewards)}')
env.close()
# %%
