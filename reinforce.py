#%% - Andriy Drozdyuk
import torch, gym
max_steps = 20_000; step = 0; lr = 0.005; γ = 0.9999
env = gym.make('CartPole-v0')

nn = torch.nn.Sequential(
    torch.nn.Linear(4, 64), torch.nn.ReLU(),
    torch.nn.Linear(64, env.action_space.n), torch.nn.Softmax(dim=-1)
)
optim = torch.optim.Adam(nn.parameters(), lr=lr)

while step <= max_steps:
    obs = torch.tensor(env.reset(), dtype=torch.float); done = False
    Actions, States, Rewards, EligibilityVector = [], [], [], []
    
    while not done:
        probs = nn(obs)
        c = torch.distributions.Categorical(probs=probs)        
        action = c.sample()        
        log_prob = c.log_prob(action)        
        action = action.item()
        
        obs_, rew, done, _info = env.step(action)
        step += 1
        
        Actions.append(action)
        States.append(obs)
        Rewards.append(rew)
        EligibilityVector.append(log_prob)

        obs = torch.tensor(obs_, dtype=torch.float)

    DiscountedReturns = []
    for t in range(len((Rewards))):
        G = 0.0
        for k, r in enumerate(Rewards[t:]):
            G += r
        DiscountedReturns.append(G)
    
    EligibilityVector = torch.stack(EligibilityVector)        
    DiscountedReturns = torch.tensor(DiscountedReturns, dtype=torch.float)

    loss = - torch.dot(EligibilityVector, DiscountedReturns)
        
    optim.zero_grad()
    loss.backward()
    optim.step()

    print(f'Step: {step}: Reward={sum(Rewards)}')


# %%

for _ in range(5):
    obs = torch.tensor(env.reset(), dtype=torch.float)    
    done = False
    env.render()
    Rewards = []
    while not done:
        probs = nn(obs)
        c = torch.distributions.Categorical(probs=probs)        
        action = c.sample()            
        action = action.item()
        
        obs_, rew, done, _info = env.step(action)
        Rewards.append(rew)
        env.render()

        obs = torch.tensor(obs_, dtype=torch.float)

    print(f'Reward: {sum(Rewards)}')
env.close()
# %%
