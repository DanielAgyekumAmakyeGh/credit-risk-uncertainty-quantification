import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import warnings 
warnings.filterwarnings('ignore') 
 
class ReclamationEnv: 
    def __init__(self): 
        self.max_steps = 24 
        self.current_step = 0 
        self.ph = 4.5 
        self.biomass = 0.2 
        self.total_cost = 0 
    def reset(self): 
        self.current_step = 0 
        self.ph = 4.5 
        self.biomass = 0.2 
        self.total_cost = 0 
        return np.array([self.ph, self.biomass, 0.0]) 
    def step(self, action): 
        action_names = ['do_nothing', 'lime_low', 'lime_med', 'lime_high', 'compost', 'hydroseed', 'reshape_drainage', 'reseed_native'] 
        action_costs = [0, 500, 1000, 2000, 800, 300, 5000, 400] 
        cost = action_costs[action] 
        self.total_cost += cost 
        if action_names[action].startswith('lime'): 
            ph_boost = {'lime_low': 0.3, 'lime_med': 0.6, 'lime_high': 1.0}[action_names[action]] 
            self.ph += ph_boost * 0.08 
        elif action_names[action] == 'compost': 
            self.ph += 0.05 
            self.biomass += 0.06 
        elif action_names[action] in ['hydroseed', 'reseed_native']: 
            self.biomass += 0.10 
        self.ph = min(7.0, self.ph + 0.02) 
        self.biomass = min(1.0, self.biomass + 0.03) 
        ph_reward = max(0, (self.ph - 5.0) / 2.0) 
        reward = ph_reward + self.biomass - self.total_cost / 10000 
        self.current_step += 1 
        done = self.current_step 
        next_state = np.array([self.ph, self.biomass, self.current_step / self.max_steps]) 
        return next_state, reward, done, {'total_cost': self.total_cost, 'final_ph': self.ph} 
 
class SimplePolicy(nn.Module): 
    def __init__(self, state_dim=3, action_dim=8): 
        super().__init__() 
        self.net = nn.Sequential( 
            nn.Linear(state_dim, 64), nn.ReLU(), 
            nn.Linear(64, 64), nn.ReLU(), 
            nn.Linear(64, action_dim), nn.Softmax(dim=-1) 
        ) 
    def forward(self, x): 
        return self.net(x) 
 
def run_control(): 
    print("=" * 60) 
    print("PHASE 3: DEEP REINFORCEMENT LEARNING CONTROL") 
    print("=" * 60) 
    env = ReclamationEnv() 
    policy = SimplePolicy() 
    optimizer = optim.Adam(policy.parameters(), lr=0.001) 
    for episode in range(200): 
        state = env.reset() 
        episode_reward = 0 
        log_probs = [] 
        for _ in range(24): 
            state_tensor = torch.FloatTensor(state).unsqueeze(0) 
            probs = policy(state_tensor) 
            dist = torch.distributions.Categorical(probs) 
            action = dist.sample() 
            log_probs.append(dist.log_prob(action)) 
            next_state, reward, done, _ = env.step(action.item()) 
            episode_reward += reward 
            state = next_state 
            if done: 
                break 
        if log_probs: 
            loss = -torch.stack(log_probs).sum() * episode_reward 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
    torch.save(policy.state_dict(), 'sac_agent.pth') 
    print("Agent saved to sac_agent.pth") 
    print("PHASE 3 COMPLETE") 
    return policy 
 
if __name__ == "__main__": 
    run_control() 
