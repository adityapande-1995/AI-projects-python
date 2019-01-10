import gym
from ag import *
import os 

env = gym.make('CartPole-v0') 
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
n_episodes = 1001 

output_dir = 'model_output/cartpole/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

agent = DQNAgent(state_size, action_size) 
done = False

for e in range(n_episodes): # For each episode
    state = env.reset() 
    state = np.reshape(state, [1, state_size])
    
    for time in range(5000): # Run episode
        env.render()
        action = agent.act(state) # action is either 0 or 1 (move cart left or right); decide on one or other here
        next_state, reward, done, _ = env.step(action) 
        reward = reward if not done else -10 # reward +1 for each additional frame with pole upright        
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)         
        state = next_state    
        if done: # episode ends if agent drops pole or we reach timestep 5000
            print("episode: {}/{}, score: {}, e: {:.2}" .format(e, n_episodes, time, agent.epsilon))
            break 
    
    if len(agent.memory) > batch_size:
        agent.replay(batch_size) # train the agent by replaying the experiences of the episode
    
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
