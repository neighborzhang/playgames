# Final Project - Deep learning in Playing Open AI Gym Games

### By Tianyi Zhang, Heran Zhang, Yizhi Zhu, Hongyi Li, Zuodi Na



# Introduction 


# literature review
## Simple DQN
## Double Q
## Dueling 
## Policy Gradient

# Theoretical Basis
## Reinforcement Learning
Reinforcement learning is similar to supervised learning in a way that the learning process is guided by "reward" of the "choice of action". Actions with positive reward is encouraged, while actions leading to negative reward is discouraged. 

## Markov Decision Process

To properly define the learning process, it is convenient to adopt Markov Decision Proecss. 
Here are the key components: States (S), Model (T), Action(A), Reward (R ), Policy($\pi$).

* States: S : a specific situation of the game
* Model: T(s, a, s’) (=P(s’|s, a)): T is a transition probability function, from situation s to s', which is determined by the learning algorithm. 
* Actions: A (s): A is an action the robot takes at state s.
* Reward: R (s), R (s,a), R (s,a,s'): Reward is determined by current state, action and future state(delayed reward).
* Policy: $\pi(s)$ -> a: Policy is the Strategy the robot takes, the policy function returns an action given any state s. 


## DQN Theory
The goal of learning is to maximize Q value, which is defined as a expectation of total sum of discounted rewards, given a policy ($\pi$) and initial state. 

$$
Q^{\pi}(s) = E \left[ \sum_{t}^{\infty} \gamma^{t} R(s_t) | \pi, s_0=s \right]
$$

* $\gamma$: discounted reward, with range　$0 \leq \gamma < 1$

* The optimal strategy at state s: 
$$
\pi^{*}(s) = argmax_a \sum_{s'} T(s, a, s') Q(s')
$$


* Bellman equation
$$
Q(s) = R(s) + \gamma max \sum_{s'} T(s, a, s') Q(s')
$$


## DQN Method
In practice, as we do not know about the future state Q, the Q function is estimated through an iteration process starting from an initial point. As we encourage actions with positive reward, discourage actions with negative reward, through millions iteration process, the robot will eventually find the optimal policy and reached a converaging value of Q function. The optimization process is essentially a Dynamic Programing problem, finding the fixed point of a function Q = F(Q).

The core algorithm is as follows:

$$
Q(s, a) \approx R(s, a) + \gamma max_{a'} E[Q(s', a')]
$$

$$
Q(s, a) = Q(s, a) + \alpha(R(s, a) + \gamma max_{a'} E[Q(s', a')] - Q(s, a))
$$
Q is updated as follows:
$$
Q(s,a) \leftarrow Q(s,a)+\alpha(R(s,a)+\gamma Q(s',a')-Q(s,a)) 
$$

* $\alpha$ is the learning rate, which can be a function of the result of the game.

Loss function: 

$$
L_\theta = E[\frac{1}{2} (\underline{R(s, a) + \gamma max_{a'} Q_{\theta_{i-1}}(s', a')} - Q_\theta(s, a))^2]
$$

Gradient update: 

$$
\nabla_\theta L(\theta_i) = E[(\underline{R(s, a) + \gamma max Q_{\theta_{i-1}}(s', a')} - Q_{\theta_i}(s, a))\nabla_\theta Q_{\theta_i}(s, a)]
$$


## Methods that improves Q-learning experience
### Experienced Replay
The experienced replay is important for getting a good result in Q-learning. Since normally, data is entered in a continuous chronological order, which may have correlations. The random shuffle of data in the learning process would improve the learning experience and reach the optimal policy faster. 
The experienced replay is a feature very similar to human minds behavior, often, we our memory and knowledge are shuffled in our mind, which lead to some enlightened ideas. 

### Fixed Target Q-Network
The fixed target Q-Network avoids possible oscillations that may occur in the Q-learning process. 

* Q(s,a,w) $\leftarrow$ R + $\gamma max_{a'} Q(s',a',w^{-})$
* $w^{-}$ is introduced in the original Q function, which is periodically updated $w^{-} \leftarrow w$.

This procedure can minimize the MSE between Q-network and Q-learning targets.

### Clipping the reward
Clipping the reward is a method to change the learning rate that assigns positive weighting to "winning" actions and negative weighting to "losing" actions (1 for winning game, -1 for a losing game). Although it can not differentiate between actions leading to small and large rewards, the merit of this method is that it prevents Q-values from being too large and ensures gradients are well-conditioned.  This method would generally make the Q-learning more efficiently. 



# DQN Implementation

Our DQN implementation use Deep Q Network, with features such as Experience Replay and epsilon reduction.

Here is a simple implementation done by chainer, which is composed of three parts agent and evironment setting. 

## Layers
3 hidden layers with 100 notes each, used RELU activation function. 
```python
            L1 = L.Linear(n_in, 100),
            L2 = L.Linear(100, 100),
            L3 = L.Linear(100, 100),
            Q_value = L.Linear(100, n_out, initialW=np.zeros((n_out, 100), dtype=np.float32))
            
            h = F.leaky_relu(self.L1(x))
            h = F.leaky_relu(self.L2(h))
            h = F.leaky_relu(self.L3(h))
            h = self.Q_value(h)
            
```
## Agent 
Agent contains the core algorithm of the robot, allowing the robot do forward and backward probagation update, store past memory, perform experience replay and epsilon reduction. 

### Parameter settings
```python
        self.loss = 0
        self.step = 0
        self.gamma = 0.99
        self.mem_size = 1000
        self.batch_size = 100
        self.epsilon = 1
        self.epsilon_decay = 0.005
        self.epsilon_min = 0
        self.exploration = 1000
        self.train_freq = 10
        self.target_update_freq = 20

```
### Features

#### Experience Replay
```python
def stock_experience(self, st, act, r, st_dash, ep_end):
        self.memory.append((st, act, r, st_dash, ep_end))
        if len(self.memory) > self.mem_size:
            self.memory.popleft()
            
def suffle_memory(self):
        mem = np.array(self.memory)
        return np.random.permutation(mem)

def experience_replay(self):
        mem = self.suffle_memory()
        perm = np.array(xrange(len(mem)))
        for start in perm[::self.batch_size]:
            index = perm[start:start+self.batch_size]
            batch = mem[index]
            st, act, r, st_d, ep_end = self.parse_batch(batch)
            self.model.zerograds()
            loss = self.forward(st, act, r, st_d, ep_end)
            loss.backward()
            self.optimizer.update()
```
#### Epsilon Reduction

```python
def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min and self.exploration < self.step:
            self.epsilon -= self.epsilon_decay
```


## Environment setup (main.py)
The main.py file has the ability to train a model and test it. 
The program is able to save a trained model and load it at a later time. 

```python
import gym, sys
import numpy as np

from agent import Agent

def main(env_name, render=False,load=False, seed=0):

    env = gym.make(env_name)

    n_st = env.observation_space.shape[0]
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        n_act = env.action_space.n
        action_list = range(0, n_act)
    elif type(env.action_space) == gym.spaces.box.Box:
        action_list = [np.array([a]) for a in [-2.0, 2.0]]
        n_act = len(action_list)

    agent = Agent(n_st, n_act, seed)
    if load:
        agent.load_model(model_path)

    for i_episode in xrange(1000):
        observation = env.reset()
        r_sum = 0
        q_list = []
        for t in xrange(200):
            if render:
                env.render()
            state = observation.astype(np.float32).reshape((1,n_st))
            act_i, q = agent.get_action(state)
            q_list.append(q)
            action = action_list[act_i]
            observation, reward, ep_end, _ = env.step(action)
            state_dash = observation.astype(np.float32).reshape((1,n_st))
            agent.stock_experience(state, act_i, reward, state_dash, ep_end)
            agent.train()
            r_sum += reward
            if ep_end:
                break
        print "\t".join(map(str,[i_episode, r_sum, agent.epsilon, agent.loss, sum(q_list)/float(t+1) ,agent.step]))
        agent.save_model(model_path)
   

if __name__=="__main__":
    env_name = sys.argv[1]
    main(env_name)


```
## 

# DQN Playing simple non-atari Games 

## CartPole-v0
After training around 80 episodes, the robot is able to balance the Pole very well, with a very small epsilon close to zero. In total, I have trained 1000 episodes, obtaining a high average reward (maximum 200).
![image](https://github.com/neighborzhang/dqn/blob/master/cartpole.png)


### Training Result

* num of episode, cumulative_reward, agent.epsilon, agent.loss, average-q ,agent.step

* sample result:

* graph

### Testing Result

## Pendulum-v0
### Training Result

* num of episode, cumulative_reward, agent.epsilon, agent.loss, average-q ,agent.step

* sample result:

* graph

### Testing Result
## Acrobot-v1
### Training Result

* num of episode, cumulative_reward, agent.epsilon, agent.loss, average-q ,agent.step

* sample result:

* graph

### Testing Result
## MountainCar-v0
### Training Result

* num of episode, cumulative_reward, agent.epsilon, agent.loss, average-q ,agent.step

* sample result:

* graph

### Testing Result




# Playing Atari Games Pong-v0

## DQN Method

### Four convolution layers
```python
    def __init__(self, n_history, n_act):
        super(ActionValue, self).__init__(
            l1=F.Convolution2D(n_history, 32, ksize=8, stride=4, nobias=False, wscale=np.sqrt(2)),
            l2=F.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, wscale=np.sqrt(2)),
            l3=F.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, wscale=np.sqrt(2)),
            l4=F.Linear(3136, 512, wscale=np.sqrt(2)),
            q_value=F.Linear(512, n_act,
                             initialW=np.zeros((n_act, 512),
                             dtype=np.float32))
        )

    def q_function(self, state):
        h1 = F.relu(self.l1(state/255.))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.q_value(h4)

```

### Hyper-Parameters
    gamma = 0.99  # Discount factor
    initial_exploration = 10**4  # Initial exploration
    replay_size = 32  # Replay (batch) size
    target_model_update_freq = 10**4  # Target model frequency
    data_size = 10**4  # Data size of history.
    img_size = 84  # 84x84 image input (fixed)

### RMSProp Initializer

```python
print("Initizlizing Optimizer")
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.model)
```
### Experience Replay

```python
def experience_replay(self, time):

        if self.initial_exploration < time:
            # Pick up replay_size number of samples from the Data
            if time < self.data_size:  # during the first sweep of the History Data
                replay_index = np.random.randint(0, time, (self.replay_size, 1))
            else:
                replay_index = np.random.randint(0, self.data_size, (self.replay_size, 1))

            hs = self.n_history
            ims = self.img_size
            rs = self.replay_size

            s_replay = np.ndarray(shape=(rs, hs, ims, ims), dtype=np.float32)
            a_replay = np.ndarray(shape=(rs, 1), dtype=np.int8)
            r_replay = np.ndarray(shape=(rs, 1), dtype=np.float32)
            s_dash_replay = np.ndarray(shape=(rs, hs, ims, ims), dtype=np.float32)
            episode_end_replay = np.ndarray(shape=(rs, 1), dtype=np.bool)
            for i in range(self.replay_size):
                s_replay[i] = np.asarray(self.replay_buffer[0][replay_index[i]], dtype=np.float32)
                a_replay[i] = self.replay_buffer[1][replay_index[i]]
                r_replay[i] = self.replay_buffer[2][replay_index[i]]
                s_dash_replay[i] = np.array(self.replay_buffer[3][replay_index[i]], dtype=np.float32)
                episode_end_replay[i] = self.replay_buffer[4][replay_index[i]]

            # Gradient-based update
            self.optimizer.zero_grads()
            loss, _ = self.get_loss(s_replay, a_replay, r_replay,   s_dash_replay, episode_end_replay)
            loss.backward()
            self.optimizer.update()
```

### $\epsilon$-Greedy

```python
def action_sample_e_greedy(self, state, epsilon):
        s = Variable(cuda.to_gpu(state))
        q = self.model.q_function(s)
        q = q.data.get()[0]

        if np.random.rand() < epsilon:
            action = np.random.randint(0, self.n_act)
            print("RANDOM : " + str(action))
        else:
            a = np.argmax(q)
            print("GREEDY  : " + str(a))
            action = np.asarray(a, dtype=np.int8)
            print(q)
        return action, q
```


### Training Result

* Output

* Graph


### Testing Result


## Policy Gradient 

Policy Gradient is very intuitive. The idea is to look at the future moves following a dynamic probability distribution. The distribution is a path dependent, building up from prevous learned probability distributions. The distribution is updated frequently, depending on the action and reward or the game result.  As for the game of the Pong, if the action of moving up would create a positive reward, for instance, successfully leading to a direct winning score, the probabilty of moving up would be higher in the future,vice versa. In short, the "good" move are encouraged, the "bad" moves are punished, through the updating of probability distribution.  

For general Atari games, policy gradient is implemented as follows: Taking input screen pixels (210x160x3), transform into (80x80x1), go through several layers and output a probability of the paddle moving Up or Down. After we have the probability of the moving distribution, for every iteration, we would be able to generate a move. Each move follows a Bernoulli distribution.  After performing the move in each frame, we would know the result of each action and reward. Therefore, given the reward, we will be able to feedback the information to the system and update the corresponding weighting parameters of different layers. 


### Hyperparameters

```python
A = 3   # 2, 3 for no-ops
H = 200 # number of hidden layer neurons
update_freq = 10
batch_size = 1000 # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?
render = False
device = 1
```

### Policy Gradient Core algorithm
```python
def policy_forward(x):
  if(len(x.shape)==1):
    x = x[np.newaxis,...]

  h = x.dot(model['W1'])
  h[h<0] = 0 # ReLU nonlinearity
  logp = h.dot(model['W2'])
  #p = sigmoid(logp)
  p = softmax(logp)

  return p, h # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = eph.T.dot(epdlogp)
  dh = epdlogp.dot(model['W2'].T)
  dh[eph <= 0] = 0 # backpro prelu

  t = time.time()

  if(be == cp):
    dh_gpu = cuda.to_gpu(dh, device=0)
    epx_gpu = cuda.to_gpu(epx.T, device=0)
    dW1 = cuda.to_cpu( epx_gpu.dot(dh_gpu) )
  else:
    dW1 = epx.T.dot(dh)

```

### Training Result



### Testing Result

# Code 
All the code can be found in github https://github.com/neighborzhang/dqn

# References

* Neural Network and deep learning course lecture notes
* Karpathy AI Blog http://karpathy.github.io/2016/05/31/rl/
* Human-Level Control through Deep Reinforcement Learning https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

* [Machine Learning: Reinforcement Learning] (https://www.udacity.com/course/machine-learning-reinforcement-learning--ud820)
 * [Markov Decision Processes and Reinforcement Learning] (https://s3.amazonaws.com/ml-class/notes/MDPIntro.pdf)
