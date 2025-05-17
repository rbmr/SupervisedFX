Reinforcement Learning

- **Reinforcement Learning**: machine learning paradigm where an agent learns to make sequential decisions by interacting with an environment.
- **Agent**: decision making entity
- **Environment**: everything the agent interacts with.
- The interaction proceeds in discrete time steps $t = 0, 1, 2, \dots$
    - Agent observes the current state.
    - Agent selects an action according to a policy.
    - Agent executes the action.
    - The environment transitions to a new state, and provides a reward signal to the agent.
    - Agent uses the reward and new state to update internal knowledge.
- The goal is to determine a policy, that maximizes the total reward.

Reinforcement Learning (Formally)

- **State** $s$: representation of the environment at a given step.
- **States** $S$: the set of all possible states.
- **Action** $a$: a choice the agent can make
- **Actions** $A(s)$: the set of all possible actions given a state $s$
- **Reward function** $R(s,a,s’)$: (expected) immediate reward received after taking action $a$ in state $s$, and transitioning to state $s’$.
- **Transition function** $P(s’|s, a)$: probability of transitioning to state $s’$ when taking action $a$ in state $s$.
- **Policy function** $\pi(a|s)$: (probabilistic) mapping of states $s \in S$ to actions $a \in A(s)$
- **Return** $G_t$: cumulative discounted reward of the agent starting from timestep $t$.
    - $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$
- **Discount factor** $\gamma \in [0, 1]$: determines how much the agent values future rewards.
- **State-Value function** $V^\pi(s)$: expected return starting from state $s$  under policy $\pi$.
    - $V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]$
- **Action-Value function** $Q^\pi(s, a)$: expected return starting from state $s$, taking action $a$, and then following policy $\pi$.
    - $Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$
- **Optimal State-Value function** $V^*(s)$:
    - $V^*(s)= \max_π​V^π(s)$
    - $V^*(s) = \max_{a\in A(s)}Q^*(s,a)$
- **Optimal Action-Value function** $Q^*(s,a)$:
    - $Q^*(s, a) = \max_\pi Q^\pi(s, a) \quad \forall s \in S, a \in A(s)$
    - $Q^*(s, a) = \mathbb{E}_{s' \sim P(s'|s,a)} \left[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \right]$
- **Optimal policy** $\pi^*$: the policy that maximizes the expected return from all states.
    - $\pi^* = \arg \max_{\pi} V_\pi(s) \quad \forall s \in S$
    - $\pi^*(s) = \arg\max_{a} Q^*(s, a) \quad \forall s \in S$
- **Advantage function** $A(s,a)$: function used to determine if an action is better or worse than expected. Used in actor-critic methods. 
    - $A(s,a) = Q(s,a)-V(s)$

Function approximation.

- **Function approximation method**: refers to how the agent represents and estimates complex functions like the state-value function $V^\pi(s)$, action-value function $Q^\pi(s, a)$, policy function $\pi(a|s)$, transition function $P(s’|s, a)$, or reward function $R(s,a,s’)$.
- **Tabular methods**: Each state, or state-action pair is mapped to a value. This mapping can be represented using a table. Only works for discrete, and small state-action spaces.
- **Linear function approximation**: the function is estimated as a weighted sum of features. Allows for continuous, and larger state-action spaces, but are limited in their ability to capture complex patterns or interactions.
- **Non-linear function approximation**: the function is estimated using non-linear models (e.g., neural networks, decision trees, kernel methods) . Captures more complex relationships but can be harder to train.

Deep Reinforcement Learning

- **Deep learning**: subfield of machine learning using multilayered neural networks.
- **Deep Reinforcement Learning** (**Deep RL**): Applies deep learning to RL algorithms. Can be utilized in function approximation, feature extraction, and joint-learning.

Model categorization

- **Policy vs Value-based** methods: Value-based methods try to learn a value function, and then derive the optimal policy using it. Policy-based methods try to learn the optimal policy directly.
- **Policy-gradient methods**: subset of policy-based methods that use gradient descent on the expected return to approximate the optimal policy.
- **Actor-critic methods**: combines value and policy based methods. It uses two seperate components: the actor, that is responsible for selecting actions (representing the policy), and the critic, evaluating the actions taken by the actor (using a value function).
- **Model-free vs. model-based**: Model-free methods learn directly from interactions without understanding the environment's dynamics, while model-based methods build a model of the environment in order to plan ahead.
- **On-policy vs. off-policy**: On-policy methods learn from the actions taken by the current policy, while off-policy methods learn from actions taken by a different policy (e.g., a past or exploratory one).
- **Algorithm**: the overall procedure used to learn how to behave optimally. It defines how the agent updates its policy or value estimates based on experience.

RL-Models

- **Tabular Q-learning**: A basic RL approach where each state-action pair has an explicitly stored Q-value in a table. Suitable for small, discrete state-action spaces.
- **PPO**: Proximal Policy Optimization, policy gradient method that clips the policy updates to increase stability.
- **DQN**: Deep Q-network, Q-learning method that replaces the Q-table with a neural network. Value-based, Good for discrete action spaces (Buy/Sell/(Hold)), Sample efficient: Deep Q-Learning
- **A2C**: Actor-Critic, Learns both the policy (actor) and value function (critic).
- **A3C**: async version of A2C. Works well with continuous input
- **SAC**: Actor-Critic. Maximizes both reward and entropy(keeps exploring). Highly stable, handles continuous action.
- **DDPG**: Actor-Critic. Like Q-learning for continuous actions. Works with continuous action space. Very sensitive to hyperparameters. Hard to train. Lowkey outdated.
- **TD3**: Actor-Critic. Improvement upon the DDPG model.

State representation

- **State Representation**: How the environment's current situation is encoded for the agent. A good state representation captures all relevant information needed for decision-making.
- **Dimensionality Reduction**: Techniques like PCA or autoencoders used to reduce state complexity while preserving essential information.
- **Feature Engineering**: The process of selecting or transforming raw data into informative inputs for the RL model (e.g., technical indicators, volatility, price momentum).

Rewards 

- **Reward Shaping**: Modifying the reward function to make learning more efficient (e.g., adding penalties for high drawdown or excessive trading).
- **Sparse vs. Dense Rewards**: Sparse: Rewards occur infrequently (e.g., only at episode end). Dense: Frequent feedback (e.g., after every action), often improves learning speed.
- **Risk-sensitive Reward Functions**: Incorporate metrics like Sharpe Ratio, Sortino Ratio, or drawdown into rewards to align learning with risk-adjusted return goals.

Exploration vs Exploitation

- **Epsilon-Greedy**: With probability ϵ\epsilonϵ, the agent explores (random action), otherwise exploits (best-known action). Simple but effective.
- **Boltzmann (Softmax) Exploration**: Chooses actions probabilistically based on their Q-values, with higher-valued actions more likely. Controlled by a temperature parameter.
- **Upper Confidence Bound (UCB)**: Balances exploration and exploitation using confidence intervals; actions with uncertain (but potentially high) value are explored more.
- **Thompson Sampling**: Probabilistic exploration using a Bayesian approach to sample from the posterior of action-value distributions.

Transfer Learning

- **Transfer Learning**: Leveraging knowledge (e.g., learned policies, Q-values, or features) from one task (e.g., a currency pair or market regime) to improve learning in another.
- **Policy Transfer**: Using a policy trained in one environment as a starting point in another.
- **Q-table Transfer**: Reusing Q-values for similar state-action pairs in a new task.
- **Domain Adaptation**: Adjusting the model to handle differences between source and target environments (e.g., volatility, spread behavior)

Trading

- **Foreign Exchange Market** (Forex): also called currency market, is a globalised decentralised market for the trading of currencies.
- **Forex currency pair**: a combination of currencies that are traded against eachother. First currency being the base currency, second being the quote currency. The pair shows how much quote currency you need to buy one unit of the base currency. Ex. EUR/USD, GBP/USD, USD/JPY, etc.
- **Timeframe**: the duration that each data point (or candlestick) represents on a chart. It determines the granularity of the data the RL agent learns from
- **Candlestick**: a visual representation of price movement over a specific timeframe. It provides four key pieces of information: Open: Price at the beginning of the timeframe. High: Highest price during the timeframe. Low: Lowest price during the timeframe. Close: Price at the end of the timeframe. Volume: Total transaction amount during the timeframe. (sometimes included in the data)

- **Ask vs Bid**
- **Spread**
- **Long, Short, Cash**
- **Buy, Sell, Hold**
- **Equity**
