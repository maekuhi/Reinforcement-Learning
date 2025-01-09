# Chapter2 _Multi-armed Bandit_
- The RL method uses training information that **evaluates** the actions taken rather than **instructs** by giving correct actions.
- Need for **active exploration**
- Purely evaluative feedback vs. purely instructive feedback: _evaluative feedback depends entirely on the action taken, whereas instructive feedback is independent of the action taken_
- The instructive feedback is the basis of **supervised learning** including _pattern classification, artificial neural networks, and system identification_

## K-armed Bandit Problem
- An agent + repeatedly a choice among k different options (actions) > a numerical reward chosen from a stationary probability distribution that depends on the selected action
- **The objective**: maximize the expected total reward over some time period (time steps)
- **The value of the action**: an expected or mean reward given that that action is selected
- The expected reward is: $q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$
- The estimated value of action: $Q_t(a)$
- We would like the estimation to be close to the real value
- **The greedy action** is the action with the highest estimated value
- Selecting the greedy action according to current knowledge means _exploiting_ resulting to maximize the expected reward on the _one step_
- selecting nongreedy action means _exploring_ may produce the greater total reward in the _long run_

## Action-value Methods




## Optimistic Initial Values
- The initial action-value estimations $Q_1(a)$ are important. _biased by initial estimates_
- 
