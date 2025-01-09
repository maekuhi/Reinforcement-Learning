# Chapter2 _Multi-armed Bandit_
- The RL method uses training information that **evaluates** the actions taken rather than **instructs** by giving correct actions.
- Need for **active exploration**
- Purely evaluative feedback vs. purely instructive feedback: _evaluative feedback depends entirely on the action taken, whereas instructive feedback is independent of the action taken_
- The instructive feedback is the basis of **supervised learning** including _pattern classification, artificial neural networks, and system identification_

## K-armed Bandit Problem
- An agent + repeatedly a choice among k different options (actions) --> a numerical reward chosen from a stationary probability distribution that depends on the selected action
- **The objective**: maximize the expected total reward over some time period (time steps)
- **The value of the action**: an expected or mean reward given that that action is selected
- The expected reward is: $q_*(a) \doteq \mathbb{E}[R_t \mid A_t = a]$
- The estimated value of action: $Q_t(a)$
- We would like the estimation to be close to the real value
- **The greedy action** is the action with the highest estimated value
- Selecting the greedy action according to current knowledge means _exploiting_ resulting to maximize the expected reward on the _one step_
- selecting nongreedy action means _exploring_ may produce the greater total reward in the _long run_

## Action-value Methods
- The true value of an action is the _mean reward_:
$Q_t(a) \doteq \frac{\text{sum of rewards when (a) taken prior to (t)}}{\text{number of times (a) taken prior to (t)}} = \frac{\sum_{i=1}^{t-1}R_i\cdot\mathbb{1}_{A_i = a}}{\sum\_{i=1}^{t-1}\mathbb{1}\_{A_i = a}}$
- 1 denotes the random variable that is 1 if _predicate_ is true and 0 if it is not
- If the denominator is zero --> define $Q_t(a)$ as some default value
- If the time goes to infinity --> $Q_t(a)$ converge to $q_*(a)$
- _sample-average method_ is not necessarily the best way
- If the is more than one greedy action: $A_t \doteq \arg\max_a Q_t(a)$
- $\epsilon$-greedy method
  - in the limit of as the number of steps increases, every action will be sampled an infinite number of times
  - ensuring all the $Q_t(a)$ converge to $q_*(a)$
  - the probability of selecting the optimal action converges to greater than $1-\epsilon$

## The 10-armed Testbed
![image](https://github.com/user-attachments/assets/c9edf1b7-a0b8-450c-9ad2-501ed1e631a8)


## Optimistic Initial Values
- The initial action-value estimations $Q_1(a)$ are important. _biased by initial estimates_
- 
