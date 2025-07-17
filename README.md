# Flappy Bird Clone with DQN Implementation

---

## How it works

- Utilises a **Deep Q-Network (DQN)** to approximate the optimal Q-function.
- The **velocty and distances to pipes** are used as input to the network.
- The network outputs Q-values for two actions: **flap or fall**.
- The action with the highest Q-value is chosen. For example, if the outputs are (2, 1) for flap and fall respectively, the agent will choose to flap.
- Every state-action pair, along with its reward and the next state, is stored in an **experience replay buffer**.
- After each episode (when the bird dies), random samples from this buffer are used to train the DQN.
- Over successive generations, the agent will learn to perform significantly better, eventually surpassing human-level consistency.

---

## Work in progress disclaimer

- This is an **experimental learning project** and is in active development so early versions will likely fail to run or learn correctly.
- As the project evolves, the code will become better documented and more robust.
