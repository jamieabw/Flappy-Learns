# Flappy Bird Clone with DQN Implementation

---

## How it works

- Utilises a **Deep Q-Network (DQN)** with optional **Double Deep Q-Learning** to approximate the optimal Q-function.  
- The network takes as input the **velocity, current y-position, and distances to the pipes**.  
- It outputs Q-values for two actions: **flap** or **fall**.  
- The action with the highest Q-value is selected. For example, if the outputs are (2, 1) for flap and fall respectively, the agent will choose to flap.  
- Every state-action pair, along with its reward and the next state, is stored in a **prioritised experience replay buffer**.  
- After each frame, random samples from this buffer are used to train the DQN.  
- Over successive generations, the agent learns to perform significantly better, eventually surpassing human-level consistency, although this may require tens of thousands of generations.

---

## Disclaimer

This is an **experimental learning project** implemented purely with NumPy for the DQN and DQL process. Consequently, reaching better than human consistency will likely require running for tens of thousands of generations due to the limited input features. Frameworks such as TensorFlow utilise raw frame inputs and highly optimised layers, which considerably improve performance. However, implementing such optimisations is unrealistic for a purely NumPy-based approach.
