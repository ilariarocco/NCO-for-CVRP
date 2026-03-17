# Learning Optimization for the TSP & VRP
This repo is for the Thesis "Learning Optimization per il Problema del Commesso Viaggiatore e del Vehicle Routing".

This repo contains the codes that uses an MLP agent to create a heuristic for the CVRP (with one vehicle).

## Set up
This repository uses Python 3.12.7. Follow the official Python documentation to install this version of Python before moving on.

After installing Python, you need to install all the dependencies (better in a virtual environment):

**(Optional) Create the virtual environment and activate it**

```bash
python3.12 -m venv .venv
```

On Linux:
```bash
source .venv/bin/activate
```

On Windows:
```bat
.venv\Scripts\activate.bat
```

**Install the required packages**
```bash
pip install -r requirements.txt
```

## Test the components
To allow for easy understanding, there is a Python file for each component, namely the Environment, the Agent and the REINFORCE algorithm. There is also a Python file which defined a greedy and a random policy, used as baselines in the REINFORCE algorithm.

Each component can be tested by calling
```bash
python <filename>.py
```

### Environment
Environment codes the simulator that at each episode generate a random instance with `n_nodes` in the [0, 1]² square.

The environment is written using the **Gymnasium** package, where the user only needs to define the `reset` and `step` method in order to define its custom environment.

The `reset` method is responsible for generating new instances whenever an episode ends, while the `step` method is responsible for updating the state and computing the reward when the agent sends an action.

Testing the environment, i.e. calling

```bash
python vrp_environment.py
```

creates a random instance with n nodes, and an agent acting according to a random policy, and show a video of the agent interacting with the environment. Also, the final reward (i.e. the negative of the length of the tour) gets printed.

In the rendered video, the current node is highlighted in red and whenever the agent acts, the edge between the current node and the next node is drawn.

### Agent
The agent is composed of the Deep Neural Network definition (that is an MLP) and of the forward method that takes as input the state of the environment and process it.

Namely, the nodes features are concatenated (the O matrix is flattened) and the first and current node features are appended.
Moreover, the masking vector is computed and applied before the *softmax* activation function.

When calling
```bash
python vrp_agent.py
```
a random instance with n nodes is spawned and the MLP agent interacts with it.

If no checkpoint model is provided, then the agent will act randomly.
Otherwise, if the user provides a checkpoint model, then the agent will act according to the training received.

To check the confidence of the agent before and after training, at each action, the policy probabilities are printed.

## Baselines
The baselines are static policies (i.e. they can not be trained) used to compare the performance of our agent.

There are two baselines: the random policy and the greedy policy.
The first one select the next node to visit randomly but ensuring the feasibility of the solution; the second policy always select the closest feasible node (at the first step, it always selects the first node in the list).

### REINFORCE
The REINFORCE algorithm is the core of this repository, where the training for the agent happens.

In the `vrp_reinforce.py` file, all the pieces (environment, agent, baseline policy) are used to perform both the data collection and the update of the DNN parameters.

## Philosophy
The aims of the code are simplicity and clarity, not performances and best practices.

For this reason, training takes a lot of time and while the agent performance improves, it does not reach satisfactory results in a reasonable amount of time.

### Results
You can run the code by typing
```bash
python vrp_reinforce.py
```
in your terminal.

We can see that slowly over time, the agent is learning how to perform better than baselines.