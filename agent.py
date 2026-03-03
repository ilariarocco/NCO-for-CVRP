import torch
import torch.nn as nn
import numpy as np
import pygame

class MLPAgent(nn.Module):
    """RL agent that uses an MLP network to decide the probabilities to assign to each action.
    
    It is composed by an input Linear layer that takes as input the xy-coordinates of the nodes of the problem,
    and the xy-coordinates of the first visited node and current node.
    Hence, the length of the input vector is n_nodes * 2 + 2*2. The output dimension is hidden_dim.
    After the input layer, is an activation layer, where the activation function is the Tanh function.
    Then there is another Linear layer. This is a hidden layer with input hidden_dim and output hidden_dim.
    This is followed by another activation layer with Tanh function.
    Finally, the there is a Linear layer that goes from dimension 64 to dimension n_nodes (number of actions).
    As last is the softmax layer, that is applied after masking (instead of before as explained in the paper) 
    for numerical reasons.
    """
    def __init__(self, n_nodes: int, hidden_dim: int = 64):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_nodes * 2 + 4, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, n_nodes),
        )
        self.softmax = torch.nn.Softmax(dim=-1) 

    def forward(self, state: np.ndarray) -> torch.Tensor:
        O_matrix = torch.from_numpy(state["nodes"]).float()
        l_t = state["visited"]
        if len(l_t) > 0:
            first_node = O_matrix[:, l_t[0]]
            current_node = O_matrix[:, l_t[-1]]
        else:
            first_node = -torch.ones(2)
            current_node = -torch.ones(2)

        o_t = torch.cat([O_matrix.T.flatten(), first_node, current_node]).unsqueeze(0)            
        y_t = self.model(o_t)

        mu_t = torch.zeros_like(y_t)
        mu_t[:, l_t] = -1e6
        pi_t = self.softmax(y_t + mu_t)
        return pi_t
    


if __name__ == "__main__":
    from environment import TSPEnv
    import time

    # Inizializza pygame prima di tutto
    pygame.init()

    n_nodes = 12
    agent = MLPAgent(n_nodes, hidden_dim=256)

    # Crea l'ambiente con render_mode="human" per vedere la finestra
    env = TSPEnv(n_nodes, render_mode="human")

    checkpoint = "policy.pt"  # put None if you want the agent to act randomly
    if checkpoint is not None:
        agent.load_state_dict(torch.load(checkpoint))

    done = False
    state = env.reset()

    # Prima render per aprire la finestra subito
    env.render("human")

    while not done:
        # Gestisci gli eventi pygame per mantenere la finestra responsiva
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        probs: torch.Tensor = agent(state)
        probs_str = ", ".join([f"{x:.2f}" for x in probs.view(-1).tolist()])
        print(f"The probabilities for this action are: [{probs_str}]")
        action = torch.multinomial(probs, 1).item()
        state, reward, done, _ = env.step(action)
        env.render("human")  # passa esplicitamente "human"
        time.sleep(0.5)  # pausa per vedere ogni step

    print(f"The order of visit for nodes is {env.visited}")
    print(f"The final reward is {reward}, meaning that the total length of the tour is {-reward}")

    # Mantieni la finestra aperta alla fine finché l'utente non la chiude
    print("Premi CTRL+C o chiudi la finestra per uscire.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    env.close()