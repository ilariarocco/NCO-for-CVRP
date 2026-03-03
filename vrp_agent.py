import torch
import torch.nn as nn
import numpy as np
import pygame


class MLPAgentVRP(nn.Module):
    """
    Agente RL per il Capacitated Vehicle Routing Problem (CVRP).

    L'agente osserva:
    - coordinate dei nodi
    - domanda residua dei clienti
    - capacità residua del veicolo
    - nodo corrente

    e produce una distribuzione di probabilità sulle azioni ammissibili.
    """

    def __init__(self, n_nodes: int, hidden_dim: int = 128):
        super().__init__()

        self.n_nodes = n_nodes

        # Dimensione input:
        # - nodi: 2 * n_nodes
        # - domanda: n_nodes
        # - capacità residua: 1
        # - nodo corrente: 2
        input_dim = 2 * n_nodes + n_nodes + 1 + 2

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_nodes)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state: dict) -> torch.Tensor:
        """
        Calcola le probabilità delle azioni dato lo stato dell'ambiente.
        """

        # =========================
        # Costruzione input
        # =========================

        nodes = torch.from_numpy(state["nodes"]).float()
        demands = torch.from_numpy(state["demands"]).float()
        capacity = torch.tensor([state["remaining_capacity"]]).float()

        current_node_idx = state["visited"][-1]
        current_node = nodes[:, current_node_idx]

        # input flat
        x = torch.cat([
            nodes.T.flatten(),   # coordinate
            demands,             # domanda residua
            capacity,            # capacità residua
            current_node         # nodo corrente
        ]).unsqueeze(0)

        logits = self.model(x)

        # =========================
        # MASCHERA AZIONI NON VALIDE
        # =========================

        mask = torch.zeros_like(logits)
    
        # clienti già serviti (domanda = 0, escluso il depot)
        served = (demands == 0)
        served[0] = False  # il depot è sempre visitabile
        mask[:, served] = -1e6

        # clienti con domanda > capacità residua
        infeasible = demands > capacity
        mask[:, infeasible] = -1e6

        if current_node_idx == 0:
            mask[:, 0] = -1e6

        probs = self.softmax(logits + mask)
        return probs


if __name__ == "__main__":
    from vrp_environment import VRPEnv
    import time

    # Inizializza pygame prima di tutto
    pygame.init()

    env = VRPEnv(n_nodes=20, render_mode="human")
    agent = MLPAgentVRP(n_nodes=20)

    state = env.reset()
    done = False

    # Prima render per aprire la finestra subito
    env.render()

    while not done:
        # Gestisci gli eventi pygame per mantenere la finestra responsiva
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        probs = agent(state)
        action = torch.multinomial(probs, 1).item()
        print("Azione scelta:", action, flush=True)
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.6)

    print("Percorso:", env.visited)

    # Mantieni la finestra aperta alla fine finché l'utente non la chiude
    print("Premi CTRL+C o chiudi la finestra per uscire.")
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    env.close()