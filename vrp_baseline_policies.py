from vrp_agent import MLPAgentVRP
import torch
import random

@torch.no_grad()
def greedy_policy_vrp(nodes, demands, vehicle_capacity):
    """
    Policy greedy per il VRP.

    Nodo 0 = depot
    - Visita sempre il cliente più vicino che può essere servito
    - Se nessun cliente può essere servito, torna al depot
    - Restituisce: lunghezza totale e lista di nodi visitati
    """
    nodes = torch.from_numpy(nodes)
    demands = demands.copy()
    remaining_capacity = vehicle_capacity
    current_node = 0
    visited = [0]
    length = 0.0

    while any(demands[1:] > 0):
        feasible = [i for i in range(1, len(demands)) if demands[i] > 0 and demands[i] <= remaining_capacity]

        if len(feasible) == 0:
            # ritorno al depot
            length += torch.norm(nodes[:, current_node] - nodes[:, 0])
            current_node = 0
            remaining_capacity = vehicle_capacity
            visited.append(0)
            continue

        distances = [torch.norm(nodes[:, current_node] - nodes[:, i]) for i in feasible]
        idx = feasible[torch.argmin(torch.tensor(distances)).item()]

        length += torch.norm(nodes[:, current_node] - nodes[:, idx])
        current_node = idx
        remaining_capacity -= demands[idx]
        demands[idx] = 0
        visited.append(idx)

    if current_node != 0:
        length += torch.norm(nodes[:, current_node] - nodes[:, 0])
        visited.append(0)

    return length, visited


@torch.no_grad()
def random_policy_vrp(nodes, demands, vehicle_capacity):
    """
    Policy random per il VRP.

    Nodo 0 = depot
    - Sceglie clienti casualmente tra quelli servibili
    - Se nessun cliente può essere servito, torna al depot
    - Restituisce: lunghezza totale e lista di nodi visitati
    """
    nodes = torch.from_numpy(nodes)
    demands = demands.copy()
    remaining_capacity = vehicle_capacity
    current_node = 0
    visited = [0]
    length = 0.0

    while any(demands[1:] > 0):
        feasible = [i for i in range(1, len(demands)) if demands[i] > 0 and demands[i] <= remaining_capacity]

        if len(feasible) == 0:
            # ritorno al depot
            length += torch.norm(nodes[:, current_node] - nodes[:, 0])
            current_node = 0
            remaining_capacity = vehicle_capacity
            visited.append(0)
            continue

        idx = random.choice(feasible)
        length += torch.norm(nodes[:, current_node] - nodes[:, idx])
        current_node = idx
        remaining_capacity -= demands[idx]
        demands[idx] = 0
        visited.append(idx)

    if current_node != 0:
        length += torch.norm(nodes[:, current_node] - nodes[:, 0])
        visited.append(0)

    return length, visited


# -------------------------------
# TEST DELLE POLICY
# -------------------------------
if __name__ == "__main__":
    from vrp_environment import VRPEnv
    import time

    n_nodes = 6
    env = VRPEnv(n_nodes=n_nodes, vehicle_capacity=1.0)

    # Inizializzo l'agente anche se non lo uso per le policy
    agent = MLPAgentVRP(n_nodes)

    # RESET dell'ambiente prima delle policy
    state = env.reset()

    # -------------------------------
    # Calcolo lunghezza e percorso delle policy
    # -------------------------------
    greedy_len, greedy_path = greedy_policy_vrp(env.nodes, env.demands, env.vehicle_capacity)
    random_len, random_path = random_policy_vrp(env.nodes, env.demands, env.vehicle_capacity)

    # Stampa dei risultati
    print("Greedy policy length:", greedy_len, "Path:", greedy_path)
    print("Random policy length:", random_len, "Path:", random_path)

    # Simulazione episodio agente
    state = env.reset()
    done = False
    while not done:
        # Azioni casuali uniformi per test
        probs = torch.ones(n_nodes) / n_nodes
        action = torch.multinomial(probs, 1).item()
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.3)

    print("Agent policy length (simulata):", -reward)
    print("Percorso agente (simulato):", env.visited)