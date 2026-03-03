import pygame
from vrp_agent import MLPAgentVRP
from vrp_environment import VRPEnv
import torch
from vrp_baseline_policies import greedy_policy_vrp, random_policy_vrp
from tqdm import tqdm

class REINFORCE_VRP:
    """
    Implementazione REINFORCE per il Vehicle Routing Problem (VRP).
    
    Come nel TSP, l'agente interagisce con l'ambiente, salva le traiettorie (stati, log-probabilities, reward)
    e aggiorna la rete usando la baseline (greedy policy).
    """
    def __init__(self, n_nodes, vehicle_capacity=1.0):
        self.env = VRPEnv(n_nodes, vehicle_capacity)
        self.agent = MLPAgentVRP(n_nodes, hidden_dim=256)
        self.buffer = []
        self.scores = []

        # Iperparametri
        self.training_epochs = 5000  # si può modificare
        self.episodes_per_epoch = 512
        self.gamma = 1.0
        learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=learning_rate)

    def collect_one_episode(self):
        """
        Interazione agente-ambiente per un episodio.
        Salva le traiettorie (s_t, log(pi(a_t|s_t)), r_t)
        """
        state = self.env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            action_probs = self.agent(state)
            action = torch.multinomial(action_probs, 1).item()
            state, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            log_probs.append(torch.log(action_probs[:, action]))

        self.buffer.append((rewards, log_probs, state))

    def update(self):
        """
        Aggiorna i pesi dell'agente usando REINFORCE con baseline greedy.
        """
        loss = 0
        for rewards, log_probs, state in self.buffer:
            # Baseline: soluzione greedy per la stessa istanza
            greedy_len, _ = greedy_policy_vrp(state["nodes"], state["demands"], self.env.vehicle_capacity)

            # Calcolo ritorni cumulativi
            returns = []
            G = 0
            for r in rewards[::-1]:
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)

            log_probs = torch.stack(log_probs)
            score = G + greedy_len  # uso baseline
            loss -= torch.sum(log_probs) * score
            self.scores.append(score.item())

        self.optimizer.zero_grad()
        (loss / len(self.buffer)).backward()
        self.optimizer.step()
        self.buffer = []

    def train(self):
        """
        Loop di addestramento REINFORCE
        """
        with tqdm(total=self.training_epochs, position=0, desc="Epoch") as pbar:
            for epoch_num in range(self.training_epochs):
                for _ in tqdm(range(self.episodes_per_epoch), desc="Episode", position=1, leave=False):
                    self.collect_one_episode()
                self.update()
                pbar.update(1)

                # Salvataggio statistiche
                with open("scores_vrp.txt", "w") as file:
                    file.write("\n".join(str(x) for x in self.scores))

                last_score = sum(self.scores[-self.episodes_per_epoch:])
                mean_score = last_score / self.episodes_per_epoch
                tqdm.write(f"Epoch {epoch_num + 1}, score: {mean_score:.2f}")

                if epoch_num % 100 == 0 or epoch_num == self.training_epochs - 1:
                    self.test()
                    # Salva screenshot dell'ambiente
                    pygame.image.save(self.env.screen, f"epoch_{epoch_num}.jpeg")
                    self.env.close()
                    # Salva checkpoint agent
                    torch.save(self.agent.state_dict(), "policy_vrp.pt")

    def test(self):
        """
        Esegue un episodio di test con l'agente addestrato
        """
        state = self.env.reset()
        done = False
        while not done:
            action_probs = self.agent(state)
            action = torch.multinomial(action_probs, 1).item()
            state, _, done, _ = self.env.step(action)
            self.env.render()


if __name__ == "__main__":
    reinforce = REINFORCE_VRP(n_nodes=20, vehicle_capacity=1.0)
    reinforce.train()

    from matplotlib import pyplot as plt
    plt.plot(reinforce.scores)
    plt.savefig("reinforce_vrp.png")
    plt.show()