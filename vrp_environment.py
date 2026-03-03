import gymnasium as gym
import numpy as np
import pygame
import math


class VRPEnv(gym.Env):
    """
    Ambiente per il Capacitated Vehicle Routing Problem (CVRP).

    - Nodo 0 = depot (deposito)
    - Nodi 1..N-1 = clienti
    - Ogni cliente ha una domanda
    - Il veicolo ha una capacità massima
    - Quando la capacità non è sufficiente, il veicolo deve tornare al depot
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(
        self,
        n_nodes,
        vehicle_capacity=1.0,
        render_mode="human",
        width=1000,
        height=600,
        seed=None
    ):
        super().__init__()

        self.n_nodes = n_nodes              # include il depot
        self.vehicle_capacity = vehicle_capacity
        self.render_mode = render_mode
        self.width = width
        self.height = height

        self.seed(seed)

        # Azioni: scegliere il prossimo nodo (incluso il depot)
        self.action_space = gym.spaces.Discrete(n_nodes)

        # Stato dell'ambiente
        self.observation_space = gym.spaces.Dict({
            "nodes": gym.spaces.Box(low=0, high=1, shape=(2, n_nodes)),
            "demands": gym.spaces.Box(low=0, high=1, shape=(n_nodes,)),
            "remaining_capacity": gym.spaces.Box(low=0, high=vehicle_capacity, shape=(1,)),
            "visited": gym.spaces.Sequence(gym.spaces.Discrete(n_nodes))
        })

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Genera una nuova istanza del VRP.
        """
        self.nodes = np.random.rand(2, self.n_nodes)

        self.demands = np.zeros(self.n_nodes)
        self.demands[1:] = np.random.uniform(0.05, 0.3, size=self.n_nodes - 1)

        self.remaining_capacity = self.vehicle_capacity
        self.visited = [0]

        return self._get_state()

    def step(self, action):
        """
        Esegue l'azione selezionata dall'agente.
        """
        reward = 0
        done = False

        previous_node = self.visited[-1]

        reward -= np.linalg.norm(
            self.nodes[:, previous_node] - self.nodes[:, action]
        )

        if action == 0:
            self.remaining_capacity = self.vehicle_capacity
        else:
            if self.demands[action] > self.remaining_capacity:
                reward -= 10
                done = True
                return self._get_state(), reward, done, {}

            self.remaining_capacity -= self.demands[action]
            self.demands[action] = 0

        self.visited.append(action)

        if np.all(self.demands[1:] == 0):
            reward -= np.linalg.norm(
                self.nodes[:, self.visited[-1]] - self.nodes[:, 0]
            )
            done = True

        return self._get_state(), reward, done, {}

    def _get_state(self):
        return {
            "nodes": self.nodes,
            "demands": self.demands,
            "remaining_capacity": self.remaining_capacity,
            "visited": self.visited
        }

    def render(self):
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.width, self.height)
            )

        self.screen.fill((255, 255, 255))

        raggio = 14  # raggio cerchi clienti (e metà lato quadrato depot)

        # Disegno percorso con frecce (prima dei nodi, così i nodi stanno sopra)
        for i in range(len(self.visited) - 1):
            a = self.visited[i]
            b = self.visited[i + 1]

            x1 = int(self.nodes[0, a] * self.width)
            y1 = int(self.nodes[1, a] * self.height)
            x2 = int(self.nodes[0, b] * self.width)
            y2 = int(self.nodes[1, b] * self.height)

            angle = math.atan2(y2 - y1, x2 - x1)

            # Accorcia la linea del raggio in partenza e in arrivo
            sx = int(x1 + raggio * math.cos(angle))
            sy = int(y1 + raggio * math.sin(angle))
            ex = int(x2 - raggio * math.cos(angle))
            ey = int(y2 - raggio * math.sin(angle))

            # Linea
            pygame.draw.line(self.screen, (0, 0, 0), (sx, sy), (ex, ey), 2)

            # Freccia posizionata al punto di arrivo (bordo del nodo)
            arrow_size = 10
            p1 = (ex, ey)
            p2 = (
                int(ex - arrow_size * math.cos(angle - math.pi / 6)),
                int(ey - arrow_size * math.sin(angle - math.pi / 6))
            )
            p3 = (
                int(ex - arrow_size * math.cos(angle + math.pi / 6)),
                int(ey - arrow_size * math.sin(angle + math.pi / 6))
            )
            pygame.draw.polygon(self.screen, (0, 0, 0), [p1, p2, p3])

        # Disegno nodi (sopra le frecce)
        font = pygame.font.SysFont(None, 22)
        for i in range(self.n_nodes):
            x = int(self.nodes[0, i] * self.width)
            y = int(self.nodes[1, i] * self.height)

            if i == 0:  # depot
                pygame.draw.rect(self.screen, (200, 0, 0), (x - raggio, y - raggio, raggio * 2, raggio * 2))
                label = font.render("D", True, (255, 255, 255))
            else:  # clienti
                pygame.draw.circle(self.screen, (0, 0, 255), (x, y), raggio)
                label = font.render(str(i), True, (255, 255, 255))

            rect = label.get_rect(center=(x, y))
            self.screen.blit(label, rect)

        pygame.display.flip()

    def close(self):
        if hasattr(self, "screen"):
            pygame.quit()
            del self.screen


# -------------------------------
# TEST DELL'AMBIENTE
# -------------------------------
if __name__ == "__main__":
    import time

    pygame.init()

    env = VRPEnv(n_nodes=20, vehicle_capacity=1.0)

    env.screen = pygame.display.set_mode(
        (env.width, env.height)
    )
    pygame.display.set_caption("VRP Environment")

    state = env.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.6)

    env.close()
    print("Percorso finale:", env.visited, flush=True)
    print("Reward finale:", reward, flush=True)