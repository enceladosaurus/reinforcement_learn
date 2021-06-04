import numpy as np
import random




test_state = np.array([
    [3, 3, 5],
    [3, 0, 5],
    [0, 0, 5]])

class TicTacToe:

    def __init__(self, role: str, epsilon: float):
        
        self.moves = 0
        self.state = np.zeros((3,3), dtype=int)
        self.previous_state = None
        self.role = role
        self.epsilon = epsilon
        self.qtable = {}
        self.history = []
        self.total_reward = 0
        self.done = False
        if self.role == "X":
            self.value = 5
        else:
            self.value = 3

    def check_victory(self):
        print("Checking victory")
        if self.moves > 3: 
            state = np.where(self.state == self.value, 0, 1).flatten()
            print(state)
            if state[0] == state[1] == state[2] == 1:
                self.done = True
                return True
            elif state[0] == state[3] == state[6] == 1:
                self.done = True
                return True
            elif state[0] == state[4] == state[8] == 1:
                self.done = True
                return True
            elif state[1] == state[4] == state[7] == 1:
                self.done = True
                return True
            elif state[2] == state[5] == state[8] == 1:
                self.done = True
                return True
            elif state[2] == state[4] == state[6] == 1:
                self.done = True
                return True
            elif state[3] == state[4] == state[5] == 1:
                self.done = True
                return True
            elif state[6] == state[7] == state[8] == 1:
                self.done = True
                return True
            else:
                return False
    def act(self, state: np.array):
        self.moves += 1
        if random.uniform(0, 1) > self.epsilon:
            print("EXPLOITATION")
            current_state = state.flatten()
            key = str(current_state)
            try: 
                possible_actions = self.qtable[key]
            except KeyError:
                print("State not recorded, creating entry...")
                self.qtable[key] = {}
                current_state = state.flatten()
                possible_actions = [i for i, x in enumerate(current_state) if x == 0]
                print("Possible actions: ", possible_actions)
                if len(possible_actions) == 0:
                    self.done = True
                    print("GAME OVER")
                    return
                for i in possible_actions:
                    self.qtable[key][i] = random.uniform(0, 10)
                best_action = possible_actions[0]
            for k in self.qtable[key].keys():
                if self.qtable[key][k] > best_action:
                    best_action = k
            new_state = current_state.copy()
            new_state[best_action] = self.value

            if self.check_victory():
                self.total_reward += 100
                reward = 100
                self.done = True
            else:
                self.total_reward += -1
                reward = -1
            self._update_dict(key, best_action, reward)
            new_state = np.array(new_state).reshape(3, 3)
            self.previous_state = self.state
            self.state = new_state

        else:
            new_state, action = self._randomact(state)
            self.previous_state = self.state
            self.state = new_state
            if self.check_victory():
                self.total_reward += 100
                reward = 100
                self.done = True
            else:
                self.total_reward += -1
                reward = -1
            self._update_dict(str(state.flatten()), action, reward)
            
    def _update_dict(self, state, action, reward):
        self.history.append((state, action, reward))
        try:
            self.qtable[state][action] = reward
        except KeyError:
            self.qtable[state] = {}
            self.qtable[state][action] = reward

    def _randomact(self, state):
        current_state = state.flatten()
        possible_actions = [i for i, x in enumerate(current_state) if x == 0]
        if len(possible_actions) > 1:
            action = random.randint(0, len(possible_actions)-1)
        else:
            action = 0
        action = possible_actions[action]
        print("Action chosen:", action)
        new_state = current_state.copy()
        new_state[action] = self.value
        
        return np.array(new_state).reshape(3,3), action

test_agent = TicTacToe("X", 0.5)

while not test_agent.done:
    test_agent.act(test_agent.state)
    print(test_agent.state)