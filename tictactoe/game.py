import numpy as np
import random
import json
import os

class TicTacToe:

    def __init__(self, X, O, mode: str, state = np.zeros(9, dtype=int)):
        self.state = state
        self.X = X
        self.O = O
        self.mode = mode
        self.done = False

    def check_victory(self, player):

        print("Checking victory for:", player.value)  
        state = np.where(self.state == player.value, 1, 0)
        print(state)
        if state[0] == state[1] == state[2] == 1:
            return True
        elif state[0] == state[3] == state[6] == 1:
            return True
        elif state[0] == state[4] == state[8] == 1:
            return True
        elif state[1] == state[4] == state[7] == 1:
            return True
        elif state[2] == state[5] == state[8] == 1:
            return True
        elif state[2] == state[4] == state[6] == 1:
            return True
        elif state[3] == state[4] == state[5] == 1:
            return True
        elif state[6] == state[7] == state[8] == 1:
            return True
        else:
            return False
    
    def agent_turn(self, agent):
        if self.check_state():
            current_state = self.state
            new_state, action = agent.act(self.state)
            self.state = new_state
            print(self.state.reshape(3, 3))
            if self.check_victory(agent):
                print(f"VICTORY FOR {agent.role}")
                reward = 100
                agent.add_history(self.state, action, reward)
                agent.victory = True
                self.done = True
            else:
                reward = -1
                agent.add_history(current_state, action, reward)
    
    def human_turn(self, human):

        if self.check_state():

            if self.check_victory(human):
                print(f"VICTORY FOR {human.role}")
                human.victory = True
                self.done = True
            else:
                print("CURRENT STATE:")
                print(self.state.reshape(3, 3))
                # Show in X and O
                print("What is the index of your move?")
                action = input()
                new_state = self.state.copy()
                assert(new_state[int(action)] == 0), "STOP CHEATING!"
                new_state[int(action)] = human.value
                self.state = new_state

    def check_state(self) -> bool:
        """
        Check if the board is full (end-condition for the game).

        Returns
        -------
        bool: True if game continues, False if game ends
        """
        state = np.where(self.state == 0, 1, 0)
        print("CHECKING STATE: ", self.state)
        if sum(state) == 0:
            self.done = True
            print("GAME OVER")
            return False
        elif self.X.victory or self.O.victory:
            return False
        else:
            return True

    def play(self, first_player, second_player):
        if self.mode == "train":
            while not self.done:
                self.agent_turn(first_player)
                self.agent_turn(second_player)
            first_player.update()
            second_player.update()
        else:
            while not self.done:
                self.agent_turn(first_player)
                self.human_turn(second_player)
            first_player.update()
            first_player.save()

    def reset(self):
        self.done = False
        self.state = np.zeros(9, dtype=int)
        self.X.victory = False
        self.O.victory = False
    
    def train(self, n_games: 100):
        X_victories = 0
        O_victories = 0
        draws = 0

        for i in range(n_games):
            print(f"Beginning game #{i}")
            if random.uniform(0, 1) > 0.5:
                print("X goes first")
                first_player = self.X
                second_player = self.O
            else:
                print("O goes first")
                first_player = self.O
                second_player = self.X
            self.play(first_player, second_player)
            if self.X.victory:
                X_victories += 1
            elif self.O.victory:
                O_victories += 1
            else:
                draws += 1
            first_player.save()
            second_player.save()
            self.reset()
        
        print(f"X victories: {X_victories}")
        print(f"O victories: {O_victories}")
        print(f"Draws: {draws}")

class Human:

    def __init__(self, role: str) -> None:
        self.role = role
        if self.role == "X":
            self.value = 5
        else:
            self.value = 3
        
        self.victory = False


class Agent:

    def __init__(
        self, role: str, epsilon: float, 
        discount: float, learning_rate: float):
        
        self.victory = False
        self.role = role
        self.epsilon = epsilon
        self.lr = learning_rate
        self.history = []
        self.total_reward = 0
        self.discount = discount
        if self.role == "X":
            self.value = 5
        else:
            self.value = 3

        model_path = f"./tictactoe/models/model{self.role}.json"

        if os.path.exists(model_path):
            print("Previous model found. Loading....")
            self.load_model()
        else:
            print("No model found. Creating blank Q-table")
            self.qtable = {}
    
    def load_model(self):

        with open(f"./tictactoe/models/model{self.role}.json", "r") as f:
            self.qtable = json.load(f)

    def act(self, state: np.array):
        if random.uniform(0, 1) > self.epsilon:
            print("EXPLOITATION")
            current_state = state
            key = str(current_state)
            possible_actions = [i for i, x in enumerate(current_state) if x == 0]
            print("Possible actions: ", possible_actions)   
            if key in self.qtable.keys(): 
                actions = self.qtable[key]
                max_value = -100
                best_action = None
                for action in actions:
                    if self.qtable[key][action] > max_value:
                        best_action = action
                        max_value = self.qtable[key][action]
            else:
                print("State not recorded, creating entry...")
                self.qtable[key] = {}           
                for i in possible_actions:
                    self.qtable[key][i] = random.uniform(0, 10)
                best_action = possible_actions[0]
                for action in self.qtable[key].keys():
                    if self.qtable[key][action] > best_action:
                        best_action = action
            new_state = current_state.copy()
            new_state[int(best_action)] = self.value

            return new_state, best_action

        else:
            print("EXPLORATION")
            new_state, action = self._randomact(state)

            return new_state, action 
            
    def add_history(self, state: np.array, action: int, reward: float):
        self.history.append((str(state), action, reward))

    def update(self):
        history = self.history
        history.reverse()
        for i in range(len(history)):
            state, action, reward = history[i]
            if i == 0:
                self._update_qtable(state, action, reward)
            else:
                future_state = history[i-1][0]
                max_value = max(list(self.qtable[future_state].values()))
            
                q_value = reward + self.discount * (max_value)
                self._update_qtable(state, action, q_value)
       
    def _update_qtable(self, state: str, action: int, reward: float):

        if state in self.qtable.keys():
            self.qtable[state][action] = reward
        
        else:
            self.qtable[state] = {}
            self.qtable[state][action] = reward

    def _randomact(self, state):
        current_state = state
        possible_actions = [i for i, x in enumerate(current_state) if x == 0]
        if len(possible_actions) > 1:
            action = random.randint(0, len(possible_actions)-1)
        else:
            action = 0
        action = possible_actions[action]
        print("Action chosen:", action)
        new_state = current_state.copy()
        new_state[action] = self.value
        
        return new_state, action
    
    def save(self):
        with open(f"./tictactoe/models/model{self.role}.json", "w") as f:
            json.dump(self.qtable, f)


agentX = Agent("X", 0.5, 0.99, 0.99)
#agentO = Agent("O", 0.5, 0.99, 0.99)
agentO = Human("O")
game = TicTacToe(agentX, agentO, mode="play")
#game.train(n_games = 100)
game.play(agentX, agentO)

# Epsilon decay
# Replace prints and shit with logging
# Document your damn code