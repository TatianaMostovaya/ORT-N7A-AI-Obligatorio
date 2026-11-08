from abc import ABC, abstractmethod
from board import Board

class Agent(ABC):

    @abstractmethod
    def next_action(self, obs):
        return NotImplementedError
    
    @abstractmethod
    def heuristic_utility(self, board: Board):
        return NotImplementedError



class InputAgent(Agent):
    """Control manual: ingresa el índice de columna y presiona Enter.

    Se aceptan valores entre 0 y el ancho del tablero menos uno; las entradas
    no numéricas o fuera de rango se vuelven a pedir.
    """

    def next_action(self, obs):
        while True:
            try:
                input_value = int(input())
                if not 0 <= input_value < obs.length:
                    print(f"Value out of range. Enter a number from 0 to {obs.length - 1}.")
                    continue
                return input_value
            except ValueError:
                print("Please insert a number.")

    def heuristic_utility(self, board: Board):
        return 0
    
        