import numpy as np
import random
from copy import deepcopy
class TicTacToeSingle:
    def __init__(self):
        self.state_size = 9
        self.action_size = 9
        self.Turn = 1
        self.PlayerA = 1
        self.PlayerB = -1
        self.Winner = 0
        self.GameOver = False
        self.Draw = 0
        self.Board = np.zeros(9)
        self.WinCase = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        self.ActionBoard = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]


    def reset(self):
        self.Turn = 1
        self.PlayerA = 1
        self.PlayerB = -1
        self.Winner = 0
        self.GameOver = False
        self.Board = np.zeros(9)
        return self.Board

    def __checkFull__(self,board):
        check_count = 0
        for i in range(board.__len__()):
            if board[i] != 0: check_count += 1
        if check_count == 9:
            return True
        return False

    def __checkWin__(self,board):
        for i in range(self.WinCase.__len__()):
            a_cnt = 0 # a -> +1
            b_cnt = 0 # b -> -1
            for j in range(self.WinCase[i].__len__()):
                if board[self.WinCase[i][j]] == 1:
                    a_cnt += 1
                if board[self.WinCase[i][j]] == -1:
                    b_cnt += 1

            if a_cnt == 3 :
                return 1
            if b_cnt == 3 :
                return -1

        return 0

    def __actionAvail__(self, board, action):
        if board[action] == 0:
            return True
        else:
            return False

    def __changeBoardTurn__(self,Board):
        for i in range(Board.__len__()):
            Board[i] *= -1

    def __getTurnChangedBoard__(self,Board):
        for i in range(Board.__len__()):
            Board[i] *= -1
        return Board

    def __getCopiedBoard(self):
        board = np.zeros(self.Board.__len__())
        for i in range(self.Board.__len__()):
            board[i] = self.Board[i]
        return board

    def __printParmBoard__(self,board,str):
        print("-------------------------------------------------------")
        print("Board > " , str)
        _cnt = 0
        for i in range(board.__len__()):
            _end = ' '
            _stone = ''
            if _cnt == 2:
                _end = '\n'
                _cnt = 0
            else:
                _cnt += 1
                _end = ' '

            if board[i] == 1:_stone = 'O'
            elif board[i] == -1:_stone = 'X'
            else:_stone = '-'

            print(_stone,end =_end)
        print("-------------------------------------------------------")

    def __printBoard__(self):
        _cnt = 0
        if self.Turn == self.PlayerA:
            print("> Current Player : O")
        elif self.Turn == self.PlayerB:
            print("> Current Player : O")

        for i in range(self.Board.__len__()):
            _end = ' '
            _stone = ''
            if _cnt == 2:
                _end = '\n'
                _cnt = 0
            else:
                _cnt += 1
                _end = ' '

            if self.Board[i] == 1:_stone = 'O'
            elif self.Board[i] == -1:_stone = 'X'
            else:_stone = '-'

            print(_stone,end =_end)


    def get_reward(self):
        reward = 0

        self.GameOver = False
        if self.__checkFull__(self.Board):
            self.GameOver = True

        origin_win_state = self.__checkWin__(self.Board)

        if origin_win_state == 1 :
            self.GameOver = True
            reward = 100 # win
        elif origin_win_state == -1:
            self.GameOver = True
            reward = -100

        if self.GameOver:
            #self.__printBoard__()
            return reward

        return reward

    def __step__(self,action):
        # Always 1 is The Agent
        self.Board[action] = 1
        # Get Reward. this checks only final state
        reward = self.get_reward()
        # return <next_state,reward,done>
        return self.Board,reward,self.GameOver



if __name__ == "__main__":
    my_game = TicTacToeSingle()
    while not my_game.GameOver:
        action = input("1~9: ")
        action = int(action)
        my_game.__step__(action)


