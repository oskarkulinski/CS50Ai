"""
Tic Tac Toe Player
"""

import math
import copy
import random

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """

    x = 0
    o = 0
    for row in range(3):
        for column in range(3):
            if board[row][column] == X:
                x += 1
            elif board[row][column] == O:
                o += 1
    if o >= x:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    moves = set()
    for row in range(3):
        for column in range(3):
            if board[row][column] == EMPTY:
                moves.add((row, column))
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action is None:
        return board
    after = copy.deepcopy(board)
    after[action[0]][action[1]] = player(board)
    return after


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """

    if utility(board) == 1:
        return X
    elif utility(board) == -1:
        return O
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if utility(board) != 0:
        return True
    else:
        if EMPTY in board[0] + board[1] + board[2]:
            return False
        else:
            return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    for row in range(3):
        if board[row][0] == board[row][1] and board[row][0] == board[row][2]:
            if board[row][0] == X:
                return 1
            else:
                return 0
    for column in range(3):
        if board[0][column] == board[1][column] and board[0][column] == board[2][column]:
            if board[0][column] == X:
                return 1
            elif board[0][column] == O:
                return -1

    if (board[0][0] == board[1][1] and board[1][1] == board[2][2]) or (
            board[0][2] == board[1][1] and board[1][1] == board[2][0]):
        if board[1][1] == X:
            return 1
        elif board[1][1] == O:
            return -1

    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    if player(board) == X:
        best = maximum(board)[1]
    else:
        best = minimum(board)[1]
    return best


def minimum(board):
    """
    Helper function, recursively calls maximum
    """

    if terminal(board):
        return utility(board), None
    else:
        score = 5
        for action in actions(board):
            temp = maximum(result(board, action))
            if int(temp[0]) < score or (random.randint(0, 2) == 0  and int(temp[0] <= score)):
                best = action
                score = int(temp[0])
        return score, best


def maximum(board):
    """
    Helper function, recursively calls minimum
    """
    if terminal(board):
        return utility(board), None
    else:
        score = -5
        for action in actions(board):
            temp = minimum(result(board, action))
            if int(temp[0]) > score or (random.randint(0, 2) == 0 and int(temp[0] >= score)):
                best = action
                score = int(temp[0])
        return score, best
