import copy
import cProfile

class Node:
    def __init__(self, board, player_color, depth=0, best_move=None):
        self.board = board
        self.player_color = player_color
        self.depth = depth
        self.best_move = best_move
        self.opponent_color = 'X' if player_color == 'O' else 'O'

    def make_move(self, move, color):
        board = copy.deepcopy(self.board)
        row, col = move
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        opponent_color = 'X' if color == 'O' else 'O'
        flipped_discs = []
        
        for d in directions:
            r, c = row + d[0], col + d[1]
            discs_to_flip = []
            
            while 0 <= r < 12 and 0 <= c < 12 and board[r][c] == opponent_color:
                discs_to_flip.append((r, c))
                r += d[0]
                c += d[1]
                
            if 0 <= r < 12 and 0 <= c < 12 and board[r][c] == color:
                flipped_discs.extend(discs_to_flip)

        board[row][col] = color
        for r, c in flipped_discs:
            board[r][c] = color

        # self.print_board()

        return board
    
    def undo_move(self, move, flipped_discs):
        row, col = move
        self.board[row][col] = '.'
        for r, c in flipped_discs:
            self.board[r][c] = 'X' if self.board[r][c] == 'O' else 'O'

    def is_valid_move(self, move, color):
        row, col = move
        if not (0 <= row < 12 and 0 <= col < 12 and self.board[row][col] == '.'):
            return False

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        opponent_color = 'X' if color == 'O' else 'O'

        for d in directions:
            r, c = row + d[0], col + d[1]
            if 0 <= r < 12 and 0 <= c < 12 and self.board[r][c] == opponent_color:
                while 0 <= r < 12 and 0 <= c < 12 and self.board[r][c] == opponent_color:
                    r += d[0]
                    c += d[1]
                if 0 <= r < 12 and 0 <= c < 12 and self.board[r][c] == color:
                    return True

        return False

    def get_available_moves(self, color):
        available_moves = []
        for row in range(12):
            for col in range(12):
                if self.board[row][col] == '.':
                    move = (row, col)
                    if self.is_valid_move(move, color):
                        available_moves.append(move)
        return available_moves

    def evaluate_board(self):
            def near_corner_check(board, player_color):
                opponent_color = 'X' if player_color == 'O' else 'O'
                my_tiles = opp_tiles = 0
                if board[0][0] == '.':
                    if board[0][1] == player_color:
                        my_tiles += 1
                    elif board[0][1] == opponent_color:
                        opp_tiles += 1
                    if board[1][1] == player_color:
                        my_tiles += 1
                    elif board[1][1] == opponent_color:
                        opp_tiles += 1
                    if board[1][0] == player_color:
                        my_tiles += 1
                    elif board[1][0] == opponent_color:
                        opp_tiles += 1
                if board[0][11] == '.':
                    if board[0][10] == player_color:
                        my_tiles += 1
                    elif board[0][10] == opponent_color:
                        opp_tiles += 1
                    if board[1][10] == player_color:
                        my_tiles += 1
                    elif board[1][10] == opponent_color:
                        opp_tiles += 1
                    if board[1][11] == player_color:
                        my_tiles += 1
                    elif board[1][11] == opponent_color:
                        opp_tiles += 1
                if board[11][0] == '.':
                    if board[11][1] == player_color:
                        my_tiles += 1
                    elif board[11][1] == opponent_color:
                        opp_tiles += 1
                    if board[10][1] == player_color:
                        my_tiles += 1
                    elif board[10][1] == opponent_color:
                        opp_tiles += 1
                    if board[10][0] == player_color:
                        my_tiles += 1
                    elif board[10][0] == opponent_color:
                        opp_tiles += 1
                if board[11][11] == '.':
                    if board[10][11] == player_color:
                        my_tiles += 1
                    elif board[10][11] == opponent_color:
                        opp_tiles += 1
                    if board[10][10] == player_color:
                        my_tiles += 1
                    elif board[10][10] == opponent_color:
                        opp_tiles += 1
                    if board[11][10] == player_color:
                        my_tiles += 1
                    elif board[11][10] == opponent_color:
                        opp_tiles += 1
                return -12.5 * (my_tiles - opp_tiles)
            
            def discs_coverage(my_tiles, opp_tiles):
                if my_tiles != opp_tiles:
                    return (100.0 * my_tiles)/(my_tiles + opp_tiles) if my_tiles > opp_tiles else -(100.0 * opp_tiles)/(my_tiles + opp_tiles)
                return 0

            def frontier_openness(my_frontier_tiles, opp_frontier_tiles):
                if my_frontier_tiles > opp_frontier_tiles:
                    return -(100.0 * my_frontier_tiles)/(my_frontier_tiles + opp_frontier_tiles)
                elif my_frontier_tiles < opp_frontier_tiles:
                    return (100.0 * opp_frontier_tiles)/(my_frontier_tiles + opp_frontier_tiles)
                return 0

            def corner_check(board, player_color):
                opponent_color = 'X' if player_color == 'O' else 'O'
                my_tiles = opp_tiles = 0
                if board[0][0] == player_color:
                    my_tiles += 1
                elif board[0][0] == opponent_color:
                    opp_tiles += 1
                if board[0][11] == player_color:
                    my_tiles += 1
                elif board[0][11] == opponent_color:
                    opp_tiles += 1
                if board[11][0] == player_color:
                    my_tiles += 1
                elif board[11][0] == opponent_color:
                    opp_tiles += 1
                if board[11][11] == player_color:
                    my_tiles += 1
                elif board[11][11] == opponent_color:
                    opp_tiles += 1
                return 25 * (my_tiles - opp_tiles)

            def mobility_check(player_color):
                opponent_color = 'X' if player_color == 'O' else 'O'
                my_tiles = len(self.get_available_moves(player_color))
                opp_tiles = len(self.get_available_moves(opponent_color))
                m = 0
                if my_tiles > opp_tiles:
                    m = (100.0 * my_tiles)/(my_tiles + opp_tiles)
                elif my_tiles < opp_tiles:
                    m = -(100.0 * opp_tiles)/(my_tiles + opp_tiles)
                return m
            
            player_color = self.player_color
            board = self.board
            if player_color == 'X':
                opponent_color = 'O'
            else:
                opponent_color = 'X'
            my_tiles = 0
            opp_tiles = 0
            my_frontier_tiles = 0
            opp_frontier_tiles = 0
            d = 0
            p = 0
            c = 0
            l = 0
            m = 0
            f = 0

            direct = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            V = [
                [20, -3, 11, -2,  8,  9,  9,  8,  -2, 11, -3, 20],
                [-3, -7, -4, -6, -5,  1,  1, -5,  -6, -4, -7, -3],
                [11, -4,  0,  0,  3,  5,  5,  3,   0,  0, -4, 11],
                [-2, -6,  0,  0, -2,  1,  1, -2,   0,  0, -6, -2],
                [8,  -5,  3, -2,  2,  2,  2,  2,  -2,  3, -5,  8],
                [9,   1,  5,  1,  2, -3, -3,  2,   1,  5,  1,  9],
                [9,   1,  5,  1,  2, -3, -3,  2,   1,  5,  1,  9],
                [8,  -5,  3, -2,  2,  2,  2,  2,  -2,  3, -5,  8],
                [-2, -6,  0,  0, -2,  1,  1, -2,   0,  0, -6, -2],
                [11, -4,  0,  0,  3,  5,  5,  3,   0,  0, -4, 11],
                [-3, -7, -4, -6, -5,  1,  1, -5,  -6, -4, -7, -3],
                [20, -3, 11, -2,  8,  9,  9,  8,  -2, 11, -3, 20],
            ]

            for i in range(12):
                for j in range(12):
                    if board[i][j] == player_color:
                        d += V[i][j]
                        my_tiles += 1
                    elif board[i][j] == opponent_color:
                        d -= V[i][j]
                        opp_tiles += 1
                    if board[i][j] != '.':
                        for k in range(8):
                            x = i + direct[k][0]
                            y = j + direct[k][1]
                            if 0 <= x < 12 and 0 <= y < 12 and board[x][y] == '.':
                                if board[i][j] == player_color:
                                    my_frontier_tiles += 1
                                else:
                                    opp_frontier_tiles += 1
                                break

            p = discs_coverage(my_tiles, opp_tiles)
            f = frontier_openness(my_frontier_tiles, opp_frontier_tiles)
            c = corner_check(board, player_color)
            l = near_corner_check(board, player_color)
            m = mobility_check(player_color)
            
            score = (10 * p) + (50 * c) + (25 * l) + (5 * f) + (10 * d) + (6 * m)
            return score

    def is_terminal(self):
        if self.depth == 0 or not self.get_available_moves(self.player_color):
            return True
        
    def print_board(self):
        for row in self.board:
            print(' '.join(row))
        print()

def alpha_beta_pruning(node):
    def alpha_beta(node, alpha, beta, maximizing_player):
        if node.is_terminal():
            return node.evaluate_board()

        if maximizing_player:
            value = -float('inf')
            for move in node.get_available_moves(node.player_color):
                new_board = node.make_move(move, node.player_color)
                value = max(value, alpha_beta(Node(new_board, node.opponent_color, node.depth - 1, move), alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float('inf')
            for move in node.get_available_moves(node.player_color):
                new_board = node.make_move(move, node.player_color)
                value = min(value, alpha_beta(Node(new_board, node.opponent_color, node.depth - 1, move), alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    for move in node.get_available_moves(node.player_color):
        new_board = node.make_move(move, node.player_color)
        new_node = Node(new_board, node.opponent_color, node.depth - 1, move)
        value = alpha_beta(new_node, alpha, beta, False)
        if value > alpha:
            alpha = value
            best_move = move
    return best_move

def main():
    player_color = ''
    player_time_remaining, opponent_time_remaining = 0, 0
    player_score, opponent_score = 0, 0
    board = [[0 for i in range(12)] for j in range(12)]
    with open('input.txt', 'r') as file:
        player_color = file.readline().rstrip('\n')
        player_time_remaining, opponent_time_remaining = map(float, file.readline().rstrip('\n').split())
        for i in range(12):
            row = file.readline().rstrip('\n').split()
            for j in range(12):
                if row[0][j] == 'X':
                    board[i][j] = 'X'
                elif row[0][j] == 'O':
                    board[i][j] = 'O'
                else:
                    board[i][j] = '.'
    if player_color == 'X':
        player_score += 1
    else:
        opponent_score += 1

    if player_time_remaining <= 1:
        initial_node = Node(board, player_color, 1)
    elif player_time_remaining <= 15:
        initial_node = Node(board, player_color, 2)
    elif player_time_remaining <= 60:
        initial_node = Node(board, player_color, 3)
    elif player_time_remaining <= 120:
        initial_node = Node(board, player_color, 4)
    else:
        initial_node = Node(board, player_color, 5)

    res = alpha_beta_pruning(initial_node)

    with open('output.txt', 'w') as file:
        file.write((str(chr(res[1] + 65)).lower() + str(res[0] + 1)))

if __name__ == '__main__':
    main()