"""Game master for Yavalath.  Designed for AI implementations.  Initially 2-player, eventually 3-player.

You probably don't want to use this code... it's just not worth it.  Maybe someday.

== Board representation 9borrowed from here: http://www.stephen.com/sue/sue_man.txt)

Give each row a letter from 'a' thru 'i', starting at the top.  Then number the
spaces in each row starting at the left.  This gives you a unique letter-number
combination for each space on the board.

a     1 2 3 4 5
b    1 2 3 4 5 6
c   1 2 3 4 5 6 7
d  1 2 3 4 5 6 7 8
e 1 2 3 4 5 6 7 8 9
f  1 2 3 4 5 6 7 8
g   1 2 3 4 5 6 7
h    1 2 3 4 5 6
i     1 2 3 4 5

To show the coordinates and the stones in a game of SUSAN, I use a diagram like
this.

       1 2 3 4 5
    a . . . . . 6
   b . . . . . . 7
  c . . . . . . . 8
 d . . . . . . . . 9
e . . . . . . . . .
 f . . . . . . . . 9
  g . . . . . . . 8
   h . . . . . . 7
    i . . . . . 6
       1 2 3 4 5

The letters label rows from the left.  The upper set of numbers labels
diagonals down to the left.  The lower set of numbers labels diagonals up to
the left.  Use the upper set of numbers on the upper half of the board and the
lower set of numbers on the lower half of the board.  The two sets of numbers
are the same for row 'e'.

== Game representation
A list of moves, in the order they occurred (also # of players)
A move is 2-character, such as "e1" or "g6"

== AI interface
Each AI is a callable, move, with this interface:

def move(game):
    # returns next move.

== Game engine rules
 - An illegal move immediately loses
    - Any occupied space is illegal
    - The first move by the second player may be "swap"
    - Any input not part of the board is illegal
 - Forming 4-in-a-row immediately wins
 - Forming 3-in-a-row immediately loses (NOTE: if also forming 4-in-row is a WIN, not loss
 - A full board, this is a draw

"""
from enum import Enum
import random
import itertools
import logging
from PIL import Image, ImageDraw, ImageFont
import numpy
import math
import pprint
logger = logging.Logger("yavalath")

# http://www.cameronius.com/games/yavalath/
# http://www.nestorgames.com/docs/YavalathCo2.pdf

# http://tabtop.blogspot.com/2009/07/yavalath.html
# https://github.com/slahmar/ai-yavalath


class MoveResult(Enum):
    PLAYER_WINS = 1
    PLAYER_LOSES = 2
    DRAW = 3
    ILLEGAL_MOVE = 4
    CONTINUE_GAME = 5


class HexBoard:
    def __init__(self):
        letters = "abcdefghi"
        assert len(letters) % 2 == 1, "Boards must have an odd number of rows"
        mid_letter = letters[int(len(letters) / 2)]
        valid_moves = list()
        for letter in letters:
            for num in range(1, 10 - abs(ord(letter) - ord(mid_letter))):
                valid_moves.append("{}{}".format(letter, num))
        self.spaces = valid_moves
        self.letters = letters

    def next_space(self, space, axis, distance):
        assert axis in (-1, 0, 1), "Axis must be -1, 0, or 1"
        letter, num = list(space)
        rank_offset = axis*distance
        if axis == 0:
            file_offset = distance
        elif axis == 1:
            if distance > 0 and letter >= "e":
                file_offset = 0
            elif distance > 0 and letter < "e":
                file_offset = min(distance, ord("e") - ord(letter))
            elif distance < 0 and letter <= "e":
                file_offset = distance
            elif distance < 0 and letter > "e":
                file_offset = min(0, distance + (ord(letter) - ord("e")))
        elif axis == -1:
            if distance > 0 and letter <= "e":
                file_offset = 0
            elif distance > 0 and letter > "e":
                file_offset = min(distance, ord(letter) - ord("e"))
            elif distance < 0 and letter >= "e":
                file_offset = distance
            elif distance < 0 and letter < "e":
                file_offset = min(0, distance + (ord("e") - ord(letter)))

        result = "{}{}".format(chr(ord(letter)+rank_offset), int(num)+ file_offset)
        return result if result in self.spaces else None

class Render:
    def __init__(self, board, moves):
        self.board = board
        self.edge_spaces = 5 # Maybe should be part of the board
        self.moves = moves
        if self.moves[1] == "swap":
            self.moves[1] = moves[0]
            self.moves[0] = None
        self.X_moves = list(self.moves[i] for i in range(0, len(moves), 2))
        self.Y_moves = list(self.moves[i] for i in range(1, len(moves), 2))
        self.image_size = 512
        self.border_size = 10
        self.y_border_size = (self.image_size - (self.image_size - 2*self.border_size) * math.sqrt(3) / 2) / 2
        self.gap_size = 5
        self.radius = (self.image_size - 2*self.border_size - self.gap_size*(self.edge_spaces*2 - 2)) / (self.edge_spaces*2 - 1) / 2

    def move_to_coord(self, move):
        rank, file = list(move)
        rank = ord(rank) - ord('a')
        file = int(file)
        y = rank
        x = 2*(file-1) + abs(rank-(self.edge_spaces-1))
        return (x, y)

    def render_ascii(self):
        grid_height = self.edge_spaces * 2 - 1
        grid_width = grid_height * 2 - 1 # Added room for whitespace
        rows = [[" " for i in range(grid_width)] for j in range(grid_height)]
        def render_moves(moves, token):
            for move in moves:
                x, y = self.move_to_coord(move)
                try:
                    rows[y][x] = token
                except Exception as ex:
                    logger.error("Failed to update (x,y): {}, {}, but len(rows)={} and len(rows[y])={}".format(x,y,len(rows),None if y >= len(rows) else len(rows[y])))
                    self.move_to_coord(move)

        render_moves(self.board.spaces, ".")
        render_moves(self.X_moves, "x")
        render_moves(self.Y_moves, "o")
        return "\n".join(["".join(row) for row in rows])

    def bounding_box(self, center, radius):
        bounding_box_min = center - radius * numpy.ones(2)
        bounding_box_max = center + radius * numpy.ones(2)
        return tuple(bounding_box_min) + tuple(bounding_box_max)

    def board_space_to_image_point(self, move):
        rank_count = self.edge_spaces * 2 - 1
        x, y = self.move_to_coord(move)
        x /= 2
        center_x = self.border_size + (self.image_size - 2*self.border_size) * x / rank_count + self.radius
        center_y = self.y_border_size + ((self.image_size - 2*self.border_size) * y / rank_count)*math.sqrt(3)/2 + self.radius
        return numpy.array([center_x, center_y])

    def render_moves(self, image, moves, R, color):
        draw = ImageDraw.Draw(image)
        for move in moves:
            draw.ellipse(self.bounding_box(self.board_space_to_image_point(move), R), fill=color, outline="black")

    def render_turns(self, image, all_moves, color):
        draw = ImageDraw.Draw(image)
        text_font = ImageFont.truetype('arial.ttf', 30)
        for turn, move in enumerate(all_moves):
            if move is None:
                continue
            center = self.board_space_to_image_point(move)
            box = self.bounding_box(center, self.radius)
            w, h = draw.textsize(str(turn), font=text_font)
            draw.text((center[0] - w/2, center[1] - h/2), str(turn), font=text_font, fill=color)

    def render_image(self, filename):
        image = Image.new("RGB", (self.image_size, self.image_size), "white")
        self.render_moves(image, self.board.spaces, self.radius/5, "gray")
        self.render_moves(image, self.X_moves, self.radius, "black")
        self.render_moves(image, self.Y_moves, self.radius, "white")
        self.render_turns(image, self.moves, "gray")
        image.save(filename)

    def render_spaces(self, filename):
        image = Image.new("RGB", (self.image_size, self.image_size), "white")
        self.render_moves(image, self.board.spaces, self.radius, "white")
        draw = ImageDraw.Draw(image)
        text_font = ImageFont.truetype('arial.ttf', 30)
        for space in self.board.spaces:
            center = self.board_space_to_image_point(space)
            box = self.bounding_box(center, self.radius)
            w, h = draw.textsize(space, font=text_font)
            draw.text((center[0] - w/2, center[1] - h/2), space, font=text_font, fill="black")
        image.save(filename)


def _connected_along_axis(board, move, other_moves, axis):
    """Returns now many connected pieces are on the named axis, that includes the 'move'
    A 4 or more result implies a win.  A 3 result implies a loss.
    """
    assert axis in (-1, 0, 1), "Axis must be -1, 0, or 1"

    pos_dist = 0
    while board.next_space(move, axis, pos_dist+1) in other_moves:
        pos_dist += 1
    neg_dist = 0
    while board.next_space(move, axis, neg_dist-1) in other_moves:
        neg_dist -= 1
    return pos_dist - neg_dist + 1


def judge_next_move(game_history, move):
    board = HexBoard()
    valid_moves = board.spaces + ["swap"]

    # Check for invalid moves
    if move in game_history or move not in valid_moves:
        return MoveResult.ILLEGAL_MOVE
    if move == "swap" and len(game_history) != 1:
        return MoveResult.ILLEGAL_MOVE

    # Extract just this player's moves
    players_moves = list([game_history[-i] for i in range(2, len(game_history)+1, 2)])
    if "swap" in players_moves:
        players_moves.append(game_history[0])

    # This move might make a longer line... find the longest line containing this piece
    max_distance = max([_connected_along_axis(board, move, players_moves, axis) for axis in [-1, 0, 1]])

    # Check for 4-in-a-row
    if max_distance >= 4:
        return MoveResult.PLAYER_WINS

    # Check for 3-in-a-row
    if max_distance == 3:
        return MoveResult.PLAYER_LOSES

    # Check for a draw
    if len(game_history) == 60:
        return MoveResult.DRAW

    # No end condition met, continue
    return MoveResult.CONTINUE_GAME


def play_two_player_yavalath(player1, player2):
    board = HexBoard()
    game_history = list()
    game_over = False
    while not game_over:
        for player in (player1, player2):
            move = player(board, game_history)
            move_result = judge_next_move(game_history, move)
            game_over = (move_result != MoveResult.CONTINUE_GAME)
            game_history.append(move)
            if game_over:
                break
    return game_history, move_result


def random_player(board, game_so_far):
    return random.choice(list(set(board.spaces) - set(game_so_far)))


def main():
    import pprint
    game, result = play_two_player_yavalath(random_player, random_player)
    #pprint.pprint(game)
    #pprint.pprint(result)

    # for turn in range(2, len(game)):
    #     renderer = Render(HexBoard(), game[0:turn])
    #     print("After Turn {}".format(turn))
    #     print("Game so far:{}".format(game[0:turn]))
    #     print(renderer.render_ascii())
    #     renderer.render_image("game_render_{}.png".format(turn))
    renderer = Render(HexBoard(), game)
    renderer.render_image("game_render.png")
    #renderer.render_spaces("game_board.png")
    print(result, "on turn", len(game)-1)

if __name__ == "__main__":
    main()