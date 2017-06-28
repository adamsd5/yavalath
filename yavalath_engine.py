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
import pathlib
import datetime
import time
import os
import summary_results
import multiprocessing
from functools import partial
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
    ERROR_WHILE_MOVING = 6


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
        if distance == 0:
            return space
        letter, num = list(space)
        rank_offset = axis * distance
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

        try:
            result = "{}{}".format(chr(ord(letter) + rank_offset), int(num) + file_offset)
        except:
            print("Failed for {}".format((space, axis, distance)))
            raise
        return result if result in self.spaces else None

    def next_space_in_dir(self, space, direction, distance=1):
        """A convenience function to get the next space in one of the 6 directions."""
        assert direction in range(6), "Directions are numbered 0..5"
        lookup = [(0, 1), (1, 1), (-1, -1), (0, -1), (1, -1), (-1, 1)]
        axis, distance_factor = lookup[direction]
        return self.next_space(space, axis, distance * distance_factor)

def moves_to_board_string(moves):
    r = Render(HexBoard, moves)
    return r.render_ascii()

class Render:
    def __init__(self, board, moves):
        self.board = board

        self.edge_spaces = 5  # Maybe should be part of the board
        self.moves = moves
        if len(self.moves) > 1 and self.moves[1] == "swap":
            self.moves[1] = moves[0]
            self.moves[0] = None
        self.X_moves = list(self.moves[i] for i in range(0, len(moves), 2))
        self.Y_moves = list(self.moves[i] for i in range(1, len(moves), 2))
        self.image_size = 512
        self.border_size = 10
        self.y_border_size = (self.image_size - (self.image_size - 2 * self.border_size) * math.sqrt(3) / 2) / 2
        self.gap_size = 5
        self.radius = (self.image_size - 2 * self.border_size - self.gap_size * (self.edge_spaces * 2 - 2)) / (
        self.edge_spaces * 2 - 1) / 2

    def move_to_coord(self, move):
        rank, file = list(move)
        rank = ord(rank) - ord('a')
        file = int(file)
        y = rank
        x = 2 * (file - 1) + abs(rank - (self.edge_spaces - 1))
        return (x, y)

    def render_ascii(self):
        grid_height = self.edge_spaces * 2 - 1
        grid_width = grid_height * 2 - 1  # Added room for whitespace
        rows = [[" " for i in range(grid_width)] for j in range(grid_height)]

        def render_moves(moves, token):
            for move in moves:
                x, y = self.move_to_coord(move)
                try:
                    rows[y][x] = token
                except Exception as ex:
                    logger.error(
                        "Failed to update (x,y): {}, {}, but len(rows)={} and len(rows[y])={}".format(x, y, len(rows),
                                                                                                      None if y >= len(
                                                                                                          rows) else len(
                                                                                                          rows[y])))
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
        center_x = self.border_size + (self.image_size - 2 * self.border_size) * x / rank_count + self.radius
        center_y = self.y_border_size + ((self.image_size - 2 * self.border_size) * y / rank_count) * math.sqrt(
            3) / 2 + self.radius
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
            w, h = draw.textsize(str(turn), font=text_font)
            draw.text((center[0] - w / 2, center[1] - h / 2), str(turn), font=text_font, fill=color)

    def render_image(self, filename):
        image = Image.new("RGB", (self.image_size, self.image_size), "white")
        self.render_moves(image, self.board.spaces, self.radius / 5, "gray")
        self.render_moves(image, self.Y_moves, self.radius, "black")
        self.render_moves(image, self.X_moves, self.radius, "white")
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
            draw.text((center[0] - w / 2, center[1] - h / 2), space, font=text_font, fill="black")
        image.save(filename)


def _connected_along_axis(board, move, other_moves, axis):
    """Returns now many connected pieces are on the named axis, that includes the 'move'
    A 4 or more result implies a win.  A 3 result implies a loss.
    """
    assert axis in (-1, 0, 1), "Axis must be -1, 0, or 1"

    pos_dist = 0
    while board.next_space(move, axis, pos_dist + 1) in other_moves:
        pos_dist += 1
    neg_dist = 0
    while board.next_space(move, axis, neg_dist - 1) in other_moves:
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
    players_moves = list([game_history[-i] for i in range(2, len(game_history) + 1, 2)])
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
        for player in [player1, player2]:
            exception = None
            try:
                move = player(board, game_history)
            except Exception as ex:
                exception = ex
                move = "ERROR"
            if exception is not None:
                move_result = MoveResult.ERROR_WHILE_MOVING
            else:
                move_result = judge_next_move(game_history, move)
            game_over = (move_result != MoveResult.CONTINUE_GAME)
            game_history.append(move)
            if game_over:
                break
    return game_history, move_result


def load_module(filename):
    import importlib.util
    import pathlib
    try:
        module_name = pathlib.Path(filename).name.replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        #print("loaded: {}".format(module.__name__))
        # 'get_player_names' and 'get_player' required
        return module.__name__, module.get_player_names, module.get_player
    except Exception as ex:
        print("Failed to load module {} from file {}".format(module_name, filename))
        print(ex)
    return None, None, None


def battle_round(p1name, p1maker, p2name, p2maker, output_dir):
    from players.blue.classifiers import NextMoveClassifier, SpaceProperies
    board = HexBoard()
    with open("{}/{}_vs_{}.log".format(output_dir, p1name, p2name), "w") as game_log:
        player1 = p1maker(p1name)
        player2 = p2maker(p2name)
        game_history = list()
        c = NextMoveClassifier([])
        game_over = False
        print("Yavalath, {} vs {}, starting at {}".format(p1name, p2name, datetime.datetime.now()), file=game_log)
        while not game_over:
            for player_name, player in [(p1name, player1), (p2name, player2)]:
                last_to_move = player_name
                try:
                    before = time.time()
                    move = player(game_history)
                    after = time.time()
                    turn_time = after - before

                    # Determine if it was a check
                    moves_by_property = NextMoveClassifier.get_board_properties(game_history + [move,])
                    token = 1 if len(game_history) % 2 == 0 else -1
                    # If there are wins for the player that just moved, it was a check
                    property = SpaceProperies.WHITE_WIN if token == 1 else SpaceProperies.BLACK_WIN
                    move_annotation = ""
                    if len(moves_by_property[property]) > 0:
                        move_annotation = "+"

                    print("Turn {} by {} moves at {}.  Time to decide: {:.03f}s".format(len(game_history), player_name, move+move_annotation, turn_time), file=game_log)
                    print("Turn {} by {} moves at {}.  Time to decide: {:.03f}s".format(len(game_history), player_name, move+move_annotation, turn_time))

                except Exception as ex:
                    print("Exception while taking a turn by {}".format(player_name), file=game_log)
                    print("Exception:{}".format(ex), file=game_log)
                    print("Traceback:{}".format(traceback.format_exc()), file=game_log)
                    game_over = True
                    move_result = MoveResult.ERROR_WHILE_MOVING
                    break
                move_result = judge_next_move(game_history, move)
                print("  Judge says: {}".format(move_result), file=game_log)
                game_log.flush()
                game_over = (move_result != MoveResult.CONTINUE_GAME)
                if move_result != MoveResult.ILLEGAL_MOVE:
                    game_history.append(move)
                if game_over:
                    break
        print("Full Game:{}".format(game_history), file=game_log)
        print("{} was last to move: {}".format(last_to_move, move_result), file=game_log)
        if len(game_history) > 0 and game_history[-1] not in board.spaces:
            moves_to_render = game_history[:-1]
        else:
            moves_to_render = game_history
        renderer = Render(HexBoard(), moves_to_render)
        renderer.render_image("{}/{}_vs_{}.png".format(output_dir, p1name, p2name))
        summary_results.summarize(output_dir, "{}/../summaries".format(output_dir))


MEMORY_LIMIT = 1024*1024*1024*4  # Everyone gets 4GB
#MEMORY_LIMIT = 1024*1024*100  # Everyone gets 4GB
#TODO: Better reporting on the exact exception that was raised when a player failed to move.

def player_in_process(child_conn, moudule_filename, player_name):
    """This is the target of multiprocessing.Process, it should be able to communicate with the parent"""
    import memorycontrol
    memorycontrol.set_memory_limit(MEMORY_LIMIT)

    try:
        module_name, get_player_names, get_player = load_module(moudule_filename)
        player = get_player(player_name)
    except MemoryError as ex:
        logging.error("MemoryError loading the module {} or get the player {}".format(module_name, player_name))
        child_conn.close()
        return
    except:
        logging.error("Failed to load the module {} or get the player {}".format(module_name, player_name))
        child_conn.close()
        return

    while True:
        try:
            move_so_far = child_conn.recv()
            exception = None
            next_move = None
            try:

                # TODO: Redirect stdout/stderr to file.
                # HERE IS WHERE THE PLAYER CODE IS CALLED
                next_move = player(move_so_far)

                print("Child Process, PID:{}, player:{}, move_so_far:{} returning {}".format(
                    os.getpid(), player_name, " ".join(move_so_far), next_move))
            except Exception as ex:
                exception = str(ex) + traceback.format_exc()
            child_conn.send((next_move, exception))
            if exception is not None:
                break
        except EOFError:
            break


def get_player_in_process(player_name, module_filename):
    print("Getting process-controlled player {} from module {}".format(player_name, module_filename))
    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(target=player_in_process, args=(child_conn, module_filename, player_name))
    process.start()
    child_conn.close()  # Close the child from the parent's end.

    def player_whisperer(moves_so_far):
        """This function runs in the main process game engine, to speak to the child process and ask for moves."""
        logger.info("Parent sending moves to child.")
        parent_conn.send(moves_so_far)
        logger.info("Parent sent moves to child... waiting on response.")
        next_move, exception = parent_conn.recv()
        logger.info("Parent got back: {}".format((next_move, exception)))

        if exception is not None:
            raise RuntimeError("Child process had an exception: {}".format(exception))
        return next_move

    return player_whisperer


def battle(module_paths):
    modules = [load_module(filename) for filename in module_paths]
    good_modules = list()
    for name, get_player_names, get_player in modules:
        if name is None or get_player_names is None or get_player is None:
            print("Missing something from module: {}".format((name, get_player_names, get_player)))
        else:
            print("Module joined: {}, with players {}".format(name, get_player_names()))
    modules = good_modules

    player_info = list()
    #random.shuffle(modules)
    for module_name, get_player_names, get_player in modules:
        for player_name in get_player_names():
            player_info.append((player_name, get_player))

    board = HexBoard()
    output_dir = "battle_{}".format(str(datetime.datetime.now()).replace(":", "").replace("-","").replace(" ", "_"))
    os.makedirs(output_dir)

    for p1info, p2info in itertools.combinations(player_info, r=2):
        p1_name, get_p1_fn = p1info
        p2_name, get_p2_fn = p2info
        print("{} vs. {}".format(p1_name, p2_name))
        battle_round(p1_name, get_p1_fn, p2_name, get_p2_fn, output_dir)
        print("{} vs. {}".format(p2_name, p1_name))
        battle_round(p2_name, get_p2_fn, p1_name, get_p1_fn, output_dir)

    summary_results.summarize(output_dir)

def battle_mp(module_paths, output_root="."):
    # player_info should become a list of pairs: (name, getter) such that getter(name) returns a player callable
    player_info = list()
    for filename in module_paths:
        moudle_name, get_player_names, get_player = load_module(filename)
        if moudle_name is None or get_player_names is None or get_player is None:
            print("Invalid module loaded {}: {}, {}, {}".format(filename, moudle_name, get_player_names, get_player))
            continue
        player_names = get_player_names()
        print("Module joined: {}, with players {}".format(moudle_name, player_names))
        for player_name in player_names:
            getter = partial(get_player_in_process, module_filename=filename)
            player_info.append((player_name, getter))

    #random.shuffle(player_info)
    output_dir = "{}/battle_{}".format(output_root, str(datetime.datetime.now()).replace(":", "").replace("-","").replace(" ", "_"))
    os.makedirs(output_dir)
    print("Battling in: {}".format(output_dir))

    # player_info should be a list of tuples: (name, getter) such that getter(name) returns a player callable
    for p1info, p2info in itertools.combinations(player_info, r=2):
        p1_name, get_p1_fn = p1info
        p2_name, get_p2_fn = p2info
        print("{} vs. {}".format(p1_name, p2_name))
        battle_round(p1_name, get_p1_fn, p2_name, get_p2_fn, output_dir)
        print("{} vs. {}".format(p2_name, p1_name))
        battle_round(p2_name, get_p2_fn, p1_name, get_p1_fn, output_dir)

    summary_dir = "{}/summaries".format(output_root)
    os.makedirs(summary_dir, exist_ok=True)
    summary_results.summarize(output_dir, outdir=summary_dir)


def main():
    official_run = False
    if official_run:
        module_names = [p.as_posix() for p in pathlib.Path("players").iterdir() if p.is_file()]
        while True:
            battle_mp(module_names, output_root="official")
    else:
        module_names = ["players/blue_player.py"]
        battle_mp(module_names, output_root="debug_battles")

if __name__ == "__main__":
    main()
