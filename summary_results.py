import pathlib
import collections
import subprocess
import argparse
import os
import sys
import shutil
import itertools

def summarize(dir, outdir="summaries"):

    total_wins = collections.defaultdict(int)
    total_losses = collections.defaultdict(int)
    wins = collections.defaultdict(int)
    losses = collections.defaultdict(int)
    fails = collections.defaultdict(int)
    names = set()
    all_games = list()

    win_table = collections.defaultdict(int)

    for f in pathlib.Path(dir).iterdir():
        try:
            if not f.is_file() or not f.name.endswith(".log") or f.name.startswith("battle_summary"):
                continue
            try:
                log_lines = open(f.as_posix(), 'r').readlines()
                final_game, last_line = log_lines[-2:]
            except:
                print("Exception reading {}".format(f.as_posix()))
                continue
            # Full Game:['b4', 'd4', 'h6', 'f3', 'f4', 'd2', 'c3', 'd5', 'd3', 'c1', 'f5', 'e3']
            # costello was last to move: MoveResult.PLAYER_WINS
            try:
                p1, _, p2 = f.name.split(" ")[0].replace(".log", "").split("_")
                names.add(p1)
                names.add(p2)
            except Exception as ex:
                raise RuntimeError("Failed to split:" + f.name) from ex

            player_name = last_line.split(" ")[0]
            opp_name = p1 if player_name == p2 else p2
            if last_line.find("PLAYER_WINS") >= 0:
                wins[player_name] += 1
                total_wins[player_name] += 1
                total_losses[opp_name] += 1
                win_table[(player_name, opp_name)] += 1
            if last_line.find("PLAYER_LOSES") >= 0:
                losses[player_name] += 1
                total_wins[opp_name] += 1
                total_losses[player_name] += 1
                win_table[(opp_name, player_name)] += 1
            if last_line.find("ILLEGAL_MOVE") >= 0:
                fails[player_name] += 1
                total_wins[opp_name] += 1
                total_losses[player_name] += 1
                win_table[(opp_name, player_name)] += 1
            if last_line.find("ERROR_WHILE_MOVING") >= 0:
                fails[player_name] += 1
                total_wins[opp_name] += 1
                total_losses[player_name] += 1
                win_table[(opp_name, player_name)] += 1
        except Exception as ex:
            print("Failed to input {}, skipping".format(f.name))
            continue

        # Generate some tex content
        tex_filename = "{}/{}".format(dir, f.name.replace(".log", "_moves.tex"))
        with open(tex_filename, "w") as move_file:
            header = log_lines[0]
            moves = [line for line in log_lines if line.find("Turn") == 0]
            judges = [line for line in log_lines if line.find("Judge") == 2]
            if len(moves) != len(moves):
                print("Failed to create tex for file: {}".format(str(f)))
            #Balogna was last to move: MoveResult.ERROR_WHILE_MOVING

            for one_move, one_judge in zip(moves, judges):
                # Turn 9 by Balogna moves at c1.  Time to decide: 3.005s
                parts = one_move.split(" ")
                turn = parts[1]
                player = parts[3]
                space = parts[6]
                time = parts[-1].rstrip()
                #  Judge says: MoveResult.CONTINUE_GAME
                result = one_judge.split('.')[-1].rstrip().replace("_", "\\_")
                print("{} & {} & {} & {} & {} \\\\".format(turn, player, space, result, time), file=move_file)

            if last_line.find("ERROR_WHILE_MOVING") >= 0:
                print("{} & {} & {} & {} & {} \\\\".format(len(moves), player_name, "NA", "ERROR_WHILE_MOVING", ""), file=move_file)

        if len(moves) > 0:
            all_games.append("{}_vs_{}".format(p1, p2))


    with open("{}/summary.txt".format(dir), 'w') as file_obj:
        print("Summary of {}".format(dir), file=file_obj)
        for name in sorted(names):
            print("{}: {} victories, {} win moves, {} loss moves, {} fail moves".format(name, total_wins[name], wins[name], losses[name], fails[name]), file=file_obj)

    with open("win_table.tex", 'w') as file_obj:
        tex_line = r"\begin{tabular}[b]{c|" + "c"*len(names) + r"}"
        print(tex_line, file=file_obj)
        tex_line = " & " + " & ".join(sorted(names)) + r" \\"
        print(tex_line, file=file_obj)
        print(r"\hline", file=file_obj)
        for player1 in sorted(names):
            tex_line = player1 + " & " + " & ".join([str(win_table[(player1, player2)]) for player2 in sorted(names)]) + r" \\"
            print(tex_line, file=file_obj)
        print(r"\end{tabular}", file=file_obj)

    with open("summary.tex".format(dir), 'w') as file_obj:
        for name in sorted(names):
            print(" & ".join([str(s) for s in [name, total_wins[name], total_losses[name], wins[name], losses[name], fails[name]]])+r" \\", file=file_obj)

    with open("all_games.tex", "w") as file_obj:
        for i, game in enumerate(all_games):
            #\onegame{battle_20170606_104039.225522/abbot_vs_costello}
            trailer = "&" if i % 2 == 0 else r"\\"
            trailer = ""
            print("\\onegame{" + "{}/{}".format(dir, game) + "} " + trailer, file=file_obj)

    pdflatex=r"C:/Program Files/MiKTeX 2.9/miktex/bin/x64/pdflatex.exe"
    subprocess.call([pdflatex, "-output-directory", dir, "battle_summary.tex"])
    shutil.copy("{}/battle_summary.pdf".format(dir), "{}/{}.summary.pdf".format(outdir, pathlib.Path(dir).name))



def parse_args():
    parser = argparse.ArgumentParser(help="Generate yavalath battle summaryreport", description="TODO: Description")
    parser.add_argument("--verbose", "-v", help="Verbose mode", default=False, action="store_true")
    return parser.parse_args()


def main():
    output_root = "official"
    summarize("{}/battle_20170627_101017.627979".format(output_root), "{}/summaries".format(output_root))
    return
    for child in pathlib.Path(output_root).iterdir():
        if child.name.startswith("battle_") and child.is_dir():
            summarize(child.as_posix(), output_root)
        summary_dir = "{}/summaries".format(output_root)
        os.makedirs(summary_dir, exist_ok=True)
        summarize(child.as_posix(), outdir=summary_dir)


if __name__ == "__main__":
    main()
    #
    #
    # out_dir = "summaries_test"
    # #summarize("battle_20170611_221939.395223", outdir=out_dir)
    # #sys.exit(0)
    # for path in pathlib.Path(".").iterdir():
    #     if path.name.find('battle_20') == 0 and path.is_dir():
    #         try:
    #             print("Summarizing:", path.name)
    #             summarize(path.name, outdir=out_dir)
    #         except Exception as ex:
    #             print("Something went wrong with:", path.name, ex)
    #             print("Exception:", ex)
