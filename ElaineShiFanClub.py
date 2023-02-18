from src.game_constants import RobotType, Direction, Team, TileState
from src.game_state import GameState, GameInfo
from src.player import Player
from src.map import TileInfo, RobotInfo
from collections import deque, OrderedDict
import random
import time


def coord_distance(coord1, coord2):
    return max(abs(coord1[0] - coord2[0]), abs(coord1[1] - coord2[1]))


def get_coords(tile):
    return (tile.row, tile.col)


class BotPlayer(Player):
    """
    Players will write a child class that implements (notably the play_turn method)
    """

    def __init__(self, team: Team):
        self.team = team
        self.explorer_goal = {}
        self.miner_goal = {}
        self.miner_charging = {}
        self.terraformer_charging = {}
        self.width = 0
        self.height = 0
        self.map = None
        self.game_state = None
        self.begin_time = 0
        self.print_warnings = False
        self.debug = False
        self.print_time = self.debug
        self.reachable_by_terraformers = None
        self.last_secure_mines_time = 0
        self.connectivity = None
        self.fog_map = None
        self.total_fog_count = 0
        return

    def compute_connectivity(self) -> int:
        self.connectivity = [[None for _ in range(self.width)] for _ in range(self.height)]
        num_blocks = 0
        for row in range(self.height):
            for col in range(self.width):
                if self.map[row][col] is not None:
                    if self.connectivity[row][col] is None:
                        num_blocks += 1
                        queue = deque([(row, col)])
                        self.connectivity[row][col] = num_blocks
                        while (queue):
                            r, c = queue.popleft()
                            for dir in Direction:
                                newRow, newCol = r + dir.value[0], c + dir.value[1]
                                if newRow < 0 or newRow >= self.height or newCol < 0 or newCol >= self.width:
                                    continue
                                if self.map[newRow][newCol] is None:
                                    continue
                                if self.connectivity[newRow][newCol] is not None:
                                    continue
                                self.connectivity[newRow][newCol] = num_blocks
                                queue.append((newRow, newCol))
        return num_blocks


    def compute_reachable_by_terraformers(self, my_terraformers):
        self.reachable_by_terraformers = [[None for _ in range(self.width)] for _ in range(self.height)]
        queue = deque([(terraformer, terraformer.row, terraformer.col) for terraformer in my_terraformers])
        for terraformer in my_terraformers:
            self.reachable_by_terraformers[terraformer.row][terraformer.col] = None
        while (queue):
            terraformer, queueRow, queueCol = queue.popleft()
            for newDir in Direction:
                newRow, newCol = queueRow + newDir.value[0], queueCol + newDir.value[1]
                if newRow < 0 or newRow >= self.height or newCol < 0 or newCol >= self.width:
                    continue
                tileState = self.map[newRow][newCol]
                if tileState is None:
                    continue
                # Check if its legal
                if (tileState.state == TileState.ILLEGAL or tileState.state == TileState.IMPASSABLE):
                    continue
                if (tileState.robot is not None):
                    continue
                if not self.reachable_by_terraformers[newRow][newCol]:
                    self.reachable_by_terraformers[newRow][newCol] = terraformer
                    queue.append((terraformer, newRow, newCol))


    def optimal_terraform(self, startRow: int, startCol: int, exclusions, checkCollisions=True) -> tuple[Direction,tuple[int,int]]:
        visited = set()
        queue = deque([(None, startRow, startCol, 0)])
        while (queue):
            dir, queueRow, queueCol, moves = queue.popleft()
            if (self.map[queueRow][queueCol] is not None and
                    (not checkCollisions or self.map[queueRow][queueCol].robot is None) and
                    self.map[queueRow][queueCol].state == TileState.TERRAFORMABLE and
                    self.map[queueRow][queueCol].terraform <= 0 and
                    (queueRow, queueCol) not in exclusions):
                return (dir, (queueRow, queueCol))
            for newDir in Direction:
                newRow, newCol = queueRow + \
                    newDir.value[0], queueCol + newDir.value[1]
                if newRow < 0 or newRow >= self.height or newCol < 0 or newCol >= self.width:
                    continue
                tileState = self.map[newRow][newCol]
                if tileState is None:
                    continue
                # Check if its legal
                if (tileState.state == TileState.ILLEGAL or tileState.state == TileState.IMPASSABLE):
                    continue
                if (checkCollisions and tileState.robot is not None):
                    continue
                if ((newRow, newCol) not in visited):
                    if dir == None:
                        queue.append((newDir, newRow, newCol, moves+1))
                    else:
                        queue.append((dir, newRow, newCol, moves+1))
                    visited.add((newRow, newCol))

        # If we reached this point, a path to the co-ordinates isn't possible
        return (None, (-1, -1))


    def optimal_explore(self, startRow: int, startCol: int, exclusions, checkCollisions=True):
        visited = set()
        queue = deque([(None, startRow, startCol, 0)])
        result = (None, (-1, -1), -1)
        while (queue):
            dir, queueRow, queueCol, moves = queue.popleft()
            if result[0] is not None and moves > result[2] + 2:  # only explore tiles at most 2 moves farther than the best
                break
            if (self.fog_map[queueRow][queueCol] > 0 and (queueRow, queueCol) not in exclusions):
                if result[0] is None or (self.fog_map[queueRow][queueCol] - self.fog_map[result[1][0]][result[1][1]]
                        >= moves - result[2]):
                    result = (dir, (queueRow, queueCol), moves)
            for newDir in Direction:
                newRow, newCol = queueRow + newDir.value[0], queueCol + newDir.value[1]
                if newRow < 0 or newRow >= self.height or newCol < 0 or newCol >= self.width:
                    continue
                tileState = self.map[newRow][newCol]
                if tileState is None:
                    continue
                # Check if its legal
                if (tileState.state == TileState.ILLEGAL or tileState.state == TileState.IMPASSABLE):
                    continue
                if (checkCollisions and tileState.robot is not None):
                    continue
                if ((newRow, newCol) not in visited):
                    if dir == None:
                        queue.append((newDir, newRow, newCol, moves + 1))
                    else:
                        queue.append((dir, newRow, newCol, moves + 1))
                    visited.add((newRow, newCol))
        return result


    def closest_ally_tile_to_spawn(self, startRow: int, startCol: int, ally_tile_set):
        visited = set()
        queue = deque([(startRow, startCol)])
        while (queue):
            queueRow, queueCol = queue.popleft()
            if ((queueRow, queueCol) in ally_tile_set):
                return (queueRow, queueCol)
            for newDir in Direction:
                newRow, newCol = queueRow + newDir.value[0], queueCol + newDir.value[1]
                if newRow < 0 or newRow >= self.height or newCol < 0 or newCol >= self.width:
                    continue
                tileState = self.map[newRow][newCol]
                if tileState is None:
                    continue
                # Check if its legal
                if (tileState.state == TileState.ILLEGAL or tileState.state == TileState.IMPASSABLE):
                    continue
                if ((newRow, newCol) not in visited):
                    queue.append((newRow, newCol))
                    visited.add((newRow, newCol))

        # If we reached this point, a path to the co-ordinates isn't possible
        return (None, None)

    def try_to_bomb_opponent_mining(self, robot):
        for dir in Direction:  # try to bomb opponent miner in one step if it's mining
            row = robot.row + dir.value[0]
            col = robot.col + dir.value[1]
            if row < 0 or row >= self.height or col < 0 or col >= self.width:
                continue
            if self.map[row][col] is None:  # fogged
                continue
            if self.map[row][col].robot is None or self.map[row][col].robot.team == self.team:
                continue
            if self.map[row][col].state != TileState.MINING:
                continue
            if self.map[row][col].robot.type != RobotType.MINER:
                continue
            if self.map[row][col].robot.battery < 20:  # they cannot mine without battery
                continue
            return dir
        return None

    def count_mine_terraform(self, row, col) -> int:
        terraform_count = 0
        for dir in Direction:
            r = row + dir.value[0]
            c = col + dir.value[1]
            if r < 0 or r >= self.height or c < 0 or c >= self.width:
                continue
            if self.map[r][c] is None:
                continue
            if self.map[r][c].terraform > 0:
                terraform_count += 1
            elif self.map[r][c].terraform < 0:
                terraform_count -= 1
        return terraform_count

    def count_mine_insecure(self, row, col) -> int:
        insecure_count = 0
        for dir in Direction:
            r = row + dir.value[0]
            c = col + dir.value[1]
            if r < 0 or r >= self.height or c < 0 or c >= self.width:
                continue
            if self.map[r][c] is None:
                continue
            if self.map[r][c].state == TileState.TERRAFORMABLE and self.map[r][c].terraform <= 0:
                insecure_count += (-self.map[r][c].terraform) + 1
        return insecure_count

    def count_fog(self, row, col) -> int:
        fogged_count = 0
        for dir in Direction:
            r = row + dir.value[0]
            c = col + dir.value[1]
            if r < 0 or r >= self.height or c < 0 or c >= self.width:
                continue
            if self.map[r][c] is None:
                fogged_count += 1
        return fogged_count

    def init_fog_map(self):
        self.fog_map = [[None for _ in range(self.width)] for _ in range(self.height)]
        for row in range(self.height):
            for col in range(self.width):
                self.fog_map[row][col] = self.count_fog(row, col)

    def move_robot(self, robot, dir) -> bool:
        rname = robot.name
        if self.game_state.can_move_robot(rname, dir):
            new_row = robot.row + dir.value[0]
            new_col = robot.col + dir.value[1]
            if self.map[new_row][new_col] is not None and self.map[new_row][new_col].robot is not None:
                # collision
                if self.map[new_row][new_col].robot.team == self.team:
                    if self.print_warnings:
                        print("Collision with ourselves:", robot, "to", new_row, new_col, "with",
                              self.map[new_row][new_col].robot)
                self.map[new_row][new_col].robot = None
                self.map[robot.row][robot.col].robot = None
            else:
                # move
                if self.map[new_row][new_col] is not None:
                    self.map[new_row][new_col].robot = self.map[robot.row][robot.col].robot
                self.map[robot.row][robot.col].robot = None
            robot.row = new_row
            robot.col = new_col
            robot.moved = True
            self.game_state.move_robot(rname, dir)
            return True
        return False


    def play_turn(self, game_state: GameState) -> None:
        if self.print_time:
            self.begin_time = time.time()
            print("Time left at the beginning:", game_state.get_time_left())
        if game_state.get_time_left() < 5:
            # no more than 5 seconds left, what shall we do?
            return
        # get info
        ginfo = game_state.get_info()

        # get turn/team info
        self.width, self.height = len(ginfo.map), len(ginfo.map[0])
        width, height = self.width, self.height
        self.map = ginfo.map
        self.game_state = game_state

        # print info about the game
        if self.debug:
            print(f"Turn {ginfo.turn}, team {ginfo.team}")
        # print("Map height", height)
        # print("Map width", width)

        # find un-occupied ally tile
        ally_tiles = []
        my_mines = []
        opponent_mines = []
        insecure_mines = OrderedDict()
        my_robots = []
        opponent_robots = []
        fogged_tiles = []
        # edge_tiles = {}
        # for i in range(9):
        #     edge_tiles[i] = []
        self.total_fog_count = 0
        self.init_fog_map()
        for row in range(height):
            for col in range(width):
                # get the tile at (row, col)
                tile = self.map[row][col]
                # skip fogged tiles
                if tile is not None:  # ignore fogged tiles
                    if tile.robot is not None:
                        if tile.robot.team == self.team: # my robot
                            my_robots += [tile.robot]
                            tile.robot.moved = False
                            tile.robot.acted = False
                        else:
                            opponent_robots += [tile.robot]
                    if tile.state == TileState.MINING:
                        terraform_count = 0
                        for dir in Direction:
                            r = row + dir.value[0]
                            c = col + dir.value[1]
                            if r < 0 or r >= height or c < 0 or c >= width:
                                continue
                            if self.map[r][c] is None:
                                continue
                            if self.map[r][c].terraform > 0:
                                terraform_count += 1
                            elif self.map[r][c].terraform < 0:
                                terraform_count -= 1
                        if terraform_count >= 0:
                            my_mines += [(tile.row, tile.col)]
                        else:
                            opponent_mines += [(tile.row, tile.col)]
                    # if tile.state in [TileState.TERRAFORMABLE, TileState.MINING]:
                    #     fogged_count = self.fog_map[row][col]
                    #     if fogged_count > 0:
                    #         edge_tiles[fogged_count] += [tile]
                else:
                    fogged_tiles += [(row, col)]
                    self.total_fog_count += 1

        my_explorers = []
        my_bombs = []
        my_miners = []
        my_terraformers = []
        for robot in my_robots:
            if robot.type == RobotType.EXPLORER:
                if robot.battery < 10:
                    my_bombs += [robot]
                else:
                    my_explorers += [robot]
            elif robot.type == RobotType.MINER:
                my_miners += [robot]
            elif robot.type == RobotType.TERRAFORMER:
                my_terraformers += [robot]

        my_explorer_names = set([robot.name for robot in my_explorers])
        dead_explorer_names = []
        for explorer_name, goal in self.explorer_goal.items():
            if explorer_name not in my_explorer_names:
                dead_explorer_names.append(explorer_name)
        for explorer_name in dead_explorer_names:
            self.explorer_goal.pop(explorer_name)

        if self.debug:
            print(f"My metal {game_state.get_metal()}")
        if self.print_time:
            print("Time after preprocessing:", time.time() - self.begin_time)


        kDistanceToTreatAsReachedGoal = 1
        # move explorers
        if len(fogged_tiles) > 0 and len(my_explorers) > 0:
            exclusions = set()
            for explorer in my_explorers:
                moved = False
                rname = explorer.name
                bomb = self.try_to_bomb_opponent_mining(explorer)
                if bomb is not None:
                    if self.fog_map[explorer.row][explorer.col] > 0 and game_state.can_robot_action(rname):
                        game_state.robot_action(rname)  # last explore before sacrificing
                    if self.move_robot(explorer, bomb):
                        continue
                    else:
                        if self.print_warnings:
                            print("Failed to bomb using explorer!")
                        continue
                if rname in self.explorer_goal.keys():
                    # try to approach goal
                    goal = self.explorer_goal[rname]
                    for goal_dir in Direction:
                        row = goal[0] + goal_dir.value[0]
                        col = goal[1] + goal_dir.value[1]
                        if row < 0 or row >= height or col < 0 or col >= width:
                            continue
                        if self.map[row][col] is None:  # fogged
                            continue
                        path = game_state.optimal_path(explorer.row, explorer.col, row, col, checkCollisions=True)
                        if path[0] is None:
                            continue
                        if self.move_robot(explorer, path[0]):
                            moved = True
                            if coord_distance((explorer.row, explorer.col), goal) <= kDistanceToTreatAsReachedGoal:
                                self.explorer_goal.pop(rname)
                            break
                temp_goal = None
                temp_fog = self.fog_map[explorer.row][explorer.col]
                if not explorer.moved:
                    # move anywhere
                    path = self.optimal_explore(explorer.row, explorer.col, exclusions)
                    if path[0] is not None:
                        if not (temp_fog == 0 or self.fog_map[path[1][0]][path[1][1]] - temp_fog >= path[2]):
                            if not explorer.acted and game_state.can_robot_action(rname):
                                game_state.robot_action(rname)
                                explorer.acted = True
                        if self.move_robot(explorer, path[0]):
                            exclusions.add(path[1])
                            temp_goal = path[1]
                            # also exclude nearby tiles?
                            # for dir in Direction:
                            #     exclusions.add((path[1][0] + dir.value[0], path[1][1] + dir.value[1]))
                    if not explorer.moved:
                        if self.print_warnings:
                            print("Explorer bot not moved", explorer, ", path=", path)
                fogged_count = self.fog_map[explorer.row][explorer.col]
                # explore
                if fogged_count > 0 and not (temp_goal is not None and
                                             (explorer.row != temp_goal[0] or explorer.col != temp_goal[1])):
                    # if there is a better position, wait until the next turn to explore
                    if not explorer.acted and game_state.can_robot_action(rname):
                        game_state.robot_action(rname)


        my_miner_names = set([robot.name for robot in my_miners])
        miner_names_to_pop = []
        for miner_name, _ in self.miner_goal.items():
            if miner_name not in my_miner_names:
                miner_names_to_pop.append(miner_name)
        for miner_name in miner_names_to_pop:
            self.miner_goal.pop(miner_name, None)
        miner_names_to_pop = []
        for miner_name, _ in self.miner_charging.items():
            if miner_name not in my_miner_names:
                miner_names_to_pop.append(miner_name)
        for miner_name in miner_names_to_pop:
            self.miner_charging.pop(miner_name, None)
        mined_mines = set()
        remaining_mines = set()
        # print("Miner goals:", self.miner_goal)
        # print("Miner charging states:", self.miner_charging)
        # print("My miners:", my_miners)
        # print("My mines:", my_mines)
        for miner in my_miners:
            rname = miner.name
            if rname in self.miner_goal.keys():
                mined_mines.add(self.miner_goal[rname])
        for mine in my_mines:
            if mine not in mined_mines:
                remaining_mines.add(mine)
        # print(mined_mines, remaining_mines)
        if self.print_time:
            print("Time after moving explorers:", time.time() - self.begin_time)

        # move miners
        for miner in my_miners:
            rname = miner.name
            bomb = False
            for dir in Direction:  # try to bomb opponent miner in one step if it's mining our mine
                row = miner.row + dir.value[0]
                col = miner.col + dir.value[1]
                if row < 0 or row >= height or col < 0 or col >= width:
                    continue
                if self.map[row][col] is None:  # fogged
                    continue
                if self.map[row][col].robot is None or self.map[row][col].robot.team == self.team:
                    continue
                if self.map[row][col].state != TileState.MINING:
                    continue
                bomb = False
                if rname in self.miner_goal.keys() and self.miner_goal[rname] == (row, col):
                    # bomb whatever robot because this mine belongs to this robot
                    bomb = True
                if self.map[row][col].robot.type == RobotType.MINER:
                    # bomb whatever mine because opponent is mining
                    bomb = True
                if bomb:  # sacrificed
                    if self.move_robot(miner, dir):
                        self.miner_goal.pop(rname, None)
                        self.miner_charging.pop(rname, None)
                        if (row, col) in mined_mines:
                            mined_mines.remove((row, col))
                            remaining_mines.add((row, col))
                        break
            if bomb:
                continue
            if miner.battery < 20:
                self.miner_charging[rname] = True
            if miner.battery >= 100:
                self.miner_charging[rname] = False
            if self.miner_charging[rname]:
                # print("Miner charging", miner, self.miner_goal[rname])
                if self.map[miner.row][miner.col].terraform <= 0:  # not charging
                    path = game_state.robot_to_base(rname, checkCollisions=True)
                    if path[0] is not None and self.move_robot(miner, path[0]):
                        pass
                        # print("Moved", miner)
                # else:
                #     print("Charging")
            else:
                if rname not in self.miner_goal.keys():
                    # print("Miner finding mine", miner)
                    # take a remaining mine
                    best_mine = (None, -1)
                    for mine in remaining_mines:
                        path = game_state.optimal_path(miner.row, miner.col, mine[0], mine[1], checkCollisions=False)
                        if path[0] is not None:
                            if best_mine[0] is None or path[1] < best_mine[1]:
                                best_mine = (mine, path[1])
                    if best_mine[0] is not None:
                        self.miner_goal[rname] = (best_mine[0][0], best_mine[0][1])
                        remaining_mines.remove(best_mine[0])
                        # print("Found mine", miner, best_mine[0])
                if rname in self.miner_goal.keys():
                    goal = self.miner_goal[rname]
                    if not (miner.row == goal[0] and miner.col == goal[1]):
                        path = game_state.optimal_path(miner.row, miner.col, goal[0], goal[1], checkCollisions=True)
                        if path[0] is not None and self.move_robot(miner, path[0]):
                            pass
                        else:
                            if self.print_warnings:
                                print("Miner stuck", miner, goal)
                            # randomly move with probability 10% to prevent blocking other robots
                            if random.randint(0, 9) == 0:
                                valid_dirs = []
                                for dir in Direction:
                                    row = miner.row + dir.value[0]
                                    col = miner.col + dir.value[1]
                                    if row < 0 or row >= height or col < 0 or col >= width:
                                        continue
                                    if self.map[row][col] is None:  # fogged
                                        continue
                                    if self.map[row][col].robot is not None and self.map[row][
                                        col].robot.team == self.team:
                                        continue
                                    if self.map[row][col].state in [TileState.ILLEGAL, TileState.IMPASSABLE]:
                                        continue
                                    valid_dirs += [dir]
                                if len(valid_dirs) > 0:
                                    self.move_robot(miner, random.choice(valid_dirs))
                    if miner.row == goal[0] and miner.col == goal[1]:
                        if game_state.can_robot_action(rname):  # mine
                            game_state.robot_action(rname)
        if self.print_time:
            print("Time after moving miners:", time.time() - self.begin_time)

        # move bombs
        if len(my_bombs) > 0 and len(opponent_mines) > 0:
            for bomb in my_bombs:
                rname = bomb.name
                if self.map[bomb.row][bomb.col].state == TileState.MINING and \
                    self.count_mine_terraform(bomb.row, bomb.col) < 0:  # already at good location
                    continue
                for dir in Direction: # try to bomb opponent robots nearby
                    row = bomb.row + dir.value[0]
                    col = bomb.col + dir.value[1]
                    if row < 0 or row >= height or col < 0 or col >= width:
                        continue
                    if self.map[row][col] is None:  # fogged
                        continue
                    if self.map[row][col].robot is None or self.map[row][col].robot.team == self.team:
                        continue
                    if self.map[row][col].robot.battery < 20:  # not worth it
                        continue
                    if self.move_robot(bomb, dir):
                        break
                if not bomb.moved:
                    for mine in opponent_mines:
                        if self.map[mine[0]][mine[1]].robot is not None and \
                                self.map[mine[0]][mine[1]].robot.team == self.team:  # not bomb myself
                            continue
                        path = game_state.optimal_path(bomb.row, bomb.col, mine[0], mine[1], checkCollisions=True)
                        if path[0] is not None and self.move_robot(bomb, path[0]):  # bomb the mine
                            break
                        for goal_dir in Direction:
                            row = mine[0] + goal_dir.value[0]
                            col = mine[1] + goal_dir.value[1]
                            if row < 0 or row >= height or col < 0 or col >= width:
                                continue
                            if self.map[row][col] is None:  # fogged
                                continue
                            path = game_state.optimal_path(bomb.row, bomb.col, row, col, checkCollisions=True)
                            if path[0] is None:
                                continue
                            if self.move_robot(bomb, path[0]):
                                break
                if not bomb.moved:
                    # randomly move with probability 20% to prevent blocking other robots
                    if random.randint(0, 4) == 0:
                        valid_dirs = []
                        for dir in Direction:
                            row = bomb.row + dir.value[0]
                            col = bomb.col + dir.value[1]
                            if row < 0 or row >= height or col < 0 or col >= width:
                                continue
                            if self.map[row][col] is None:  # fogged
                                continue
                            if self.map[row][col].robot is not None and self.map[row][col].robot.team == self.team:
                                continue
                            if self.map[row][col].state in [TileState.ILLEGAL, TileState.IMPASSABLE]:
                                continue
                            valid_dirs += [dir]
                        if len(valid_dirs) > 0:
                            self.move_robot(bomb, random.choice(valid_dirs))
        if self.print_time:
            print("Time after moving bombs:", time.time() - self.begin_time)

        kSecureMinesTurns = 100
        # move terraformers
        # print("My terraformers", my_terraformers)
        self.compute_reachable_by_terraformers(my_terraformers)
        for row in range(height):
            for col in range(width):
                tile = self.map[row][col]
                if tile is not None and tile.state == TileState.MINING and self.reachable_by_terraformers[row][col] is not None:
                    insecure_count = self.count_mine_insecure(row, col)
                    if insecure_count > 0:
                        if insecure_count not in insecure_mines.keys():
                            insecure_mines[insecure_count] = []
                        insecure_mines[insecure_count] += [(tile.row, tile.col)]
        select_secure_mines = False
        if ginfo.turn < kSecureMinesTurns:
            # greedily secure the mines
            for insecure_count, mines in insecure_mines.items():
                for mine in mines:
                    for dir in Direction:
                        r = mine[0] + dir.value[0]
                        c = mine[1] + dir.value[1]
                        if r < 0 or r >= height or c < 0 or c >= width:
                            continue
                        if self.map[r][c] is None:
                            continue
                        if self.map[r][c].state != TileState.TERRAFORMABLE:
                            continue
                        if self.map[r][c].robot is not None:
                            continue
                        if self.map[r][c].terraform > 0:
                            continue
                        if self.reachable_by_terraformers[r][c] is None:
                            continue
                        terraformer = self.reachable_by_terraformers[mine[0]][mine[1]]
                        if terraformer.moved or terraformer.acted:
                            continue
                        if terraformer.battery < 20:
                            continue
                        if self.map[terraformer.row][terraformer.col].state == TileState.TERRAFORMABLE and \
                                self.map[terraformer.row][terraformer.col].terraform <= 0:
                            continue
                        rname = terraformer.name
                        path = game_state.optimal_path(terraformer.row, terraformer.col, r, c, checkCollisions=True)
                        if path[0] is not None and self.move_robot(terraformer, path[0]):
                            self.terraformer_charging[rname] = False # avoid charging again before reaching the destination
                            # no need to break -- we're not working on one robot
                            if self.map[terraformer.row][terraformer.col] is not None and \
                                    self.map[terraformer.row][terraformer.col].terraform <= 9 and \
                                    (self.map[terraformer.row][terraformer.col].terraform <= 0 or terraformer.battery >= 100):
                                if game_state.can_robot_action(rname):
                                    game_state.robot_action(rname)
                                    terraformer.acted = True

        if self.last_secure_mines_time < 0.5 and ginfo.turn < kSecureMinesTurns and \
                len(my_terraformers) <= len(insecure_mines) * 2:  # never secure mines again if any time it takes > 0.5 seconds
            start_secure_mines_time = time.time()
            select_secure_mines = True
            exclusions = set()
            for terraformer in my_terraformers:
                rname = terraformer.name
                if terraformer.moved or terraformer.acted:
                    continue
                bomb = self.try_to_bomb_opponent_mining(terraformer)
                if bomb is not None:
                    if self.map[terraformer.row][terraformer.col].state == TileState.TERRAFORMABLE and \
                            self.map[terraformer.row][terraformer.col].terraform <= 9 and game_state.can_robot_action(rname):
                        game_state.robot_action(rname)  # last terraform before sacrificing
                    if self.move_robot(terraformer, bomb):
                        continue
                    else:
                        if self.print_warnings:
                            print("Failed to bomb using terraformer!")
                        continue
                if terraformer.battery < 20:
                    self.terraformer_charging[rname] = True
                if terraformer.battery >= 100:
                    self.terraformer_charging[rname] = False
                if self.terraformer_charging[rname]:
                    if self.map[terraformer.row][terraformer.col].terraform <= 0:  # not charging
                        path = game_state.robot_to_base(rname, checkCollisions=True)
                        if path[0] is not None and self.move_robot(terraformer, path[0]):
                            pass
                else:
                    moved = False
                    best_goal = (None, -1)
                    if self.map[terraformer.row][terraformer.col].state == TileState.TERRAFORMABLE and \
                            self.map[terraformer.row][terraformer.col].terraform <= 0:
                        if game_state.can_robot_action(rname):
                            game_state.robot_action(rname)
                            terraformer.acted = True
                            continue
                    # move
                    try_count = 0
                    for insecure_count, mines in insecure_mines.items():
                        for mine in mines:
                            # check reachable
                            try_count += 1
                            path = game_state.optimal_path(terraformer.row, terraformer.col, mine[0], mine[1], checkCollisions=False)
                            if path[0] is None:
                                if self.debug:
                                    print("Unreachable from", terraformer, "to", mine)
                                continue
                            for dir in Direction:
                                r = mine[0] + dir.value[0]
                                c = mine[1] + dir.value[1]
                                if r < 0 or r >= height or c < 0 or c >= width:
                                    continue
                                if self.map[r][c] is None:
                                    continue
                                if self.map[r][c].state != TileState.TERRAFORMABLE:
                                    continue
                                if self.map[r][c].robot is not None:
                                    continue
                                if self.map[r][c].terraform > 0:
                                    continue
                                if self.reachable_by_terraformers[r][c] is None:
                                    continue
                                try_count += 1
                                path = game_state.optimal_path(terraformer.row, terraformer.col, r, c, checkCollisions=True)
                                if path[0] is not None:
                                    if best_goal[0] is None or path[1] < best_goal[2]:
                                        best_goal = ((r, c), (insecure_count, mine), path[1])
                                    if path[1] <= 1:
                                        break
                                else:
                                    if self.debug:
                                        print("Unreachable 2 from", terraformer, "to", mine)
                                    pass
                        if best_goal[0] is not None:
                            goal = (best_goal[0][0], best_goal[0][1])
                            path = game_state.optimal_path(terraformer.row, terraformer.col, goal[0], goal[1], checkCollisions=True)
                            if path[0] is not None and self.move_robot(terraformer, path[0]):
                                moved = True
                                break
                    if self.debug:
                        print("Tried", try_count, best_goal)
                    if moved:
                        insecure_count, mine = best_goal[1]
                        insecure_mines[insecure_count].remove(mine)
                    if moved and not terraformer.acted and self.map[terraformer.row][terraformer.col] is not None and \
                            self.map[terraformer.row][terraformer.col].terraform <= 9 and \
                            (self.map[terraformer.row][terraformer.col].terraform <= 0 or terraformer.battery >= 100):
                        if game_state.can_robot_action(rname):
                            game_state.robot_action(rname)
                            terraformer.acted = True
                    if not moved and not terraformer.acted:
                        # fall back to terraform anything
                        path = self.optimal_terraform(terraformer.row, terraformer.col,
                                                      exclusions, checkCollisions=True)
                        if path[0] is not None and self.move_robot(terraformer, path[0]):
                            exclusions.add(path[1])
                            moved = True
                        if not terraformer.acted and self.map[terraformer.row][terraformer.col] is not None and \
                                self.map[terraformer.row][terraformer.col].terraform <= 9 and \
                                (self.map[terraformer.row][terraformer.col].terraform <= 0 or terraformer.battery >= 100):
                            if game_state.can_robot_action(rname):
                                game_state.robot_action(rname)
                        if not terraformer.moved:
                            # randomly move with probability 20% to prevent blocking other robots
                            if random.randint(0, 4) == 0:
                                valid_dirs = []
                                for dir in Direction:
                                    row = terraformer.row + dir.value[0]
                                    col = terraformer.col + dir.value[1]
                                    if row < 0 or row >= height or col < 0 or col >= width:
                                        continue
                                    if self.map[row][col] is None:  # fogged
                                        continue
                                    if self.map[row][col].robot is not None and self.map[row][col].robot.team == self.team:
                                        continue
                                    if self.map[row][col].state in [TileState.ILLEGAL, TileState.IMPASSABLE]:
                                        continue
                                    valid_dirs += [dir]
                                if len(valid_dirs) > 0:
                                    self.move_robot(terraformer, random.choice(valid_dirs))

            end_secure_mines_time = time.time()
            self.last_secure_mines_time = end_secure_mines_time - start_secure_mines_time
        else:  # terraform as much as we can
            exclusions = set()
            for terraformer in my_terraformers:
                if terraformer.moved or terraformer.acted:
                    continue
                rname = terraformer.name
                if terraformer.battery < 20:
                    self.terraformer_charging[rname] = True
                if terraformer.battery >= 100:
                    self.terraformer_charging[rname] = False
                if terraformer.battery // 20 >= 200 - ginfo.turn:  # at the end
                    self.terraformer_charging[rname] = False
                if self.terraformer_charging[rname]:
                    if self.map[terraformer.row][terraformer.col].terraform <= 0:  # not charging
                        path = game_state.robot_to_base(rname, checkCollisions=True)
                        if path[0] is not None and self.move_robot(terraformer, path[0]):
                            pass
                else:
                    if not terraformer.acted and self.map[terraformer.row][terraformer.col] is not None and \
                            self.map[terraformer.row][terraformer.col].terraform <= 0:
                        if game_state.can_robot_action(rname):
                            game_state.robot_action(rname)
                            terraformer.acted = True
                    path = self.optimal_terraform(terraformer.row, terraformer.col,
                                                  exclusions, checkCollisions=True)
                    if path[0] is not None and self.move_robot(terraformer, path[0]):
                        exclusions.add(path[1])
                        moved = True
                    if not terraformer.acted and self.map[terraformer.row][terraformer.col] is not None and \
                            self.map[terraformer.row][terraformer.col].terraform <= 9 and \
                            (self.map[terraformer.row][terraformer.col].terraform <= 0 or terraformer.battery >= 100):
                        if game_state.can_robot_action(rname):
                            game_state.robot_action(rname)
                    if not terraformer.moved:
                        # randomly move with probability 20% to prevent blocking other robots
                        if random.randint(0, 4) == 0:
                            valid_dirs = []
                            for dir in Direction:
                                row = terraformer.row + dir.value[0]
                                col = terraformer.col + dir.value[1]
                                if row < 0 or row >= height or col < 0 or col >= width:
                                    continue
                                if self.map[row][col] is None:  # fogged
                                    continue
                                if self.map[row][col].robot is not None and self.map[row][col].robot.team == self.team:
                                    continue
                                if self.map[row][col].state in [TileState.ILLEGAL, TileState.IMPASSABLE]:
                                    continue
                                valid_dirs += [dir]
                            if len(valid_dirs) > 0:
                                self.move_robot(terraformer, random.choice(valid_dirs))

        if self.print_time:
            print("Time after moving terraformers (select_secure_mines =", select_secure_mines, "):",
                  time.time() - self.begin_time)


        # spawn locations
        for row in range(height):
            for col in range(width):
                # get the tile at (row, col)
                tile = self.map[row][col]
                # skip fogged tiles
                if tile is not None:  # ignore fogged tiles
                    if tile.robot is None and tile.terraform > 0:  # ensure tile is ally-terraformed
                        ally_tiles += [(row, col)]

        kOneExplorerBotDistanceThreshold = 2
        num_blocks = self.compute_connectivity()
        kExpectedNumExplorers = min(num_blocks + 2, 10)
        if ginfo.turn == 0:
            kExpectedNumExplorers = min(num_blocks, 10)
            if kExpectedNumExplorers < 10 and len(ally_tiles) - kExpectedNumExplorers > 1:
                kExpectedNumExplorers += 1
        if len(remaining_mines) > 0 and kExpectedNumExplorers > num_blocks:
            kExpectedNumExplorers = min(kExpectedNumExplorers,
                                        max(num_blocks,
                                            game_state.get_metal() // game_state.get_spawn_cost() - len(remaining_mines)))
        if self.total_fog_count / self.height / self.width > 0.1:
            kExpectedNumExplorers += len(my_terraformers) // 5
        blocks_to_spawn = set()
        for i in range(num_blocks):
            blocks_to_spawn.add(i)
        for explorer in my_explorers:
            if self.connectivity[explorer.row][explorer.col] is not None and \
                    self.connectivity[explorer.row][explorer.col] in blocks_to_spawn:
                blocks_to_spawn.remove(self.connectivity[explorer.row][explorer.col])
        # try to spawn explorers
        if len(ally_tiles) > 0 and len(fogged_tiles) > 0 and len(my_explorers) < kExpectedNumExplorers and \
                game_state.get_metal() >= game_state.get_spawn_cost():
            spawn_num = min(min(kExpectedNumExplorers - len(my_explorers),
                            game_state.get_metal() // game_state.get_spawn_cost()),
                            len(ally_tiles))
            for spawn_loc in ally_tiles:
                if spawn_num <= 0 or len(blocks_to_spawn) == 0:
                    break
                if self.connectivity[spawn_loc[0]][spawn_loc[1]] in blocks_to_spawn:
                    # spawn the robot
                    spawn_type = RobotType.EXPLORER
                    if self.debug:
                        print(f"Spawning explorer at {spawn_loc[0], spawn_loc[1]}")
                    # check if we can spawn here (checks if we can afford, tile is empty, and tile is ours)
                    if game_state.can_spawn_robot(spawn_type, spawn_loc[0], spawn_loc[1]):
                        robot = game_state.spawn_robot(spawn_type, spawn_loc[0], spawn_loc[1])
                        ally_tiles.remove(spawn_loc)
                        spawn_num -= 1
                        blocks_to_spawn.remove(self.connectivity[spawn_loc[0]][spawn_loc[1]])
                    else:
                        if self.print_warnings:
                            print(f"Failed to spawn explorer for connectivity!")

            exploring_fogged_tiles = fogged_tiles
            for i in range(spawn_num):
                min_fog_distances = [(min([coord_distance(co, ally_tile) for co in exploring_fogged_tiles]), ally_tile)
                                     for ally_tile in ally_tiles]
                min_fog_distance = min_fog_distances[0]
                for tmp in min_fog_distances:
                    if tmp[0] < min_fog_distance[0]:
                        min_fog_distance = tmp
                spawn_loc = min_fog_distance[1]
                spawn_type = RobotType.EXPLORER
                goal = None
                new_exploring_fogged_tiles = []
                for tile in exploring_fogged_tiles:
                    if coord_distance(spawn_loc, tile) == min_fog_distance[0]:
                        goal = tile
                    elif coord_distance(spawn_loc, tile) > min_fog_distance[0] + kOneExplorerBotDistanceThreshold:
                        new_exploring_fogged_tiles += [tile]
                # spawn the robot
                if self.debug:
                    print(f"Spawning explorer at {spawn_loc[0], spawn_loc[1]} with goal {goal[0], goal[1]}")
                # check if we can spawn here (checks if we can afford, tile is empty, and tile is ours)
                if game_state.can_spawn_robot(spawn_type, spawn_loc[0], spawn_loc[1]):
                    robot = game_state.spawn_robot(spawn_type, spawn_loc[0], spawn_loc[1])
                    self.explorer_goal[robot.name] = goal
                    ally_tiles.remove(spawn_loc)
                else:
                    if self.print_warnings:
                        print(f"Failed to spawn explorer!")
                exploring_fogged_tiles = new_exploring_fogged_tiles
                if len(exploring_fogged_tiles) == 0:
                    break
        if self.print_time:
            print("Time after spawning explorers:", time.time() - self.begin_time)

        kMinTerraformerPerHowManyMiners = 3
        # try to spawn miners
        if len(ally_tiles) > 0 and len(remaining_mines) > 0 and game_state.get_metal() >= game_state.get_spawn_cost():
            spawn_num = min(min(len(remaining_mines),
                                game_state.get_metal() // game_state.get_spawn_cost()),
                            len(ally_tiles))
            if len(my_miners) + spawn_num >= kMinTerraformerPerHowManyMiners * (len(my_terraformers) + 1):
                spawn_num -= (len(my_miners) + spawn_num - kMinTerraformerPerHowManyMiners * len(my_terraformers)) \
                             // (kMinTerraformerPerHowManyMiners + 1)
                if spawn_num < 0:
                    spawn_num = 0
            for mine in remaining_mines:
                if spawn_num <= 0:
                    break
                ally_tile_set = set([(tile[0], tile[1]) for tile in ally_tiles])
                spawn_loc = self.closest_ally_tile_to_spawn(mine[0], mine[1], ally_tile_set)
                if spawn_loc[0] is None:
                    if self.print_warnings:
                        print("Failed to select spawn location for mine", mine, ally_tiles)
                    spawn_loc = random.choice(ally_tiles)
                spawn_type = RobotType.MINER
                # spawn the robot
                if self.debug:
                    print(f"Spawning miner at {spawn_loc[0], spawn_loc[1]} with goal {mine}")
                # check if we can spawn here (checks if we can afford, tile is empty, and tile is ours)
                if game_state.can_spawn_robot(spawn_type, spawn_loc[0], spawn_loc[1]):
                    robot = game_state.spawn_robot(spawn_type, spawn_loc[0], spawn_loc[1])
                    self.miner_charging[robot.name] = False
                    self.miner_goal[robot.name] = mine
                    ally_tiles.remove(spawn_loc)
                    spawn_num -= 1
                else:
                    if self.print_warnings:
                        print(f"Failed to spawn miner!")
        if self.print_time:
            print("Time after spawning miners:", time.time() - self.begin_time)

        # try to spawn terraformers
        if len(ally_tiles) > 0 and game_state.get_metal() >= game_state.get_spawn_cost():
            spawn_num = min(game_state.get_metal() // game_state.get_spawn_cost(),
                            len(ally_tiles))
            for i in range(spawn_num):
                spawn_loc = random.choice(ally_tiles)  # TODO
                spawn_type = RobotType.TERRAFORMER
                # spawn the robot
                if self.debug:
                    print(f"Spawning terraformer at {spawn_loc[0], spawn_loc[1]}")
                # check if we can spawn here (checks if we can afford, tile is empty, and tile is ours)
                if game_state.can_spawn_robot(spawn_type, spawn_loc[0], spawn_loc[1]):
                    robot = game_state.spawn_robot(spawn_type, spawn_loc[0], spawn_loc[1])
                    self.terraformer_charging[robot.name] = False
                    ally_tiles.remove(spawn_loc)
                else:
                    if self.print_warnings:
                        print(f"Failed to spawn terraformer!")
        if self.print_time:
            print("Time after spawning terraformers:", time.time() - self.begin_time)
