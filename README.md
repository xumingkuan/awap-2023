# AWAP 2023 Game AI

See game engine at https://github.com/ACM-CMU/awap-engine-2023-public.

Usage: put ElaineShiFanClub.py in the bots/ folder in the repo above, and follow the directions there.

## Strategy order
- Move explorers.
- Move miners.
- Move bombs (explorers without energy).
- Move terraformers.
- Spawn explorers.
- Spawn miners.
- Spawn terraformers.

## Move explorers
Each explorer has a pre-set goal tile at the beginning. It will try to approach the goal tile first.
At any time, if any robot encounter an opponent miner that is mining at a mine with at least 20 battery level, it will go to collide with that robot.

If an explorer has already approached the goal tile, it will run a BFS to find the nearest reachable tile with some fogged tiles next to it.
The BFS then find the tile with minimum ((number of moves to the tile) - (number of fogged tiles next to it)), and the explorer will go there, and we will exclude that goal tile from other explorers' search space.

## Move miners
Each miner has a corresponding mine. If the battery level is at least 100, it will try to go to that mine and mine it until its battery level drops to below 20.

If the corresponding mine is occupied by an opponent robot, it will collide regardless of its battery level and opponent's robot type.

When the battery level drops to be low 20, it will go back to the nearest ally-terraformed tile to recharge until the battery level increases to at least 100.

If a miner is unable to find a move (usually because the mine is blocked by ally robots), they will move randomly to an empty tile with probability 10% in total, to prevent blocking other robots.

## Move bombs
When explorers exhaust energy, they become bombs and they will not go back to recharge.

Bombs try to occupy opponent-controlled mines. A mine is opponent-controlled if and only if there are more opponent-terraformed tiles than ally-terraformed tiles in the 8 tiles next to the mine.

If at any time a bomb sees an opponent robot with battery level at least 20, it will collide with that robot regardless of its type.

If a bomb is unable to find a place to go (usually because there are no visible opponent-controlled mines), they will move randomly to an empty tile with probability 20% in total, to prevent blocking other robots.

## Move terraformers
For each terraformer, if it is already at a terraformable tile that is not ally-terraformed and it has enough battery level to do that, it will terraform first. If it is still not ally-terraformed after that, it will stay there for the turn.

In the first 100 out of the 200 turns, we try to secure all mines. A mine is secured if and only if all of the 8 tiles next to it are either illegal, impassible, or ally-terraformed.

As an optimization, we do a BFS from all terraformers together, and then enumerate all tiles next to insecured mines to try to pull the closest terraformer to secure it.
Of course, if a terraformer is already pulled to one tile, it cannot be pulled to another. So we still need to enumerate the rest of terraformers.

For each terraformer, we enumerate all insecured mines and call `optimal_path` to go there. This is very time-consuming and could be replaced with one BFS.
If the entire process of moving all terraformers takes more than 0.5 seconds in one turn (or after 100 turns), we abort this process forever and go to the next phase.

In the next phase, if there are still more terraformers or if the previous phase is skipped, we do one BFS for each terraformer to find the closest terraformable but not ally-terraformed tile, try to go there, and exclude that tile from options for other terraformers.

If there are still terraformers that are unable to find a place to go (usually because they are blocked by other terraformers), they will move randomly to an empty tile with probability 20% in total, to prevent blocking other robots.

After moving, each terraformer, if not terraformed before moving, will try to terraform the tile if it is not ally-terraformed or if the ally-terraformed level is not full (10) and the battery level is at least 100.

## Spawn explorers
We gather all empty ally tiles before all the movement and actions. Newly spawned robots cannot move or perform actions anyway.

For explorers, we try to maintain a fixed number of them. The number is initially set as the number of connected blocks in terms of visible regions plus 2. Considering the map can be arbitrary, we limit the number of explorers at 10 initially.
At the first turn, i.e., turn 0, realizing we may want to spawn a terraformer instead of 4 explorers, we set the number of explorers to be the number of connected blocks, and if there are at least 2 more available ally tiles left, we increase the number by 1.
We then consider how many miners we want to spawn in this turn. If there is not enough money or tiles to spawn all of them, we cut at most 2 explorers.
For large maps with a lot of resources, we may need more explorers. If there are still more than 10% of the map being fogged, we increase the number of explorers by the number of terraformers divided by 5 then rounded down.

We spawn explorers on the tiles with lowest infinity-norm to any fogged tile, and set the explorer's goal to the closest fogged tile.

## Spawn miners
We try to assign a miner to each mine that is not opponent-controlled. However, to prevent keep spawning miners for maps full of mines, we require spawning one terraformer after whenever 3 more miners are spawned.

For each mine, we spawn the miner on the tile closest to it.

## Spawn terraformers
We try to use up our metal to spawn terraformers. However, it seems that we're not doing that exactly in replays.

The spawn locations for terraformers are random.

## Implementation tricks
We keep track of the whole map on our own after calling `game_state.get_info()` for one time each turn. This is because the function `game_state.get_info()` is rather expensive.

Because time out is automatically losing, we simply do nothing when there are less than 5 seconds left (and we hope this will never happen).
