import os
import math
import torch
import json
import numpy as np
import random

from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from networks.hypnets import HyperbolicMLP, HyperbolicDeepSet, HyperbolicCategoricalMLP
from networks.nets import SmallEncoder, DeepSet, LabelEncoder, CategoricalMLP

from environments.maze.pyramid import create_pyramid
from environments.maze.continuous_maze import bfs, ContinuousGridEnvironment
from environments.tree.tree import NaryTreeEnvironment
from environments.tree.discrete_maze import GridMazeEnvironment


def evaluate(
    maze,
    num_trials,
    encoder1,
    encoder2,
    manifold,
    device,
    max_steps=100,
    hyperbolic=False,
    eps=10.0,
    step_size=0.5,
    verbose=False,
):
    """
    Run policy on maze, collect failure metrics
    """
    valid_indices = np.argwhere(maze == 0)
    np.random.shuffle(valid_indices)

    results = []
    for i in range(num_trials):
        with torch.no_grad():
            start, end = np.random.randint(0, len(valid_indices), size=2)
            start = tuple(valid_indices[start])
            end = tuple(valid_indices[end])

            goal = torch.tensor(end).to(torch.float32).to(device).unsqueeze(0)
            # if hyperbolic:
            #     goal = manifold_map(goal, manifold=manifold)
            goal = encoder2(goal)

            # print(start)
            env = ContinuousGridEnvironment(maze, start, {})

            def reached(cur_pos, goal_pos):
                # print(f'cur pos: {cur_pos}')
                cur_pos = (int(cur_pos[0]), int(cur_pos[1]))
                goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
                return cur_pos == goal_pos

            def step():
                cur_pos = env.agent_position
                if verbose:
                    print(f"cur_pos: {cur_pos}, goal: {goal}")
                activations = []
                angles = torch.linspace(0.0, 2 * torch.pi, 16)
                for a in angles:
                    action = torch.tensor([torch.sin(a), torch.cos(a)])
                    cur = torch.tensor(
                        [cur_pos[0], cur_pos[1], torch.sin(a), torch.cos(a)]
                    ).to(device, torch.float32)
                    # if hyperbolic:
                    #     cur = manifold_map(cur, manifold)
                    cur = encoder1(cur)

                    # MANIFOLD EVAL
                    if hyperbolic:
                        activations.append((action, -manifold.dist(x=cur, y=goal)))
                    else:
                        activations.append((action, -torch.norm(cur - goal)))

                best_action = activations[np.argmax([x[1].cpu() for x in activations])][
                    0
                ]
                angle = np.arctan2(
                    best_action[0], best_action[1]
                ) + np.random.normal() * eps * (2 * np.pi / 360)
                best_action = torch.tensor(np.array([np.sin(angle), np.cos(angle)]))
                env.move_agent(best_action)
                # print(f'agent position: {env.agent_position}')

            def SPL(
                maze, start, end, num_steps, success
            ):  # Success weighted by (normalized inverse) Path Length
                if not success:
                    return 0
                else:
                    p = num_steps * step_size
                    l = len(bfs(maze, start, end))
                    return l / max(p, l)

            steps = 0
            while not reached(env.agent_position, end):
                if steps > max_steps:
                    break
                step()
                steps += 1

            result = (
                not reached(env.agent_position, end),
                steps,
                SPL(maze, start, end, steps, reached(env.agent_position, end)),
            )
            if verbose:
                print(reached(env.agent_position, end))
                print(
                    f"start: {start}, goal: {end}, end_pos: {env.agent_position}, steps: {steps}"
                )
                print(results)

            results.append(result)

    return results

def eval_tree(
    depth,
    branching_factor,
    num_trials,
    encoder1,
    encoder2,
    manifold,
    device,
    max_steps=100,
    hyperbolic=False,
    verbose=False,
):
    """
    Run policy on maze, collect failure metrics
    """
    

    results = []
    for i in range(num_trials):
        with torch.no_grad():
            env = NaryTreeEnvironment(depth=depth, branching_factor=branching_factor)
            
            start, end = np.random.randint(0, env.num_nodes, size=2)
            
            goal = torch.tensor(end).to(device).unsqueeze(0)
            # if hyperbolic:
            #     goal = manifold_map(goal, manifold=manifold)
            goal = encoder2(goal)

            # print(f'start: {start}, goal: {goal}')

            def reached(cur_pos, goal_pos):
                return cur_pos == goal_pos

            def step():
                cur_pos = env.agent_position
                if verbose:
                    print(f"cur_pos: {cur_pos}, goal: {goal}")
                activations = []
                
                actions = [i for i in range(branching_factor + 2)]
                for a in actions:
                    cur = torch.tensor(
                        [env.agent_position, a]
                    ).to(device)

                    cur = encoder1(cur)

                    # MANIFOLD EVAL
                    if hyperbolic:
                        activations.append((a, -manifold.dist(x=cur, y=goal)))
                    else:
                        activations.append((a, -torch.norm(cur - goal)))

                best_action = activations[np.argmax([x[1].cpu() for x in activations])][
                    0
                ]
                env.move_agent(best_action)
                if verbose:
                    print(f'agent position: {env.agent_position}')

            def SPL(
                start, end, num_steps, success
            ):  # Success weighted by (normalized inverse) Path Length
                if not success:
                    return 0
                else:
                    p = num_steps
                    l = len(env.get_action_path(start, end))
                    return l / max(p, l)

            steps = 0
            while not reached(env.agent_position, end):
                if steps > max_steps:
                    break
                step()
                steps += 1

            result = (
                not reached(env.agent_position, end),
                steps,
                SPL(start, end, steps, reached(env.agent_position, end)),
            )
            if verbose:
                print(reached(env.agent_position, end))
                print(
                    f"start: {start}, goal: {end}, end_pos: {env.agent_position}, steps: {steps}"
                )
                print(results)

            results.append(result)

    return results


def eval_maze(
    maze,
    num_trials,
    encoder1,
    encoder2,
    manifold,
    device,
    max_steps=100,
    hyperbolic=False,
    verbose=False
):
    """
    Run policy on maze, collect failure metrics
    """
    results = []
    for i in range(num_trials):
        with torch.no_grad():
            env = GridMazeEnvironment(maze)
            valid_positions = [env.flatten_state(pos) for pos in env.valid_indices]
            start, end = random.sample(valid_positions, 2)
            
            env.agent_position = start
            goal = torch.tensor(end).to(device).unsqueeze(0)
            goal = encoder2(goal)

            def reached(cur_pos, goal_pos):
                return cur_pos == goal_pos

            def step():
                cur_pos = env.agent_position
                if verbose:
                    print(f"cur_pos: {cur_pos}, goal: {end}")
                activations = []
                
                actions = [i for i in range(5)]
                for a in actions:
                    cur = torch.tensor([env.agent_position, a]).to(device)
                    cur = encoder1(cur)

                    if hyperbolic:
                        activations.append((a, -manifold.dist(x=cur, y=goal)))
                    else:
                        activations.append((a, -torch.norm(cur - goal)))

                best_action = activations[np.argmax([x[1].cpu() for x in activations])][0]
                env.move_agent(best_action)
                if verbose:
                    print(f'agent position: {env.agent_position}')

            def SPL(start, end, num_steps, success):
                if not success:
                    return 0
                else:
                    p = num_steps
                    l = len(env.get_action_path(start, end))
                    return l / max(p, l)

            steps = 0
            while not reached(env.agent_position, end):
                if steps > max_steps:
                    break
                step()
                steps += 1

            result = (
                not reached(env.agent_position, end),
                steps,
                SPL(start, end, steps, reached(env.agent_position, end)),
            )
            if verbose:
                print(reached(env.agent_position, end))
                print(f"start: {start}, goal: {end}, end_pos: {env.agent_position}, steps: {steps}")
                print(result)

            results.append(result)

    return results

def get_maze(name):
    """
    Pre-set mazes
    """
    maze = np.zeros((10, 10))

    if "blank" in name:
        print("blank maze")
        maze = np.zeros((10, 10))
    elif "slit" in name:
        print("slit maze")
        maze = np.zeros((11, 11))
        maze[:, 5] = 1
        maze[5, 5] = 0
    elif "blocker" in name:
        maze = np.zeros((11, 11))
        maze[3, :] = 1
        maze[3, 10] = 0
    elif "nested_pyramid" in name:
        maze = create_pyramid(np.zeros((2, 2)), 2)[0]
    elif "fork" in name:
        maze = np.zeros((15, 15))
        for i in range(1, 14, 2):
            maze[i, : maze.shape[1] - 5] = 1
    elif "island" in name:
        maze = np.zeros((11, 11))
        maze[3:8, 5:7] = 1
    else:
        maze = create_pyramid(np.zeros((2, 2)), 1)[0]

    return maze


def get_order_function(name):
    """
    order trajectories by some arbitrary metric. (e.g.) horizontal is where trajectories can only end to the right of the start
    """

    def order_by_second_coordinate(point1, point2):
        return point1[1] - point2[1]

    def order_by_first_coordinate(point1, point2):
        return point1[0] - point2[0]

    def order_by_distance_from_origin(point1, point2):
        dist1 = math.sqrt(point1[0] ** 2 + point1[1] ** 2)
        dist2 = math.sqrt(point2[0] ** 2 + point2[1] ** 2)
        return dist1 - dist2

    name = name.lower()
    if "horizontal" in name:
        print(f"horizontal order fn")
        return order_by_second_coordinate
    elif "vertical" in name:
        print(f"vertical order fn")
        return order_by_first_coordinate
    elif "origin" in name:
        print(f"dist order fn")
        return order_by_distance_from_origin
    else:
        print(f"no order")
        return None


def save_models(config, encoder1, encoder2, epoch, name="", street_encoder=None):
    print(f'saved model to {name}')
    # Create the main models directory if it doesn't exist
    config = dict(config)
    os.makedirs("saved_models", exist_ok=True)

    model_dir = os.path.join("saved_models", name)
    os.makedirs(model_dir, exist_ok=True)

    # Save the encoder models
    torch.save(
        encoder1.state_dict(), os.path.join(model_dir, f"encoder1_epoch_{epoch}.pth")
    )
    torch.save(
        encoder2.state_dict(), os.path.join(model_dir, f"encoder2_epoch_{epoch}.pth")
    )
    if street_encoder is not None:
        torch.save(
            street_encoder.state_dict(),
            os.path.join(model_dir, f"street_encoder_epoch_{epoch}.pth"),
        )

    # Save the config as JSON
    config_path = os.path.join(model_dir, f"config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)


def load_model(config, device, pretrained_path="", epoch=0):
    config = dict(config)

    curvature = Curvature(
        value=config["curvature"], requires_grad=config["learnable_curvature"]
    )
    manifold = PoincareBall(c=curvature)
    if config["architecture"] == "MLP":
        if config["hyperbolic"]:
            encoder1 = HyperbolicMLP(
                in_features=4,
                out_features=config["embedding_dim"],
                euc_width=64,
                hyp_width=64,
                manifold=manifold,
            ).to(device)
            encoder2 = HyperbolicMLP(
                in_features=2,
                out_features=config["embedding_dim"],
                euc_width=64,
                hyp_width=64,
                manifold=manifold,
            ).to(device)
        else:
            encoder1 = SmallEncoder(
                input_dim=4, embedding_dim=config["embedding_dim"]
            ).to(device)
            encoder2 = SmallEncoder(
                input_dim=2, embedding_dim=config["embedding_dim"]
            ).to(device)
            manifold = None
    elif config["architecture"] == "DeepSet":
        if config["hyperbolic"]:
            encoder1 = HyperbolicDeepSet(
                input_dim=2,
                hidden_dim=64,
                output_dim=config["embedding_dim"],
                manifold=manifold,
            ).to(device)
            encoder2 = HyperbolicDeepSet(
                input_dim=2,
                hidden_dim=64,
                output_dim=config["embedding_dim"],
                manifold=manifold,
            ).to(device)
            optimizer = RiemannianAdam(
                list(encoder1.parameters()) + list(encoder2.parameters()),
                lr=config["learning_rate"],
            )
        else:
            encoder1 = DeepSet(
                input_dim=2, hidden_dim=64, output_dim=config["embedding_dim"]
            ).to(device)
            encoder2 = DeepSet(
                input_dim=2, hidden_dim=64, output_dim=config["embedding_dim"]
            ).to(device)
            optimizer = torch.optim.Adam(
                list(encoder1.parameters()) + list(encoder2.parameters()),
                lr=config["learning_rate"],
            )
    else:
        encoder1 = None
        encoder2 = None
        optimizer = None
        manifold = None

    if len(pretrained_path) > 0:
        # print("loading pretrained...")
        encoder1.load_state_dict(
            torch.load(
                os.path.join(pretrained_path, f"encoder1_epoch_{epoch}.pth"),
                map_location=torch.device(device),
            )
        )
        encoder2.load_state_dict(
            torch.load(
                os.path.join(pretrained_path, f"encoder2_epoch_{epoch}.pth"),
                map_location=torch.device(device),
            )
        )

    return {"encoder1": encoder1, "encoder2": encoder2, "manifold": manifold}


def load_tree_model(config, state_dim, action_dim, device, pretrained_path="", epoch=0):
    config = dict(config)

    curvature = Curvature(
        value=config["curvature"], requires_grad=config["learnable_curvature"]
    )
    manifold = PoincareBall(c=curvature)

    if config['embedding_dim'] > 16:
        e = config['embedding_dim']
        
        euc_hidden_dims = [2 * e, 2 * e, e]
        hyp_hidden_dims = [e, e]
        
    else:
        euc_hidden_dims = [64, 64, 32]
        hyp_hidden_dims = [32, 32]

    first_dim = 2 * euc_hidden_dims[0] 
    if config["hyperbolic"] and config['architecture'] != 'DeepSet':
        encoder1 = HyperbolicCategoricalMLP(
            cat_features=[state_dim, action_dim],
            embedding_dims=[first_dim, first_dim],
            euc_hidden_dims=euc_hidden_dims,
            hyp_hidden_dims=hyp_hidden_dims,
            output_dim=config["embedding_dim"],
            manifold=manifold
        ).to(device)
        encoder2 = HyperbolicCategoricalMLP(
            cat_features=[state_dim],
            embedding_dims=[first_dim],
            euc_hidden_dims=euc_hidden_dims,
            hyp_hidden_dims=hyp_hidden_dims,
            output_dim=config["embedding_dim"],
            manifold=manifold
        ).to(device)
    else:
        
        encoder1 = CategoricalMLP(
            cat_features=[state_dim, action_dim],
            embedding_dims=[first_dim, first_dim],
            hidden_dims=euc_hidden_dims,
            output_dim=config["embedding_dim"],
        ).to(device)
        encoder2 = CategoricalMLP(
            cat_features=[state_dim],
            embedding_dims=[first_dim],
            hidden_dims=euc_hidden_dims,
            output_dim=config["embedding_dim"],
        ).to(device)

        if config['architecture'] == 'DeepSet':
            intermed_dim = 3 * config["embedding_dim"]
            encoder1 = CategoricalMLP(
                cat_features=[state_dim, action_dim],
                embedding_dims=[first_dim, first_dim],
                hidden_dims=euc_hidden_dims,
                output_dim=intermed_dim,
            ).to(device)
            encoder2 = CategoricalMLP(
                cat_features=[state_dim],
                embedding_dims=[first_dim],
                hidden_dims=euc_hidden_dims,
                output_dim=intermed_dim,
            ).to(device)

            if config['hyperbolic']:
                encoder1 = HyperbolicDeepSet(input_dim=2, hidden_dim=intermed_dim, output_dim=config["embedding_dim"], manifold=manifold, phi=encoder1).to(device)
                encoder2 = HyperbolicDeepSet(input_dim=1, hidden_dim=intermed_dim, output_dim=config["embedding_dim"], manifold=manifold, phi=encoder2).to(device)
            else:
                print('constructing non-hyperbolic Deep Set')
                encoder1 = DeepSet(input_dim=2, hidden_dim=intermed_dim, output_dim=config["embedding_dim"], phi=encoder1).to(device)
                encoder2 = DeepSet(input_dim=1, hidden_dim=intermed_dim, output_dim=config["embedding_dim"], phi=encoder2).to(device)


    if len(pretrained_path) > 0:
        print("loading pretrained...")
        encoder1.load_state_dict(
            torch.load(
                os.path.join(pretrained_path, f"encoder1_epoch_{epoch}.pth"),
                map_location=torch.device(device),
            )
        )
        encoder2.load_state_dict(
            torch.load(
                os.path.join(pretrained_path, f"encoder2_epoch_{epoch}.pth"),
                map_location=torch.device(device),
            )
        )

    return {"encoder1": encoder1, "encoder2": encoder2, "manifold": manifold}



def load_street_model(config, device, pretrained_path="", epoch=0):
    """
    loads the categorical encoder for language experiments
    """
    maze = get_maze(config["maze_type"])
    street_encoder = LabelEncoder(
        num_categories=(sum(maze.shape)), embedding_dim=config["embedding_dim"]
    ).to(device)

    if len(pretrained_path) > 0:
        street_encoder.load_state_dict(
            torch.load(
                os.path.join(pretrained_path, f"street_encoder_epoch_{epoch}.pth"),
                map_location=torch.device(device),
            )
        )

    return street_encoder