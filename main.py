import argparse
import requests
import random
import json
import math
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

API_KEY = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDJaiXgI3BG+mtmm3aagRe3T8kYO0cesB7/qKHcB8K8AYmql4bcZ2rJW4wkt/yeVstItD+yvzLhstXoYea5XvlEMZw4zxrNVqxL0QmK157eeZtJFVtZRJ4vfnHgvMsU8lumLfV49qK+W9IjsgdlrlTzo8u6pHASPIA8MIxrX15ARz36TS4r4b18jKPOL7HJNwGG9r5Hl/nffkR9AZyrlilzckiCYhw6wjBpMumgbR+yxy11bFwn29d1czWW4JjcQ6YqR+aFDtzqCfKibE2YAHal18iVdybShupaPFe+gGL24JvqlixDUSSkSJMX9SDv/xEFSkHFBGdtMr3iCP077IgQ8f3d1eSgRpVpVeDXh7xyP5gpe9Bb1jv4B2Y4l5fT8/VALmuHNjk5kJWkVUbHgGWR4oGUe+5WFyTeGTc9Asl3NQJII/qHowLirFmbQME5R9NqATBDKs8qlZwCGlmyz9pfSOEIhjRYm2FFyKc2zJ7dyMrw/gqH33ANEr884KNxsGKRy3kPRqk93ZhuEIc5qdB+wno1u3DpH7G+TG1HcR02H7GvI//bfvu3BSzF6z80tt7e4w2oZhnEsWyUqjTtv33qaFapyxzonSbD7oGpx2i/uMTCB9itzqPW3Fmrt4HkkLsgdel20kyPyeJVgyqmL/CaNGhsLyGQJZur6g2YKmgtqw== mattreed@stanford.edu"
FLOP_LIMIT = 2e18

D_MODEL_CHOICES = range(64, 1025)
NUM_LAYERS_CHOICES = range(2, 25)
NUM_HEADS_CHOICES = range(2, 17)
BATCH_SIZE_CHOICES = [128, 256]
LEARNING_RATE_CHOICES = [1e-3, 1e-4]
# FLOPS_CHOICES = [1e13, 3e13, 6e13, 1e14, 3e14, 6e14]
FLOPS_CHOICES = [1e13, 3e13, 6e13, 1e14, 3e14, 6e14, 1e15, 3e15, 6e15, 1e16, 3e16, 6e16, 1e17, 3e17, 6e17, 1e18]

CHOICES_DICT = {
    "d_model": D_MODEL_CHOICES,
    "num_layers": NUM_LAYERS_CHOICES,
    "num_heads": NUM_HEADS_CHOICES,
    "batch_size": BATCH_SIZE_CHOICES,
    "learning_rate": LEARNING_RATE_CHOICES,
    "train_flops": FLOPS_CHOICES
}
# flops_choices = [1e13, 3e13, 6e13, 1e14, 3e14, 6e14, 1e15, 3e15, 6e15, 1e16,
    #                  3e16, 6e16, 1e17, 3e17, 6e17, 1e18]\

def get_loss(d_model, num_layers, num_heads, batch_size, learning_rate, train_flops):
    config = {
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_flops": train_flops,
        "api_key": API_KEY
    }
    response = requests.get("http://tahoma.stanford.edu:8000/loss", config).json()

    # if train_flops > 3e13:
    #     raise ValueError("FLOPS TOO HIGH")

    if "loss" not in response:
        print(response)
        return None

    loss = response["loss"]
    with open("output.csv", "a") as f:
        output = f"{d_model},{num_layers},{num_heads},{batch_size},{learning_rate},{train_flops},{loss}\n"
        f.write(output)
        print(output)
    return loss

def loss_from_indices(model_indices, flops):
    return get_loss(D_MODEL_CHOICES[model_indices["d_model"]], NUM_LAYERS_CHOICES[model_indices["num_layers"]],
                    NUM_HEADS_CHOICES[model_indices["num_heads"]], BATCH_SIZE_CHOICES[model_indices["batch_size"]],
                    LEARNING_RATE_CHOICES[model_indices["learning_rate"]], flops)

def save_indices(model_indices, flops, loss):
    with open("model_bests.json", "r") as f:
        data = json.load(f)
        data[str(flops)] = {
            "indices": model_indices,
            "loss": loss
        }
    with open("model_bests.json", "w") as f:
        json.dump(data, f, indent=4)

def load_indices(flops):
    flops = str(flops)
    with open("model_bests.json", "r") as f:
        data = json.load(f)
        if flops in data:
            print("initializing from existing data")
            print(data[flops])
            return data[flops]["indices"], data[flops]["loss"]
        else:
            print("No data for this FLOPS value, initializing randomly")
            # closest_flops = min(data.keys(), key=lambda x: abs(int(x) - int(flops)))
            # return data[closest_flops]["indices"], 1e50
            return {
                "d_model": random.randint(0, len(D_MODEL_CHOICES) - 1),
                "num_layers": random.randint(0, len(NUM_LAYERS_CHOICES) - 1),
                "num_heads": random.randint(0, len(NUM_HEADS_CHOICES) - 1),
                "batch_size": random.randint(0, len(BATCH_SIZE_CHOICES) - 1),
                "learning_rate": random.randint(0, len(LEARNING_RATE_CHOICES) - 1),
            }, 1e50

def gradient_descent_hyperparameter_search(flops, steps=100, threshold=0):
    print("Starting Gradient Descent Search")
    cost = 2 * steps * flops
    cost_percent = cost / FLOP_LIMIT * 100
    print(f"Cost is {cost} ({cost_percent:.4f}%)")
    print("Do you wish continue? (y/n)")
    if input() != "y":
        print("Exiting")
        return
    current_model_indices, best_loss = load_indices(flops)


    for _ in range(steps):

        curr_loss = loss_from_indices(current_model_indices, flops)
        if curr_loss < best_loss:
            print("New Best Loss Found!")
            best_loss = new_loss
            save_indices(current_model_indices, flops, best_loss)


        new_model_indices = current_model_indices.copy()
        for key in current_model_indices:
            possible_next_choices = [index for index in range(len(CHOICES_DICT[key]))]
            new_model_indices[key] = random.choice(possible_next_choices)
        
        new_loss = loss_from_indices(new_model_indices, flops)
        if new_loss < best_loss:
            print("New Best Loss Found!")
            best_loss = new_loss
            save_indices(new_model_indices, flops, new_loss)
            current_model_indices = new_model_indices
            continue

        def get_gradient(key):
            if float(new_model_indices[key]) == float(current_model_indices[key]):
                return 0
            else:
                return (new_loss - curr_loss)/(float(new_model_indices[key]) - float(current_model_indices[key]))

        gradient = {
            key: get_gradient(key) for key in new_model_indices
        }

        changes = 0
        for key in current_model_indices:
            curr_index = current_model_indices[key]
            if gradient[key] > threshold:
                curr_index += 1
            elif gradient[key] < -threshold:
                curr_index -= 1

            if curr_index in range(len(CHOICES_DICT[key])):
                changes += 1
                current_model_indices[key] = curr_index

        print(changes, "changes made")


        curr_loss = new_loss

def cardinal_hyperparameter_search(flops):
    print("Starting Gradient Descent Search")
    with open("output.csv", "a") as f:
        f.write(f"Gradient Descent Search with {flops} flops\n")
    cost = sum([math.log2(len(CHOICES_DICT[key])) for key in CHOICES_DICT.keys()]) * flops
    cost_percent = cost / FLOP_LIMIT * 100
    print(f"Cost is {cost} ({cost_percent:.4f}%)")
    print("Do you wish continue? (y/n)")
    if input() != "y":
        print("Exiting")
        return
    
    current_model_indices, best_loss = load_indices(flops)

    keys = [key for key in current_model_indices.keys()]
    random.shuffle(keys)

    iterations = 0

    for key in keys:
        print("Optimizing", key)
        with open("output.csv", "a") as f:
            f.write(f"Optimizing {key}\n")

        going_up = current_model_indices[key] < len(CHOICES_DICT[key]) // 2
        step_size = len(CHOICES_DICT[key]) - current_model_indices[key] if going_up else current_model_indices[key]
        step_size //= 2
        step_size = max(step_size, 1)

        while step_size > 0:
            new_model_indices = current_model_indices.copy()
            new_value = None
            if going_up:
                new_value = min(new_model_indices[key] + step_size, len(CHOICES_DICT[key]))
            else:
                new_value = max(new_model_indices[key] - step_size, 0)

            new_model_indices[key] = int(new_value)
            print("Trying", new_value)
            new_loss = loss_from_indices(new_model_indices, flops)
            iterations += 1
            if new_loss < best_loss:
                print("New Best Loss Found!")
                best_loss = new_loss
                save_indices(new_model_indices, flops, new_loss)
                current_model_indices = new_model_indices

            step_size //= 2

    print("Iterations:", iterations)
    print("Total FLOPS used:", iterations * flops)

def save_values(model_values, flops, loss):
    print(f"Saving values {model_values} for {flops} with loss {loss}")
    with open("best_params.json", "r") as f:
        data = json.load(f)
        if str(flops) in data:
            print("Overwriting existing data")
            if data[str(flops)]["loss"] < loss:
                print("New loss is worse than existing loss, not saving")
                return
            
        data[str(flops)] = {
            "values": model_values,
            "loss": loss
        }
    with open("best_params.json", "w") as f:
        json.dump(data, f, indent=4)

def baysian_search(flops, n_calls=10):
    cost = n_calls * flops
    cost_percent = cost / FLOP_LIMIT * 100
    print(f"Cost is {cost} ({cost_percent:.4f}%)")
    print("Do you wish continue? (y/n)")
    if input() != "y":
        print("Exiting")
        return
    
    if cost_percent > 3:
        raise ValueError("TOO MANY FLOPS")

    space = [
        Integer(64, 1024, name='d_model'),
        Integer(2, 24, name='num_layers'),
        Integer(2, 16, name='num_heads'),
        Categorical([128, 256], name='batch_size'),
        Categorical([1e-3, 1e-4], name='learning_rate')
    ]

    @use_named_args(space)
    def objective(**params):
        print("Trying", params)
        return get_loss(
            params["d_model"],
            params["num_layers"],
            params["num_heads"],
            params["batch_size"],
            params["learning_rate"],
            flops
        )

    result = gp_minimize(objective, space, n_calls=n_calls, random_state=0, n_initial_points=max(1, n_calls // 5))

    best_params = result.x
    best_loss = float(result.fun)

    best_params_dict = {
        "d_model": int(best_params[0]),
        "num_layers": int(best_params[1]),
        "num_heads": int(best_params[2]),
        "batch_size": int(best_params[3]),
        "learning_rate": float(best_params[4])
    }

    save_values(best_params_dict, flops, best_loss)


def get_remaining_flops():
    used_flops = requests.get("http://tahoma.stanford.edu:8000/total_flops_used", {"api_key": API_KEY}).json()
    total_flops = FLOP_LIMIT

    used_percent = used_flops / total_flops * 100

    print(f"FLOPS used: {used_flops} ({used_percent:.4f}%)")

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--status', action="store_true", help='Remaining FLOPS')
    parser.add_argument('--loss', action="store_true", help='Remaining FLOPS')
    parser.add_argument('--search', action="store_true", help='Remaining FLOPS')
    parser.add_argument('--cardinal', action="store_true", help='Remaining FLOPS')
    parser.add_argument('--baysian', action="store_true", help='Remaining FLOPS')
    parser.add_argument("-d", "--d_model", type=int, choices=D_MODEL_CHOICES, help="Dimensionality of the model (between 64 and 1024)")
    parser.add_argument("-l", "--num_layers", type=int, choices=NUM_LAYERS_CHOICES, help="Number of layers in the model (between 2 and 24)")
    parser.add_argument("-n", "--num_heads", type=int, choices=NUM_HEADS_CHOICES, help="Number of attention heads (between 2 and 16)")
    parser.add_argument("-b", "--batch_size", type=int, choices=BATCH_SIZE_CHOICES, help="Batch size for training (either 128 or 256)")
    parser.add_argument("-lr", "--learning_rate", type=float, choices=LEARNING_RATE_CHOICES, help="Learning rate for optimizer (either 1e-3 or 1e-4)")
    parser.add_argument("-f", "--train_flops", type=float, default=1e13, choices=FLOPS_CHOICES, help="Number of floating point operations (FLOPS) for training")
    parser.add_argument("--n_calls", type=int, default=1, help="Number of calls to loss function (for baysian search)")


    random.seed(0)

    args = parser.parse_args()
    args = parser.parse_args()
    if args.status:
        get_remaining_flops()
    elif args.loss:
        get_loss(args.d_model, args.num_layers, args.num_heads, args.batch_size, args.learning_rate, int(args.train_flops))
    elif args.search:
        gradient_descent_hyperparameter_search(int(args.train_flops))
    elif args.cardinal:
        cardinal_hyperparameter_search(int(args.train_flops))
    elif args.baysian:
        baysian_search(int(args.train_flops), args.n_calls)

if __name__ == '__main__':
    main()

# python main.py --loss -d 1024 -l 2 -n 2 -b 128 -lr 1e-3 -f 1e13
# python main.py --loss --d_model 1024 -l 2 -n 2 -b 128 -lr 1e-3 -f 1e13