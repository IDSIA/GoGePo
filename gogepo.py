import torch
import numpy as np
import gym
import core_gogepo as core
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env_name",
    default="Swimmer-v3",
    choices=[
        "Swimmer-v3",
        "Hopper-v3",
        "InvertedPendulum-v2",
        "MountainCarContinuous-v0",
    ],
    type=str,
    required=False,
)
parser.add_argument("--verbose", default=0, type=int, required=False)
parser.add_argument("--show_plots", default=0, type=int, required=False)
parser.add_argument("--use_gpu", default=1, type=int, required=False)
parser.add_argument("--seed", default=1234, type=int, required=False)
args = parser.parse_args()

verbose = args.verbose
show_plots = args.show_plots

# Default hyperparameters
config = dict(
    algo="hpg_r",
    size_buffer=10000,
    size_buffer_command=64,
    max_episodes=1000000000,
    max_timesteps=3000000,
    run=1,  # id
    ts_evaluation=10000,
    episodes_per_epoch=1,
    start_steps=0,
    seed=args.seed,
    # Env
    env_name=args.env_name,
    survival_bonus=False,
    # Policy
    neurons_policy=(256, 256),
    noise_policy=0.1,  # std of distribution generating the noise for the perturbed policy
    observation_normalization=True,
    deterministic_actor=True,
    # Command evaluation
    ts_evaluation_generator=100000,
    rew_min=0,
    rew_max=3000,
    n_steps=20,
    # Command optimization
    normalize_command=True,
    noise_command=0,  # max, heu
    drive_parameter=20,  # max, heu
    update_command="sampled",  # max, sampled, combine
    # vf
    neurons_vf=(256, 256),
    learning_rate_vf=5e-3,
    vf_iters=5,
    n_probing_states=200,
    # Generator
    use_hyper=True,
    gen_iters=20,
    neurons_generator=(256, 256),
    batch_size=16,
    learning_rate_gen=2e-6,
    z_dim_w=8,
    z_dim_b=8,
    out_size_w=[16, 16],
    out_size_b=16,
    reset_command=100,
    weighted_sampling_command=False,
    use_bound=True,
    use_virtual_class=True,
    update_every_ts=False,
    update_every=100,
    weighted_sampling=True,
    scale=1.1,
    # IS
    use_is=False,
    learning_rate_command=1e-3,  # is
    batch_size_command=16,  # is
    updates_command=5,  # is
    delta=0.5,
    use_gradient=False,
    use_bh=True,
    use_parallel=True,
    scale_layer_out=True,
    scale_parameter=2,
    use_max_pred=False,
    noise_command_up=0,
    drift_command_up=0,
    save=False,
    save_model_every=100000000,
)


if config["env_name"] == "CartPole-v1":
    config.update({"rew_min": 0}, allow_val_change=True)
    config.update({"rew_max": 500}, allow_val_change=True)
elif config["env_name"] == "Swimmer-v3":
    config.update({"rew_min": -100}, allow_val_change=True)
    config.update({"rew_max": 365}, allow_val_change=True)
elif config["env_name"] == "InvertedPendulum-v2":
    config.update({"rew_min": 0}, allow_val_change=True)
    config.update({"rew_max": 1000}, allow_val_change=True)
    config.update({"ts_evaluation_generator": 10000}, allow_val_change=True)
    config.update({"max_timesteps": 100000}, allow_val_change=True)
    config.update({"ts_evaluation": 1000}, allow_val_change=True)


elif config["env_name"] == "Walker2d-v3":
    config.update({"rew_min": -100}, allow_val_change=True)
    config.update({"rew_max": 3000}, allow_val_change=True)
elif config["env_name"] == "HalfCheetah-v3":
    config.update({"rew_min": -100}, allow_val_change=True)
    config.update({"rew_max": 4000}, allow_val_change=True)
elif config["env_name"] == "Hopper-v3":
    config.update({"rew_min": -100}, allow_val_change=True)
    config.update({"rew_max": 3000}, allow_val_change=True)
elif config["env_name"] == "InvertedDoublePendulum-v2":
    config.update({"rew_min": 0}, allow_val_change=True)
    config.update({"rew_max": 10000}, allow_val_change=True)
    config.update({"ts_evaluation_generator": 10000}, allow_val_change=True)
    config.update({"max_timesteps": 100000}, allow_val_change=True)
    config.update({"ts_evaluation": 1000}, allow_val_change=True)

elif config["env_name"] == "MountainCarContinuous-v0":
    config.update({"rew_min": -100}, allow_val_change=True)
    config.update({"rew_max": 100}, allow_val_change=True)
    config.update({"ts_evaluation_generator": 10000}, allow_val_change=True)
    config.update({"max_timesteps": 100000}, allow_val_change=True)
    config.update({"ts_evaluation": 1000}, allow_val_change=True)


if config["env_name"] in [
    "MountainCarContinuous-v0",
    "InvertedPendulum-v2",
    "Reacher-v2",
]:
    config.update(
        {
            "ts_evaluation": 1000,
            "max_timesteps": 100000,
        },
        allow_val_change=True,
    )
config.update(
    {
        "size_buffer_command": config["batch_size_command"],
    },
    allow_val_change=True,
)

# Use GPU or CPU
if args.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

torch.manual_seed(config["seed"])
np.random.seed(config["seed"])

# Create env
env = gym.make(config["env_name"])
env_test = gym.make(config["env_name"])

# Create replay buffer, policy, vf
buffer = core.Buffer(config["size_buffer"], scale=config["scale"])
command_buffer = core.Buffer(config["size_buffer_command"])
statistics = core.Statistics(env.observation_space.shape)
ac = core.MLPActorCritic(
    env.observation_space,
    env.action_space,
    config["n_probing_states"],
    hidden_sizes_actor=tuple(config["neurons_policy"]),
    activation=nn.Tanh,
    hidden_sizes_critic=tuple(config["neurons_vf"]),
    device=device,
    critic=True,
    deterministic_actor=config["deterministic_actor"],
).to(device)
command = core.Command()

params_dim = len(nn.utils.parameters_to_vector(list(ac.pi.parameters())))

sizes = [param.shape for param in list(ac.parameters())]

generator = core.Generator(
    ac.pi,
    params_dim,
    hidden_sizes=tuple(config["neurons_generator"]),
    use_hyper=config["use_hyper"],
    use_parallel=config["use_parallel"],
    hid_size_w=config["z_dim_w"],
    hid_size_b=config["z_dim_b"],
    out_size_w=config["out_size_w"],
    out_size_b=config["out_size_b"],
    device=device,
    scale_layer_out=config["scale_layer_out"],
    scale_parameter=config["scale_parameter"],
).to(device=device)

if config["use_virtual_class"]:
    virtual_mlp = core.VirtualMLPPolicy(
        layer_sizes=[env.observation_space.shape[0]]
        + list(tuple(config["neurons_policy"]))
        + [env.action_space.shape[0]],
        act_lim=env.action_space.high[0],
    )

if verbose:
    print(generator.encoder)

    print(ac.pi)
    print(
        "Number of policy params:",
        len(nn.utils.parameters_to_vector(list(ac.pi.parameters()))),
    )
    print(
        "Number of vf params:",
        len(nn.utils.parameters_to_vector(list(ac.v.parameters()))),
    )
    print(
        "Number of generator params:",
        len(nn.utils.parameters_to_vector(list(generator.parameters()))),
    )

    model_params = (
        nn.utils.parameters_to_vector(list(ac.pi.parameters())).detach().to("cpu")
    )
    q = torch.tensor([0.25, 0.5, 0.75])
    print(
        "init quant 0.25, 0.5, 0.75",
        torch.quantile(model_params, q, dim=0, keepdim=True),
    )

    for p in ac.pi.parameters():
        print("max", torch.max(p))

# Setup optimizer
optimize_generator = optim.Adam(generator.parameters(), lr=config["learning_rate_gen"])
optimize_vf = optim.Adam(ac.v.parameters(), lr=config["learning_rate_vf"])


def compute_vf_loss(progs, rew):
    q = ac.v(
        progs,
        use_virtual_module=config["use_virtual_class"],
        virtual_module=virtual_mlp,
    )
    statistics.max_pred = max(statistics.max_pred, torch.max(q).detach().item())
    loss = ((q - rew) ** 2).mean()
    return loss


def compute_generator_loss(rew):
    loss = (
        (
            generator(
                rew / (config["rew_max"] - config["rew_min"]),
                use_virtual_module=config["use_virtual_class"],
                evaluator=ac.v,
                virtual_module=virtual_mlp,
            )
            - rew.squeeze()
        )
        ** 2
    ).mean()
    return loss


def perturb_policy(policy):
    dist = Normal(
        torch.zeros(len(torch.nn.utils.parameters_to_vector(policy.parameters()))),
        scale=1,
    )
    delta = dist.sample().to(device=device, non_blocking=True).detach()

    # Perturbe policy parameters
    params = torch.nn.utils.parameters_to_vector(policy.parameters()).detach()
    perturbed_params = params + config["noise_policy"] * delta

    # Copy perturbed parameters into a new policy
    perturbed_policy = core.MLPActorCritic(
        env.observation_space,
        env.action_space,
        config["n_probing_states"],
        hidden_sizes_actor=tuple(config["neurons_policy"]),
        activation=nn.Tanh,
        hidden_sizes_critic=tuple(config["neurons_vf"]),
        device=device,
        critic=False,
        deterministic_actor=config["deterministic_actor"],
    ).to(device)

    torch.nn.utils.vector_to_parameters(perturbed_params, perturbed_policy.parameters())

    return perturbed_policy


def evaluate_behavior_offline(rew_min, rew_max, n_steps):
    step = (rew_max - rew_min) / n_steps
    outputs = []

    for idx in range(n_steps + 1):
        rew = rew_min + idx * step

        with torch.no_grad():
            out_rew, parameters = generator(
                torch.tensor([rew / (config["rew_max"] - config["rew_min"])])
                .float()
                .to(device)
                .unsqueeze(0),
                use_virtual_module=config["use_virtual_class"],
                evaluator=ac.v,
                virtual_module=virtual_mlp,
                return_all=True,
            )

            outputs.append((out_rew.squeeze().cpu().numpy(), rew))

    return outputs


def perturb_command(command):
    if config["use_is"]:
        if config["use_gradient"]:
            perturbed_rew = command.cpu().item() * (
                config["rew_max"] - config["rew_min"]
            )

        else:
            perturbed_rew = command.cpu().item()

    else:
        dist = Normal(0, scale=1)
        delta = dist.sample().item()
        # Perturbe policy parameters
        perturbed_rew = (
            command + config["drive_parameter"] + config["noise_command"] * delta
        )

    return perturbed_rew


def update_command():
    if config["use_max_pred"]:
        steps = torch.rand(50).float().unsqueeze(1).to(device)
        steps = (
            config["rew_min"] + steps * (statistics.max_rew - config["rew_min"]) * 1.2
        )
        with torch.no_grad():
            values = generator(
                steps / (config["rew_max"] - config["rew_min"]),
                use_virtual_module=config["use_virtual_class"],
                evaluator=ac.v,
                virtual_module=virtual_mlp,
            )
        command.command = torch.nn.Parameter(torch.max(values))
    else:
        command.command = torch.nn.Parameter(torch.as_tensor(statistics.max_rew))
        return


def update():
    # Update evaluator
    start_time = time.perf_counter()

    for idx in range(1, config["vf_iters"]):
        # Sample batch
        hist = buffer.sample_replay(
            config["batch_size"], weighted_sampling=config["weighted_sampling"]
        )
        prog, rew, rew_gen = zip(*hist)
        rew = (
            torch.from_numpy(np.asarray(rew))
            .float()
            .to(device=device, non_blocking=True)
            .detach()
        )
        prog = torch.stack(prog)
        optimize_vf.zero_grad()
        loss_vf = compute_vf_loss(prog, rew)
        loss_vf.backward()
        optimize_vf.step()

    statistics.up_v_time += time.perf_counter() - start_time
    start_time = time.perf_counter()

    for p in ac.v.parameters():
        p.requires_grad = False
    for _ in range(1, config["gen_iters"]):
        if config["update_command"] == "generated":
            t = config["drive_parameter"] + config["noise_command"]
            rew_gen = (
                statistics.min_rew
                + t
                + torch.rand(config["batch_size"]).float().unsqueeze(1).to(device)
                * (statistics.max_rew - statistics.min_rew)
            )

            optimize_generator.zero_grad()
            loss_gen = compute_generator_loss(rew_gen.float().to(device))
            loss_gen.backward()
            optimize_generator.step()
        elif config["update_command"] == "sampled":
            hist = buffer.sample_replay(
                config["batch_size"], weighted_sampling=config["weighted_sampling"]
            )
            _, _, rew_gen = zip(*hist)

            rew_gen = torch.stack(rew_gen)
            rew_gen += (
                config["drift_command_up"]
                + torch.rand(rew_gen.shape[0]).float().unsqueeze(1)
                * config["noise_command_up"]
            )

            optimize_generator.zero_grad()
            loss_gen = compute_generator_loss(rew_gen.float().to(device))
            loss_gen.backward()
            optimize_generator.step()
        else:
            raise ValueError

    for p in ac.v.parameters():
        p.requires_grad = True

    statistics.up_policy_time += time.perf_counter() - start_time

    log_dict = {
        "loss_gen": loss_gen.item(),
        "loss_vf": loss_vf.item(),
        "grads_norm_generator": core.grad_norm(generator.parameters()),
        "norm_generator": core.norm(generator.parameters()),
        "norm_pvf": core.norm(ac.v.parameters()),
        "grads_norm_pvf": core.grad_norm(ac.v.parameters()),
        "norm_prob_states": core.norm(ac.v.probing_states.parameters()),
        "grads_norm_prob_states": core.grad_norm(ac.v.probing_states.parameters()),
        "max_rew": statistics.max_rew,
    }
    if verbose:
        print(log_dict)

    return


def evaluate(policy_params, log=True, n_eval=10):
    rew_evals = []
    with torch.no_grad():
        for _ in range(n_eval):

            # Simulate a trajectory and compute the total reward
            done = False
            obs = env_test.reset()
            rew_eval = 0
            while not done:
                obs = torch.as_tensor(obs, dtype=torch.float32)
                if config["observation_normalization"] and statistics.episode > 0:
                    obs = statistics.normalize(obs)

                with torch.no_grad():
                    action = ac.act(
                        obs.to(device), policy_params, virtual_module=virtual_mlp
                    )
                    # action = ac.act(obs.to(device), policy_params)
                obs_new, r, done, _ = env_test.step(action[0])

                rew_eval += r
                obs = obs_new

            rew_evals.append(rew_eval)
        if log:
            statistics.rew_eval = np.mean(rew_evals)
            statistics.push_rew(np.mean(rew_evals))
    # Log results
    if log:
        print(
            "Ts",
            statistics.total_ts,
            "Ep",
            statistics.episode,
            "rew_eval",
            statistics.rew_eval,
        )
        print(
            "time_sim",
            statistics.sim_time,
            "time_gen",
            statistics.gen_time,
            "time_up_pi",
            statistics.up_policy_time,
            "time_up_v",
            statistics.up_v_time,
            "total_time",
            statistics.total_time,
        )
    return np.mean(rew_evals)


def simulate_policy(perturbed_params):
    # Simulate a trajectory and compute the total reward
    done = False
    obs = env.reset()
    rew = 0
    while not done:
        obs = torch.as_tensor(obs, dtype=torch.float32)
        if config["observation_normalization"]:
            statistics.push_obs(obs)
            if statistics.episode > 0:
                obs = statistics.normalize(obs)

        with torch.no_grad():
            action = ac.act(
                obs.to(device), perturbed_params, virtual_module=virtual_mlp
            )
        obs_new, r, done, _ = env.step(action[0])
        if not config["survival_bonus"]:
            if (
                config["env_name"] == "Hopper-v3"
                or config["env_name"] == "Ant-v3"
                or config["env_name"] == "Walker2d-v3"
            ):
                rew += r - 1
            elif config["env_name"] == "Humanoid-v3":
                rew += r - 5
            else:
                rew += r
        else:
            rew += r

        statistics.total_ts += 1

        # Evaluate current policy
        if (
            statistics.total_ts % config["ts_evaluation"] == 0
            and statistics.episode > 0
        ):
            with torch.no_grad():
                if config["use_max_pred"]:
                    parameters = generator(
                        torch.tensor(
                            [
                                command.command.cpu().item()
                                / (config["rew_max"] - config["rew_min"])
                            ]
                        )
                        .float()
                        .to(device)
                        .unsqueeze(0)
                    )

                else:
                    parameters = generator(
                        torch.tensor(
                            [
                                statistics.max_rew
                                / (config["rew_max"] - config["rew_min"])
                            ]
                        )
                        .float()
                        .to(device)
                        .unsqueeze(0)
                    )
                parameters = parameters.squeeze()
            evaluate(parameters)

        # Update
        if (
            statistics.total_ts > config["start_steps"]
            and config["update_every_ts"]
            and statistics.episode > 0
        ):
            if statistics.total_ts % config["update_every"] == 0:
                update()

        # save metrics
        model_states = {"ac": ac, "generator": generator, "statistics": statistics}

        if config["save"]:

            if statistics.total_ts % config["save_model_every"] == 0:
                torch.save(
                    model_states,
                    "data/model_"
                    + str(config["seed"])
                    + str(config["env_name"])
                    + str(statistics.total_ts)
                    + ".pth",
                )

                print("saving model")

        if statistics.total_ts == 1000000:
            log_dict = {
                "rew_eval_1M": statistics.rew_eval,
                "average_reward_1M": np.mean(statistics.rewards),
                "average_last_rewards_1M": np.mean(statistics.last_rewards),
            }
            if verbose:
                print(log_dict)

        if statistics.total_ts % config["ts_evaluation_generator"] == 0:
            result = evaluate_behavior(
                config["rew_min"], config["rew_max"], config["n_steps"]
            )
            if show_plots:
                y, x = zip(*result)
                fig, ax = plt.subplots(1, 1)
                ax.plot(x, y)
                ax.plot(x, x)
                plt.show()

            result = evaluate_behavior_offline(
                config["rew_min"], config["rew_max"], config["n_steps"]
            )
            if show_plots:
                y, x = zip(*result)
                fig2, ax2 = plt.subplots(1, 1)
                ax2.plot(x, y)
                ax2.plot(x, x)
                plt.show()
        obs = obs_new
    return rew


def evaluate_behavior(rew_min, rew_max, n_steps):
    step = (rew_max - rew_min) / n_steps
    outputs = []

    for idx in range(n_steps + 1):
        rew = rew_min + idx * step

        with torch.no_grad():
            parameters = generator(
                torch.tensor([rew / (config["rew_max"] - config["rew_min"])])
                .float()
                .to(device)
                .unsqueeze(0)
            )
            parameters = parameters.squeeze()
            out_rew = evaluate(parameters, log=False, n_eval=1)
            outputs.append((out_rew, rew))

    return outputs


def train():
    start_time = time.perf_counter()

    # Choose command
    if statistics.episode > 0:
        perturbed_command = perturb_command(command.command)
    else:
        perturbed_command = 1

    # Generate policy and perturbe it
    with torch.no_grad():
        perturbed_params, logp_a = generator(
            torch.tensor([perturbed_command / (config["rew_max"] - config["rew_min"])])
            .unsqueeze(0)
            .float()
            .to(device),
            noise=config["noise_policy"],
        )
        perturbed_params = perturbed_params.squeeze()

        mean_param = generator(
            torch.tensor([perturbed_command / (config["rew_max"] - config["rew_min"])])
            .unsqueeze(0)
            .float()
            .to(device)
        )
        mean_param = mean_param.squeeze().to("cpu")

        if statistics.episode == 0:
            torch.nn.utils.vector_to_parameters(mean_param, ac.pi.parameters())

            if verbose:
                for p in ac.pi.parameters():
                    print("max gen", torch.max(p))

    statistics.gen_time += time.perf_counter() - start_time
    start_time = time.perf_counter()

    # Simulate a trajectory and compute the total reward
    rew = simulate_policy(perturbed_params)
    statistics.max_rew = max(statistics.max_rew, rew)
    statistics.min_rew = min(statistics.min_rew, rew)

    # Store data in replay buffer
    buffer.history.append((perturbed_params, rew, torch.tensor([rew]).float()))
    command_buffer.history.append(
        (perturbed_params, torch.tensor([rew]).float(), logp_a, mean_param)
    )
    if len(command_buffer.history) > command_buffer.size_buffer:
        command_buffer.history.pop(0)

    statistics.episode += 1

    statistics.sim_time += time.perf_counter() - start_time

    # Update
    if statistics.total_ts > config["start_steps"] and not config["update_every_ts"]:
        update()

    # Log results
    if statistics.episode % 50 == 0:
        print("Ts", statistics.total_ts, "Rew", rew)

    log_dict = {
        "rew": rew,
        "steps": statistics.total_ts,
        "episode": statistics.episode,
        "command": command.command.detach().item(),
        "perturbed_command": perturbed_command,
        "max_pred": statistics.max_pred,
    }
    if verbose:
        print(log_dict)

    statistics.push_rew_env(rew)

    # Update command
    if (
        statistics.episode % config["episodes_per_epoch"] == 0
        and statistics.total_ts > config["start_steps"]
    ):
        update_command()
    return


# Loop over episodes
while (
    statistics.total_ts < config["max_timesteps"]
    and statistics.episode < config["max_episodes"]
):
    start_time = time.perf_counter()
    train()
    statistics.total_time += time.perf_counter() - start_time

if config["save"]:
    # save metrics
    model_states = {
        "ac": ac,
        "generator": generator,
        "statistics": statistics,
        "buffer": buffer,
    }

    torch.save(
        model_states,
        "data/final_model_"
        + str(config["seed"])
        + str(config["env_name"])
        + str(statistics.total_ts)
        + ".pth",
    )

    print("saving final model")
