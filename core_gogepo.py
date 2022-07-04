import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gym.spaces import Box, Discrete
import numpy as np
import random
from collections import OrderedDict


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]

    return nn.Sequential(*layers)


class VirtualModule:
    def __init__(self):
        self._parameter_shapes = self.get_parameter_shapes()

        self._num_parameters = 0
        for shape in self.parameter_shapes.values():
            numel = np.prod(shape)
            self._num_parameters += numel

    def get_parameter_shapes(self):
        # return an OrderedDict with the parameter names and their shape
        return NotImplementedError

    def parameter_initialization(self, num_instances):
        factor = 1 / ((self.num_parameters / 100) ** 0.5)
        initializations = []
        for i in range(num_instances):
            p = []
            for key, shape in self.parameter_shapes.items():
                p.append(torch.randn(shape).view(-1) * factor)
            p = torch.cat(p, dim=0)
            initializations.append(p)
        return initializations

    def split_parameters(self, p):
        if len(p.shape) == 1:
            batch_size = []
        else:
            batch_size = [p.shape[0]]
        pointer = 0
        parameters = []
        for shape in self.parameter_shapes.values():
            numel = np.prod(shape)
            x = p[..., pointer : pointer + numel].view(*(batch_size + list(shape)))
            parameters.append(x)
            pointer += numel
        return parameters

    @property
    def parameter_shapes(self):
        return self._parameter_shapes

    @property
    def num_parameters(self):
        return self._num_parameters


class VirtualModuleWrapper(torch.nn.Module):
    # Allows treating a virtual module as a normal pytorch module (train with standard optimizers etc.)
    def __init__(self, virtual_module):
        super().__init__()
        self.virtual_module = virtual_module
        self.virtual_parameters = torch.nn.Parameter(
            self.virtual_module.parameter_initialization(1)[0]
        )

    def forward(self, x):
        output = self.virtual_module.forward(x, self.virtual_parameters)
        return output


def linear_multi_parameter(input, weight, bias=None):
    """
    n: input batch dimension
    m: parameter batch dimension (not obligatory)
    i: input feature dimension
    o: output feature dimension
    :param input: n x (m x) i
    :param weight: (m x) o x i
    :param bias:  (m x) o
    :return: n x (m x) o
    """

    if len(weight.shape) == 2:
        # no parameter batch dimension
        x = torch.einsum("ni,oi->no", input, weight)
    elif len(input.shape) == 3:
        # parameter batch dimension for input and weights
        x = torch.einsum("nmi,moi->nmo", input, weight)
    else:
        # no parameter dimension batch for input
        x = torch.einsum("ni,moi->nmo", input, weight)
    if bias is not None:
        x = x + bias.unsqueeze(0)
    return x


class VirtualMLP(VirtualModule):
    def __init__(self, layer_sizes, nonlinearity="tanh", output_activation="linear"):
        self.layer_sizes = layer_sizes

        if nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        elif nonlinearity == "sigmoid":
            self.nonlinearity = torch.sigmoid
        else:
            self.nonlinearity = torch.relu

        if output_activation == "linear":
            self.output_activation = None
        elif output_activation == "sigmoid":
            self.output_activation = torch.sigmoid
        elif output_activation == "tanh":
            self.output_activation = torch.tanh
        elif output_activation == "softmax":
            self.output_activation = lambda x: torch.softmax(x, dim=-1)

        super(VirtualMLP, self).__init__()

    def get_parameter_shapes(self):
        parameter_shapes = OrderedDict()
        for i in range(1, len(self.layer_sizes)):
            parameter_shapes["w" + str(i)] = (
                self.layer_sizes[i],
                self.layer_sizes[i - 1],
            )
            parameter_shapes["wb" + str(i)] = (self.layer_sizes[i],)

        return parameter_shapes

    def forward(self, input, parameters, callback_func=None):
        # input_sequence: input_batch x (parameter_batch x) input_size
        # parameters: (parameter_batch x) num_params
        # return: input_batch x (parameter_batch x) output_size
        p = self.split_parameters(parameters)
        num_layers = len(self.layer_sizes) - 1
        x = input
        for l in range(0, num_layers):
            w = p[l * 2]
            a = linear_multi_parameter(x, w, bias=p[l * 2 + 1])
            if l < num_layers - 1:
                x = self.nonlinearity(a)
                if callback_func is not None:
                    callback_func(x, l)
            else:
                x = a if self.output_activation is None else self.output_activation(a)
        return x

    def parameter_initialization(self, num_instances, bias_var=0.0):
        initializations = []
        for i in range(num_instances):
            p = []
            for i in range(1, len(self.layer_sizes)):
                w = torch.empty(self.parameter_shapes["w" + str(i)])
                torch.nn.init.xavier_normal_(w)
                p.append(w.view(-1))
            if self.bias:
                for i in range(1, len(self.layer_sizes)):
                    b = torch.empty(self.parameter_shapes["wb" + str(i)])
                    if bias_var == 0:
                        torch.nn.init.zeros_(b)
                    else:
                        torch.nn.init.normal_(b, std=bias_var**0.5)
                    p.append(b.view(-1))
            p = torch.cat(p, dim=0)
            initializations.append(p)
        return initializations


class VirtualMLPPolicy(VirtualMLP):
    def __init__(self, layer_sizes, bias=True, act_lim=1):
        super().__init__(
            layer_sizes=layer_sizes, nonlinearity="tanh", output_activation="tanh"
        )
        self.act_lim = act_lim

    def forward(self, input, parameters, callback_func=None):
        x = super().forward(input, parameters, callback_func)
        x = x * self.act_lim
        return x


def get_hypernetwork_mlp_generator(
    layer_sizes,
    hidden_sizes,
    embedding_dim,
    features_per_embedding=32,
    scale_layer_out=False,
    scale_parameter=1,
):
    input_hn = HyperNetwork(
        hidden_sizes=hidden_sizes,
        z_dim_w=embedding_dim + 1,
        z_dim_b=embedding_dim + 1,
        out_size_w=[
            layer_sizes[1] if len(layer_sizes) == 2 else features_per_embedding,
            layer_sizes[0],
        ],
        out_size_b=layer_sizes[1] if len(layer_sizes) == 2 else features_per_embedding,
    )

    if len(layer_sizes) > 2:
        output_hn = HyperNetwork(
            hidden_sizes=hidden_sizes,
            z_dim_w=embedding_dim + 1,
            z_dim_b=embedding_dim + 1,
            out_size_w=[layer_sizes[-1], features_per_embedding],
            out_size_b=layer_sizes[-1],
        )
    else:
        output_hn = None

    if len(layer_sizes) > 3:
        hidden_hn = HyperNetwork(
            hidden_sizes=hidden_sizes,
            z_dim_w=embedding_dim + 1,
            z_dim_b=embedding_dim + 1,
            out_size_w=[features_per_embedding, features_per_embedding],
            out_size_b=features_per_embedding,
        )
    else:
        hidden_hn = None

    in_tiling = [
        1,
        1 if len(layer_sizes) == 2 else layer_sizes[1] // features_per_embedding,
    ]
    out_tiling = (
        [layer_sizes[-2] // features_per_embedding, 1]
        if len(layer_sizes) >= 2
        else None
    )
    if len(layer_sizes) > 3:
        hidden_tiling = []
        for i in range(1, len(layer_sizes) - 2):
            ht = [
                layer_sizes[i] // features_per_embedding,
                layer_sizes[i + 1] // features_per_embedding,
            ]
            hidden_tiling.append(ht)
    else:
        hidden_tiling = None

    fc_generator = HyperNetworkGenerator(
        input_fc_hn=input_hn,
        hidden_fc_hn=hidden_hn,
        output_fc_hn=output_hn,
        in_tiling=in_tiling,
        hidden_tiling=hidden_tiling,
        out_tiling=out_tiling,
        embedding_dim=embedding_dim,
        layer_sizes=layer_sizes,
        scale_layer_out=scale_layer_out,
        scale_parameter=scale_parameter,
    )

    return fc_generator


class HyperNetwork(nn.Module):
    def __init__(
        self, hidden_sizes, z_dim_w=65, z_dim_b=4, out_size_w=[8, 8], out_size_b=8
    ):
        super(HyperNetwork, self).__init__()
        self.z_dim_w = z_dim_w
        self.z_dim_b = z_dim_b

        self.out_size_w = out_size_w
        self.out_size_b = out_size_b
        self.total_el_w = self.out_size_w[0] * self.out_size_w[1]

        sizes_w = [self.z_dim_w] + list(hidden_sizes) + [self.total_el_w]
        self.net_w = mlp(sizes_w, activation=nn.ReLU)
        sizes_b = [self.z_dim_b] + list(hidden_sizes) + [self.out_size_b]
        self.net_b = mlp(sizes_b, activation=nn.ReLU)

    def forward(self, z, command):
        # z: batch_size x z_dim
        # command: batch_size x 1
        kernel_w = self.net_w(torch.cat((z, command), dim=1))
        kernel_w = kernel_w.view(-1, self.out_size_w[0], self.out_size_w[1])

        kernel_b = self.net_b(torch.cat((z, command), dim=1))
        kernel_b = kernel_b.view(-1, self.out_size_b)

        return kernel_w, kernel_b


class HyperNetworkGenerator(torch.nn.Module):
    def __init__(
        self,
        input_fc_hn: HyperNetwork,
        hidden_fc_hn: HyperNetwork = None,
        output_fc_hn: HyperNetwork = None,
        in_tiling=[1, 1],
        hidden_tiling=None,
        out_tiling=None,
        embedding_dim: int = 64,
        layer_sizes=None,
        scale_layer_out=False,
        scale_parameter=1,
    ):
        super().__init__()
        # layer generators
        self.input_hn = input_fc_hn
        self.hidden_hn = hidden_fc_hn
        self.output_hn = output_fc_hn
        # tilings
        self.in_tiling = in_tiling
        self.hidden_tiling = hidden_tiling
        self.out_tiling = out_tiling
        if layer_sizes is not None:
            self.layer_sizes = layer_sizes
        else:
            raise ValueError
        self.scale_layer_out = scale_layer_out
        self.scale_parameter = scale_parameter
        self.num_layers = 1
        if self.hidden_tiling is not None:
            self.num_layers += len(self.hidden_tiling)
        if self.out_tiling is not None:
            self.num_layers += 1

        # embeddings
        self.in_embeddings = torch.nn.Parameter(
            torch.randn(self.in_tiling + [embedding_dim])
        )
        self.out_embeddings = torch.nn.Parameter(
            torch.randn(self.out_tiling + [embedding_dim])
        )
        if self.num_layers >= 3:
            self.hidden_embeddings = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.randn(ft + [embedding_dim]))
                    for ft in self.hidden_tiling
                ]
            )
        else:
            self.hidden_embeddings = None

    def forward(
        self, command: torch.FloatTensor, conditioning: torch.FloatTensor = None
    ):
        """
        :param command: batch_size x 1
        :param command: batch_size x conditioning_size
        :return:
        """
        batch_size = command.shape[0]

        if conditioning is not None:
            command = torch.cat([command, conditioning], dim=1)

        generated_parameters = []

        # fully connected
        for i in range(self.num_layers):
            if i == 0:
                hn = self.input_hn
                tiling = self.in_tiling
                embeddings = self.in_embeddings
            elif i == self.num_layers - 1:
                hn = self.output_hn
                tiling = self.out_tiling
                embeddings = self.out_embeddings
            else:
                hn = self.hidden_hn
                tiling = self.hidden_tiling[i - 1]
                embeddings = self.hidden_embeddings[i - 1]
            # repeat embeddings across batch
            embeddings = embeddings[None].repeat(
                batch_size, 1, 1, 1
            )  # batch_size x tiles_in x tiles_out x z_dim
            # repeat command across tiles
            r_command = command[:, None, None, :].repeat(1, tiling[0], tiling[1], 1)
            embeddings = embeddings.view(-1, embeddings.shape[-1])
            r_command = r_command.view(-1, r_command.shape[-1])
            w, b = hn(embeddings, r_command)
            if self.scale_layer_out:
                w = (
                    w
                    * self.scale_parameter
                    / torch.sqrt(torch.tensor([self.layer_sizes[i]]).float()).to(
                        w.device
                    )
                )
                b = (
                    b
                    * self.scale_parameter
                    / torch.sqrt(torch.tensor([self.layer_sizes[i]]).float()).to(
                        b.device
                    )
                )

            w = w.view(
                batch_size, tiling[0], tiling[1], hn.out_size_w[0], hn.out_size_w[1]
            ).permute(0, 2, 3, 1, 4)
            w = w.reshape(
                batch_size, tiling[1] * hn.out_size_w[0], tiling[0] * hn.out_size_w[1]
            )
            b = b.reshape(batch_size, tiling[0], tiling[1], hn.out_size_w[0]).mean(
                dim=1
            )
            b = b.view(batch_size, tiling[1] * hn.out_size_w[0])
            generated_parameters.extend([w, b])

        flat_parameters = [p.view(p.shape[0], -1) for p in generated_parameters]
        flat_parameters = torch.cat(flat_parameters, dim=1)

        generated_parameters = flat_parameters
        return generated_parameters


class PSSVF_linear(nn.Module):
    def __init__(self, parameter_space_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([parameter_space_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, parameters):
        return torch.squeeze(self.v_net(parameters), -1)


class Command(nn.Module):
    def __init__(self):
        super().__init__()
        self.command = torch.nn.Parameter((torch.as_tensor(0.0).float()))


class PSSVF(nn.Module):
    def __init__(
        self, obs_dim, num_probing_states, parameter_space_dim, hidden_sizes, activation
    ):
        super().__init__()

        self.probing_states = nn.ParameterList(
            [nn.Parameter(torch.rand([obs_dim])) for _ in range(num_probing_states)]
        )
        self.v_net = mlp([parameter_space_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, parameters, use_virtual_module=True, virtual_module=None):
        prob_sates = torch.stack(
            [
                torch.nn.utils.parameters_to_vector(state)
                for state in self.probing_states
            ]
        )
        actions = (
            virtual_module.forward(prob_sates, parameters)
            .transpose(0, 1)
            .reshape(parameters.shape[0], -1)
        )
        return torch.squeeze(self.v_net(actions), -1)


class MLPActorCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        n_probing_states,
        hidden_sizes_actor,
        activation,
        hidden_sizes_critic,
        device,
        critic,
        deterministic_actor,
    ):
        super().__init__()

        self.device = device
        self.deterministic_actor = deterministic_actor
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.act_dim = action_space.shape[0]
            act_limit = action_space.high[0]
            self.act_limit = act_limit
        elif isinstance(action_space, Discrete):
            self.act_dim = action_space.n

        self.pi = MLPActor(
            observation_space,
            action_space,
            hidden_sizes_actor,
            activation,
            device,
            deterministic_actor,
        ).to(device=device)

        named_params = self.pi.named_parameters()
        self.names, params = zip(*named_params)
        self.shapes = [param.shape for param in params]
        self.flat_shapes = [torch.flatten(param).shape for param in params]

        if critic:
            if isinstance(action_space, Box) and not deterministic_actor:
                # mean and sd of gaussian
                self.parameters_dim = n_probing_states * self.act_dim * 2
            else:
                self.parameters_dim = n_probing_states * self.act_dim

            self.v = PSSVF(
                obs_dim,
                n_probing_states,
                self.parameters_dim,
                hidden_sizes_critic,
                nn.ReLU,
            ).to(device=device)

    def act(self, obs, params, virtual_module=None):
        with torch.no_grad():
            a = virtual_module.forward(obs.unsqueeze(0), params)
            return a.to(device="cpu").numpy()


class DeterministicActorFunc(nn.Module):
    def __init__(self, act_limit, names, shapes, flat_shapes):
        super().__init__()
        self.act_limit = act_limit
        self.names = names
        self.shapes = shapes
        self.flat_shapes = flat_shapes

    def forward(self, x, params):
        temp = 0
        for idx, name in enumerate(self.names):
            if "weight" in name:
                weight = (
                    params[temp : temp + self.flat_shapes[idx][0]]
                    .unsqueeze(0)
                    .reshape(self.shapes[idx])
                )  # .reshape([0:self.shapes[idx][0], 0:self.shapes[idx][1]])
                temp += self.flat_shapes[idx][0]
            elif "bias" in name:
                bias = params[temp : temp + self.flat_shapes[idx][0]].reshape(
                    self.shapes[idx]
                )
                temp += self.flat_shapes[idx][0]
            else:
                raise ValueError
            if "bias" in name:
                x = F.linear(x, weight, bias)
                x = F.tanh(x)

        x = x * self.act_limit
        return x

    def get_probing_action(self, obs):
        return torch.tanh(self.pi_net(obs)) * self.act_limit


# class Generator_lin(nn.Module):
#     def __init__(self, parameter_space_dim, hidden_sizes, activation, scale_parameter, device):
#         super().__init__()
#         self.scale_parameter = scale_parameter
#         self.device = device
#         self.parameter_dim = parameter_space_dim
#         self.v_net = mlp([1] + list(hidden_sizes) + [parameter_space_dim], activation)
#
#     def forward(self, command, noise=None, evaluator=None, return_all=False, param=None):
#         parameters = torch.squeeze(self.v_net(command), -1)
#         parameters = parameters / self.scale_parameter
#
#         if evaluator is None:
#
#             if noise is not None:
#                 pi = self._distribution(parameters, std=torch.as_tensor(noise).float().to(self.device))
#
#                 dist = Normal(torch.zeros(self.parameter_dim), scale=1)
#                 delta = dist.sample().to(device=self.device, non_blocking=True).detach()
#                 parameters = parameters + noise * delta
#
#                 if param is not None:
#                     logp_a = self._log_prob_from_distribution(pi, param)
#                 else:
#                     #print(pi, parameters)
#                     logp_a = self._log_prob_from_distribution(pi, parameters)
#
#                 return parameters, logp_a
#             else:
#                 return parameters
#
#         else:
#             value = evaluator(parameters)
#             if return_all == False:
#                 return value
#             else:
#                 return value, parameters
#
#     def _distribution(self, parameters, std):
#         mu = parameters
#         return Normal(mu, std)
#
#     def _log_prob_from_distribution(self, pi, act):
#         return pi.log_prob(act).sum(axis=-1)


class Generator(nn.Module):
    def __init__(
        self,
        pi,
        parameter_dim,
        hidden_sizes,
        use_hyper,
        use_parallel,
        hid_size_w,
        hid_size_b,
        out_size_w,
        out_size_b,
        device,
        policy_neurons=None,
        scale_layer_out=False,
        scale_parameter=None,
    ):
        super().__init__()

        self.device = device
        self.parameter_dim = parameter_dim
        self.use_hyper = use_hyper
        self.scale_parameter = scale_parameter
        self.scale_layer_out = scale_layer_out

        print(
            "scale layer by layer:",
            self.scale_layer_out,
            "otherwise scale everything with scale",
            self.scale_parameter,
        )

        if policy_neurons is None:
            pi_shapes = [p.shape for p in pi.parameters()]
            policy_neurons = [
                pi_shapes[0][1],
                pi_shapes[2][1],
                pi_shapes[4][1],
                pi_shapes[4][0],
            ]

        self.encoder = get_hypernetwork_mlp_generator(
            policy_neurons,
            hidden_sizes,
            embedding_dim=hid_size_w,
            features_per_embedding=out_size_w[0],
            scale_layer_out=scale_layer_out,
            scale_parameter=scale_parameter,
        ).to(device)

    def forward(
        self,
        tot_reward,
        noise=None,
        use_virtual_module=True,
        evaluator=None,
        virtual_module=None,
        return_all=False,
        param=None,
    ):
        parameters = self.encoder(tot_reward)

        # print("shape", parameters.shape)
        if evaluator is None:
            if noise is not None:
                dist = Normal(torch.zeros(self.parameter_dim), scale=1)
                delta = dist.sample().to(device=self.device, non_blocking=True).detach()
                parameters = parameters + noise * delta
                logp_a = None

                return parameters, logp_a
            else:
                return parameters

        else:
            value = evaluator(
                parameters,
                use_virtual_module=use_virtual_module,
                virtual_module=virtual_module,
            )
            if return_all == False:
                return value
            else:
                return value, parameters

    def _distribution(self, parameters, std):
        mu = parameters
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPStochasticCategoricalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_net = mlp(pi_sizes, activation)

    def _distribution(self, obs):
        logits = self.pi_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs):
        pi = self._distribution(obs)
        return pi

    def get_probing_action(self, obs):
        pi = self._distribution(obs)
        return pi.probs


class MLPDeterministicCategoricalActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_net = mlp(pi_sizes, activation)
        self.softmax = nn.Softmax()

    def forward(self, obs):
        logits = self.pi_net(obs)
        out = self.softmax(logits)
        a = out.argmax()
        return a

    def get_probing_action(self, obs):
        logits = self.pi_net(obs)
        out = self.softmax(logits)
        a = out.argmax()
        return a


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_net = mlp(pi_sizes, activation)

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.act_limit = act_limit

    def _distribution(self, obs):
        mu = self.pi_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs):

        pi = self._distribution(obs)
        return pi

    def get_probing_action(self, obs):
        pi = self._distribution(obs)
        return torch.cat((pi.mean, pi.scale))


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi_net = mlp(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.pi_net(obs)

    def get_probing_action(self, obs):
        return torch.tanh(self.pi_net(obs)) * self.act_limit


class MLPActor(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes_actor,
        activation,
        device,
        deterministic_actor,
    ):
        super().__init__()

        self.act_limit = None
        self.deterministic_actor = deterministic_actor
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            act_dim = action_space.shape[0]
            act_limit = action_space.high[0]
            self.act_limit = act_limit
        elif isinstance(action_space, Discrete):
            act_dim = action_space.n

        if isinstance(action_space, Box):
            if deterministic_actor:
                self.pi = DeterministicActor(
                    obs_dim, act_dim, hidden_sizes_actor, activation, act_limit
                ).to(device=device)
            else:
                self.pi = MLPGaussianActor(
                    obs_dim, act_dim, hidden_sizes_actor, activation, act_limit
                ).to(device=device)

        elif isinstance(action_space, Discrete):
            if deterministic_actor:
                self.pi = MLPDeterministicCategoricalActor(
                    obs_dim, action_space.n, hidden_sizes_actor, activation
                ).to(device=device)
            else:
                self.pi = MLPStochasticCategoricalActor(
                    obs_dim, action_space.n, hidden_sizes_actor, activation
                ).to(device=device)

    def act(self, obs):
        with torch.no_grad():
            if self.deterministic_actor:
                a = self.pi(obs)
                a = (torch.tanh(a) * self.act_limit).to(device="cpu").numpy()
                return a
            else:
                pi = self.pi(obs)
                a = pi.sample()
                if isinstance(self.pi, MLPGaussianActor):
                    a = self.act_limit * torch.tanh(a).to(device="cpu").numpy()
                return a


class Statistics(object):
    def __init__(self, obs_dim):
        super().__init__()

        self.total_ts = 0
        self.episode = 0
        self.len_episode = 0
        self.rew_shaped_eval = 0
        self.rew_eval = 0
        self.rewards = []
        self.last_rewards = []
        self.position = 0
        self.n = 0
        self.mean = torch.zeros(obs_dim)
        self.mean_diff = torch.zeros(obs_dim)
        self.std = torch.zeros(obs_dim)
        self.command = 0
        self.last_rewards_env = []
        self.rewards_env = []
        self.position_env = 0
        self.max_rew = -np.inf
        self.min_rew = np.inf
        self.sim_time = 0
        self.up_policy_time = 0
        self.up_v_time = 0
        self.total_time = 0
        self.gen_time = 0
        self.max_pred = -np.inf

    def push_obs(self, obs):
        self.n += 1.0
        last_mean = self.mean
        self.mean += (obs - self.mean) / self.n
        self.mean_diff += (obs - last_mean) * (obs - self.mean)
        var = self.mean_diff / (self.n - 1) if self.n > 1 else np.square(self.mean)
        self.std = np.sqrt(var)
        return

    def push_rew(self, rew):
        if len(self.last_rewards) < 20:
            self.last_rewards.append(rew)
        else:
            self.last_rewards[self.position] = rew
            self.position = (self.position + 1) % 20
        self.rewards.append(rew)

    def push_rew_env(self, rew):
        if len(self.last_rewards_env) < 20:
            self.last_rewards_env.append(rew)
        else:
            self.last_rewards_env[self.position_env] = rew
            self.position_env = (self.position_env + 1) % 20
        self.rewards_env.append(rew)

    def normalize(self, obs):
        return (obs - self.mean) / (self.std + 1e-8)


class Buffer(object):
    def __init__(self, size_buffer, scale=1.0):
        self.history = []
        self.size_buffer = size_buffer
        self.scale = scale

    def sample_replay(self, batch_size, weighted_sampling=False):

        if weighted_sampling:
            self.weights = list(
                np.reciprocal(np.arange(1, len(self.history) + 1, dtype=float))
            )
            self.weights.reverse()
            self.weights = np.array(self.weights) ** self.scale
            self.weights = list(self.weights)
            sampled_hist = random.choices(
                self.history,
                weights=self.weights,
                k=min(int(batch_size), len(self.history)),
            )
        else:
            sampled_hist = random.sample(
                self.history, min(int(batch_size), len(self.history))
            )
        if len(self.history) > self.size_buffer:
            self.history.pop(0)
        return sampled_hist


class BufferTD(object):
    def __init__(self, capacity):
        self.history = []
        self.capacity = capacity
        self.position = 0

    def push(self, transition):
        if len(self.history) < self.capacity:
            self.history.append(transition)
        else:
            self.history[self.position] = transition
            self.position = (self.position + 1) % self.capacity

    def sample_replay_td(self, batch_size):

        sampled_trans = random.choices(self.history, k=int(batch_size))
        return sampled_trans


def grad_norm(parameters):
    # Compute the norm of the gradient
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(2)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def norm(parameters):
    # Compute the norm of the weights of a model
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(2)
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
