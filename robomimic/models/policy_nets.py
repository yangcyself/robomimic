"""
Contains torch Modules for policy networks. These networks take an
observation dictionary as input (and possibly additional conditioning,
such as subgoal or goal dictionaries) and produce action predictions,
samples, or distributions as outputs. Note that actions
are assumed to lie in [-1, 1], and most networks will have a final
tanh activation to help ensure this range.
"""
import textwrap
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.base_nets import Module
from robomimic.models.obs_nets import MIMO_MLP, RNN_MIMO_MLP
from robomimic.models.vae_nets import VAE
from robomimic.models.distributions import TanhWrappedDistribution


class ActorNetwork(MIMO_MLP):
    """
    A basic policy network that predicts actions from observations.
    Can optionally be goal conditioned on future observations.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            goal_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-observation key information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim

        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(ActorNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @MIMO_MLP, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        return OrderedDict(action=(self.ac_dim,))

    def output_shape(self, input_shape=None):
        return [self.ac_dim]

    def forward(self, obs_dict, goal_dict=None):
        actions = super(ActorNetwork, self).forward(obs=obs_dict, goal=goal_dict)["action"]
        # apply tanh squashing to ensure actions are in [-1, 1]
        return torch.tanh(actions)

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)


class PerturbationActorNetwork(ActorNetwork):
    """
    An action perturbation network - primarily used in BCQ.
    It takes states and actions and returns action perturbations.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        perturbation_scale=0.05,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps observation keys to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            perturbation_scale (float): the perturbation network output is always squashed to 
                lie in +/- @perturbation_scale. The final action output is equal to the original 
                input action added to the output perturbation (and clipped to lie in [-1, 1]).

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.perturbation_scale = perturbation_scale

        # add in action as a modality
        new_obs_shapes = OrderedDict(obs_shapes)
        new_obs_shapes["action"] = (ac_dim,)

        # pass to super class to instantiate network
        super(PerturbationActorNetwork, self).__init__(
            obs_shapes=new_obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def forward(self, obs_dict, acts, goal_dict=None):
        """Forward pass through perturbation actor."""
        # add in actions
        inputs = dict(obs_dict)
        inputs["action"] = acts
        perturbations = super(PerturbationActorNetwork, self).forward(inputs, goal_dict)

        # add perturbations from network to original actions, and ensure the new actions lie in [-1, 1]
        output_actions = acts + self.perturbation_scale * perturbations
        output_actions = output_actions.clamp(-1.0, 1.0)
        return output_actions

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}, perturbation_scale={}".format(self.ac_dim, self.perturbation_scale)


class GaussianActorNetwork(ActorNetwork):
    """
    Variant of actor network that learns a diagonal unimodal Gaussian distribution
    over actions.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        fixed_std=False,
        std_activation="softplus",
        init_last_fc_weight=None,
        init_std=0.3,
        mean_limits=(-9.0, 9.0),
        std_limits=(0.007, 7.5),
        low_noise_eval=True,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            fixed_std (bool): if True, std is not learned, but kept constant at @init_std

            std_activation (None or str): type of activation to use for std deviation. Options are:

                None: no activation applied (not recommended unless using fixed std)

                `'softplus'`: Only applicable if not using fixed std. Softplus activation applied, after which the
                    output is scaled by init_std / softplus(0)

                `'exp'`: Only applicable if not using fixed std. Exp applied; this corresponds to network output
                    as being interpreted as log_std instead of std

                NOTE: In all cases, the final result is clipped to be within @std_limits

            init_last_fc_weight (None or float): if specified, will intialize the final layer network weights to be
                uniformly sampled from [-init_weight, init_weight]

            init_std (None or float): approximate initial scaling for standard deviation outputs
                from network. If None

            mean_limits (2-array): (min, max) to clamp final mean output by

            std_limits (2-array): (min, max) to clamp final std output by

            low_noise_eval (float): if True, model will output means of Gaussian distribution
                at eval time.

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to Gaussian actor
        self.fixed_std = fixed_std
        self.init_std = init_std
        self.mean_limits = np.array(mean_limits)
        self.std_limits = np.array(std_limits)

        # Define activations to use
        def softplus_scaled(x):
            out = F.softplus(x)
            out = out * (self.init_std / F.softplus(torch.zeros(1).to(x.device)))
            return out

        self.activations = {
            None: lambda x: x,
            "softplus": softplus_scaled,
            "exp": torch.exp,
        }
        assert std_activation in self.activations, \
            "std_activation must be one of: {}; instead got: {}".format(self.activations.keys(), std_activation)
        self.std_activation = std_activation if not self.fixed_std else None

        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        super(GaussianActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # If initialization weight was specified, make sure all final layer network weights are specified correctly
        if init_last_fc_weight is not None:
            with torch.no_grad():
                for name, layer in self.nets["decoder"].nets.items():
                    torch.nn.init.uniform_(layer.weight, -init_last_fc_weight, init_last_fc_weight)
                    torch.nn.init.uniform_(layer.bias, -init_last_fc_weight, init_last_fc_weight)

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of Gaussian distribution.
        """
        return OrderedDict(
            mean=(self.ac_dim,), 
            scale=(self.ac_dim,),
        )

    def forward_train(self, obs_dict, goal_dict=None):
        """
        Return full Gaussian distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL 
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            dist (Distribution): Gaussian distribution
        """
        out = MIMO_MLP.forward(self, obs=obs_dict, goal=goal_dict)
        mean = out["mean"]
        # Use either constant std or learned std depending on setting
        scale = out["scale"] if not self.fixed_std else torch.ones_like(mean) * self.init_std

        # Clamp the mean
        mean = torch.clamp(mean, min=self.mean_limits[0], max=self.mean_limits[1])

        # apply tanh squashing to mean if not using tanh-Gaussian to ensure mean is in [-1, 1]
        if not self.use_tanh:
            mean = torch.tanh(mean)

        # Calculate scale
        if self.low_noise_eval and (not self.training):
            # override std value so that you always approximately sample the mean
            scale = torch.ones_like(mean) * 1e-4
        else:
            # Post-process the scale accordingly
            scale = self.activations[self.std_activation](scale)
            # Clamp the scale
            scale = torch.clamp(scale, min=self.std_limits[0], max=self.std_limits[1])


        # the Independent call will make it so that `batch_shape` for dist will be equal to batch size
        # while `event_shape` will be equal to action dimension - ensuring that log-probability 
        # computations are summed across the action dimension
        dist = D.Normal(loc=mean, scale=scale)
        dist = D.Independent(dist, 1)

        if self.use_tanh:
            # Wrap distribution with Tanh
            dist = TanhWrappedDistribution(base_dist=dist, scale=1.)

        return dist

    def forward(self, obs_dict, goal_dict=None):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        dist = self.forward_train(obs_dict, goal_dict)
        if self.low_noise_eval and (not self.training):
            if self.use_tanh:
                # # scaling factor lets us output actions like [-1. 1.] and is consistent with the distribution transform
                # return (1. + 1e-6) * torch.tanh(dist.base_dist.mean)
                return torch.tanh(dist.mean)
            return dist.mean
        return dist.sample()

    def _to_string(self):
        """Info to pretty print."""
        msg = "action_dim={}\nfixed_std={}\nstd_activation={}\ninit_std={}\nmean_limits={}\nstd_limits={}\nlow_noise_eval={}".format(
            self.ac_dim, self.fixed_std, self.std_activation, self.init_std, self.mean_limits, self.std_limits, self.low_noise_eval)
        return msg


class GMMActorNetwork(ActorNetwork):
    """
    Variant of actor network that learns a multimodal Gaussian mixture distribution
    over actions.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        num_modes=5,
        min_std=0.01,
        std_activation="softplus",
        low_noise_eval=True,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert std_activation in self.activations, \
            "std_activation must be one of: {}; instead got: {}".format(self.activations.keys(), std_activation)
        self.std_activation = std_activation

        super(GMMActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            mean=(self.num_modes, self.ac_dim), 
            scale=(self.num_modes, self.ac_dim), 
            logits=(self.num_modes,),
        )

    def forward_train(self, obs_dict, goal_dict=None):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL 
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            dist (Distribution): GMM distribution
        """
        out = MIMO_MLP.forward(self, obs=obs_dict, goal=goal_dict)
        means = out["mean"]
        scales = out["scale"]
        logits = out["logits"]

        # apply tanh squashing to means if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        # Calculate scale
        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1)

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dist = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            # Wrap distribution with Tanh
            dist = TanhWrappedDistribution(base_dist=dist, scale=1.)

        return dist

    def forward(self, obs_dict, goal_dict=None):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        dist = self.forward_train(obs_dict, goal_dict)
        return dist.sample()

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}\nnum_modes={}\nmin_std={}\nstd_activation={}\nlow_noise_eval={}".format(
            self.ac_dim, self.num_modes, self.min_std, self.std_activation, self.low_noise_eval)


class RNNActorNetwork(RNN_MIMO_MLP):
    """
    An RNN policy network that predicts actions from observations.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.ac_dim = ac_dim

        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes

        # set up different observation groups for @RNN_MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(RNNActorNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            mlp_layer_dims=mlp_layer_dims,
            mlp_activation=nn.ReLU,
            mlp_layer_func=nn.Linear,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            per_step=True,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @RNN_MIMO_MLP, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        return OrderedDict(action=(self.ac_dim,))

    def output_shape(self, input_shape):
        # note: @input_shape should be dictionary (key: mod)
        # infers temporal dimension from input shape
        mod = list(self.obs_shapes.keys())[0]
        T = input_shape[mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="RNNActorNetwork: input_shape inconsistent in temporal dimension")
        return [T, self.ac_dim]

    def forward(self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False):
        """
        Forward a sequence of inputs through the RNN and the per-step network.

        Args:
            obs_dict (dict): batch of observations - each tensor in the dictionary
                should have leading dimensions batch and time [B, T, ...]
            goal_dict (dict): if not None, batch of goal observations
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state

        Returns:
            actions (torch.Tensor): predicted action sequence
            rnn_state: return rnn state at the end if return_state is set to True
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)

        outputs = super(RNNActorNetwork, self).forward(
            obs=obs_dict, goal=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state)

        if return_state:
            actions, state = outputs
        else:
            actions = outputs
            state = None
        
        # apply tanh squashing to ensure actions are in [-1, 1]
        actions = torch.tanh(actions["action"])

        if return_state:
            return actions, state
        else:
            return actions

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            actions (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        action, state = self.forward(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True)
        return action[:, 0], state

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)


class RNNGMMActorNetwork(RNNActorNetwork):
    """
    An RNN GMM policy network that predicts sequences of action distributions from observation sequences.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        num_modes=5,
        min_std=0.01,
        std_activation="softplus",
        low_noise_eval=True,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert std_activation in self.activations, \
            "std_activation must be one of: {}; instead got: {}".format(self.activations.keys(), std_activation)
        self.std_activation = std_activation

        super(RNNGMMActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            mlp_layer_dims=mlp_layer_dims,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            mean=(self.num_modes, self.ac_dim), 
            scale=(self.num_modes, self.ac_dim), 
            logits=(self.num_modes,),
        )

    def forward_train(self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL 
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
            rnn_state: return rnn state at the end if return_state is set to True
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)

        outputs = RNN_MIMO_MLP.forward(
            self, obs=obs_dict, goal=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state)

        if return_state:
            outputs, state = outputs
        else:
            state = None
        
        means = outputs["mean"]
        scales = outputs["scale"]
        logits = outputs["logits"]

        # apply tanh squashing to mean if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, timesteps, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(component_distribution, 1) # shift action dim to event shape

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            # Wrap distribution with Tanh
            dists = TanhWrappedDistribution(base_dist=dists, scale=1.)

        if return_state:
            return dists, state
        else:
            return dists

    def forward(self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out = self.forward_train(obs_dict=obs_dict, goal_dict=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state)
        if return_state:
            ad, state = out
            return ad.sample(), state
        return out.sample()

    def forward_train_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get action GMM distribution, which 
        is useful for computing quantities necessary at train-time, like 
        log-likelihood, KL divergence, etc.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            ad (Distribution): GMM action distributions
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        ad, state = self.forward_train(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True)

        # to squeeze time dimension, make another action distribution
        assert ad.component_distribution.base_dist.loc.shape[1] == 1
        assert ad.component_distribution.base_dist.scale.shape[1] == 1
        assert ad.mixture_distribution.logits.shape[1] == 1
        component_distribution = D.Normal(
            loc=ad.component_distribution.base_dist.loc.squeeze(1),
            scale=ad.component_distribution.base_dist.scale.squeeze(1),
        )
        component_distribution = D.Independent(component_distribution, 1)
        mixture_distribution = D.Categorical(logits=ad.mixture_distribution.logits.squeeze(1))
        ad = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )
        return ad, state

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get sampled actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            acts (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        acts, state = self.forward(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True)
        assert acts.shape[1] == 1
        return acts[:, 0], state

    def _to_string(self):
        """Info to pretty print."""
        msg = "action_dim={}, std_activation={}, low_noise_eval={}, num_nodes={}, min_std={}".format(
            self.ac_dim, self.std_activation, self.low_noise_eval, self.num_modes, self.min_std)
        return msg


class VAEActor(Module):
    """
    A VAE that models a distribution of actions conditioned on observations.
    The VAE prior and decoder are used at test-time as the policy.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        encoder_layer_dims,
        decoder_layer_dims,
        latent_dim,
        device,
        decoder_is_conditioned=True,
        decoder_reconstruction_sum_across_elements=False,
        latent_clip=None,
        prior_learn=False,
        prior_is_conditioned=False,
        prior_layer_dims=(),
        prior_use_gmm=False,
        prior_gmm_num_modes=10,
        prior_gmm_learn_weights=False,
        prior_use_categorical=False,
        prior_categorical_dim=10,
        prior_categorical_gumbel_softmax_hard=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(VAEActor, self).__init__()

        self.obs_shapes = obs_shapes
        self.ac_dim = ac_dim
        action_shapes = OrderedDict(action=(self.ac_dim,))

        # ensure VAE decoder will squash actions into [-1, 1]
        output_squash = ['action']
        output_scales = OrderedDict(action=1.)

        self._vae = VAE(
            input_shapes=action_shapes,
            output_shapes=action_shapes,
            encoder_layer_dims=encoder_layer_dims,
            decoder_layer_dims=decoder_layer_dims,
            latent_dim=latent_dim,
            device=device,
            condition_shapes=self.obs_shapes,
            decoder_is_conditioned=decoder_is_conditioned,
            decoder_reconstruction_sum_across_elements=decoder_reconstruction_sum_across_elements,
            latent_clip=latent_clip,
            output_squash=output_squash,
            output_scales=output_scales,
            prior_learn=prior_learn,
            prior_is_conditioned=prior_is_conditioned,
            prior_layer_dims=prior_layer_dims,
            prior_use_gmm=prior_use_gmm,
            prior_gmm_num_modes=prior_gmm_num_modes,
            prior_gmm_learn_weights=prior_gmm_learn_weights,
            prior_use_categorical=prior_use_categorical,
            prior_categorical_dim=prior_categorical_dim,
            prior_categorical_gumbel_softmax_hard=prior_categorical_gumbel_softmax_hard,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def encode(self, actions, obs_dict, goal_dict=None):
        """
        Args:
            actions (torch.Tensor): a batch of actions

            obs_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to the observation modalities 
                used for conditioning in either the decoder or the prior (or both).

            goal_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities.

        Returns:
            posterior params (dict): dictionary with the following keys:

                mean (torch.Tensor): posterior encoder means

                logvar (torch.Tensor): posterior encoder logvars
        """
        inputs = OrderedDict(action=actions)
        return self._vae.encode(inputs=inputs, conditions=obs_dict, goals=goal_dict)

    def decode(self, obs_dict=None, goal_dict=None, z=None, n=None):
        """
        Thin wrapper around @VaeNets.VAE implementation.

        Args:
            obs_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. Only needs to be provided if @decoder_is_conditioned
                or @z is None (since the prior will require it to generate z).

            goal_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities.

            z (torch.Tensor): if provided, these latents are used to generate
                reconstructions from the VAE, and the prior is not sampled.

            n (int): this argument is used to specify the number of samples to 
                generate from the prior. Only required if @z is None - i.e.
                sampling takes place

        Returns:
            recons (dict): dictionary of reconstructed inputs (this will be a dictionary
                with a single "action" key)
        """
        return self._vae.decode(conditions=obs_dict, goals=goal_dict, z=z, n=n)

    def sample_prior(self, obs_dict=None, goal_dict=None, n=None):
        """
        Thin wrapper around @VaeNets.VAE implementation.

        Args:
            n (int): this argument is used to specify the number
                of samples to generate from the prior.

            obs_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. Only needs to be provided if @prior_is_conditioned.

            goal_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities.

        Returns:
            z (torch.Tensor): latents sampled from the prior
        """
        return self._vae.sample_prior(n=n, conditions=obs_dict, goals=goal_dict)

    def set_gumbel_temperature(self, temperature):
        """
        Used by external algorithms to schedule Gumbel-Softmax temperature,
        which is used during reparametrization at train-time. Should only be
        used if @prior_use_categorical is True.
        """
        self._vae.set_gumbel_temperature(temperature)

    def get_gumbel_temperature(self):
        """
        Return current Gumbel-Softmax temperature. Should only be used if
        @prior_use_categorical is True.
        """
        return self._vae.get_gumbel_temperature()

    def output_shape(self, input_shape=None):
        """
        This implementation is required by the Module superclass, but is unused since we 
        never chain this module to other ones.
        """
        return [self.ac_dim]

    def forward_train(self, actions, obs_dict, goal_dict=None, freeze_encoder=False):
        """
        A full pass through the VAE network used during training to construct KL
        and reconstruction losses. See @VAE class for more info.

        Args:
            actions (torch.Tensor): a batch of actions

            obs_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to the observation modalities 
                used for conditioning in either the decoder or the prior (or both).

            goal_dict (dict): a dictionary that maps modalities to torch.Tensor
                batches. These should correspond to goal modalities.

        Returns:
            vae_outputs (dict): a dictionary that contains the following outputs.

                encoder_params (dict): parameters for the posterior distribution
                    from the encoder forward pass

                encoder_z (torch.Tensor): latents sampled from the encoder posterior

                decoder_outputs (dict): action reconstructions from the decoder

                kl_loss (torch.Tensor): KL loss over the batch of data

                reconstruction_loss (torch.Tensor): reconstruction loss over the batch of data
        """
        action_inputs = OrderedDict(action=actions)
        return self._vae.forward(
            inputs=action_inputs, 
            outputs=action_inputs, 
            conditions=obs_dict, 
            goals=goal_dict,
            freeze_encoder=freeze_encoder)

    def forward(self, obs_dict, goal_dict=None, z=None):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations
            z (torch.Tensor): if not None, use the provided batch of latents instead
                of sampling from the prior

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        n = None
        if z is None:
            # prior will be sampled - so we must provide number of samples explicitly
            mod = list(obs_dict.keys())[0]
            n = obs_dict[mod].shape[0]
        return self.decode(obs_dict=obs_dict, goal_dict=goal_dict, z=z, n=n)["action"]

from torch.autograd import Variable # TODO Reorganize

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class DETRVAEActor(Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, encoder, state_dim, action_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            action_dim: robot action dimension 
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbone is not None:
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.backbone = backbone
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbone = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_state_proj = nn.Linear(state_dim, hidden_dim) # project action to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_state_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbone is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbone(image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]
