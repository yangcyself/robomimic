"""
Implementation of YCY: Haven't come up with a good name yet.
"""
from collections import OrderedDict, deque
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
from robomimic.models.obs_nets import MIMO_TRANSENCODER, MIMO_TRANSFORM_DECODER


import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, ActionChunkingAlgo
import torchvision.transforms as transforms
from torch.autograd import Variable

@register_algo_factory_func("ycy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the ycy algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    return YCY, {}


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class YCY(ActionChunkingAlgo):
    """
    Normal ACT training.
    """

    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        self.optim_params = deepcopy(algo_config.optim_params)
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config
        self.latent_dim = algo_config.latent_dim

        self.ac_dim = ac_dim
        self.device = device
        self.obs_key_shapes = obs_key_shapes

        self.query_frequency = algo_config.rollout.query_frequency
        self.action_space_normalizer = self.algo_config.action_space_normalizer

        self.nets = nn.ModuleDict()

        self._update_utils()
        self._create_shapes(obs_config.action_encoder.modalities, obs_config.actor.modalities, obs_key_shapes)
        self._create_networks()
        self._create_optimizers()
        assert isinstance(self.nets, nn.ModuleDict)

    def _update_utils(self):
        """Update the robotmimic systems such as ObsUtil. OBS_KEYS_TO_MODALITIES
        """
        act_internal_spec = {
            "seq:actions":{ # the action sequence to action_encoder
                "low_dim" : ["pad_mask"],
            },
            "latent":{
                "low_dim" : ["style"],
            }
        }
        ObsUtils.update_obs_utils_with_obs_specs(obs_modality_specs=act_internal_spec)

    def _create_shapes(self, encoder_obs_keys, actor_obs_keys, obs_key_shapes):
        """
        Create encoder_obs_group_shapes and actor_obs_group_shapes dictionaries, to make it
        easy for this algorithm object to keep track of observation key shapes. 
        Each dictionary maps observation key to shape.

        Args:
            obs_keys (dict): dict of required observation keys for this training run (usually
                specified by the obs config), e.g., {"joints": ["xxx", "xxx"], "cams": ["proxxx_image"]}
            obs_key_shapes (dict): dict of observation key shapes, e.g., {"rgb": [3, 224, 224]}
        """
        # determine shapes
        self.actor_obs_group_shapes = OrderedDict()
        # We keep the "seq:" at the end of the OrderedDict
        self.actor_obs_group_shapes["latent"] = OrderedDict(style=[self.latent_dim])
        self.encoder_obs_group_shapes = OrderedDict()
        # We check across all modalitie specified in the config. store its corresponding shape internally
        for k in obs_key_shapes:
            for group, modality in encoder_obs_keys.items():
                modality_obs = [v for vv in modality.values() for v in vv] # flatten, vv are [low_dim, rgb, ...]
                if k in modality_obs:
                    if group not in self.encoder_obs_group_shapes:
                        self.encoder_obs_group_shapes[group] = OrderedDict()
                    if(group.startswith("seq:")):
                        self.encoder_obs_group_shapes[group][k] = [-self.algo_config.max_len] + obs_key_shapes[k]
                    else:
                        self.encoder_obs_group_shapes[group][k] = obs_key_shapes[k]
            for group, modality in actor_obs_keys.items():
                modality_obs = [v for vv in modality.values() for v in vv] # flatten
                if k in modality_obs:
                    if group not in self.actor_obs_group_shapes:
                        self.actor_obs_group_shapes[group] = OrderedDict()
                    if(group.startswith("seq:")):
                        self.actor_obs_group_shapes[group][k] = [-self.algo_config.max_len] + obs_key_shapes[k]
                    else:
                        self.actor_obs_group_shapes[group][k] = obs_key_shapes[k]
        
        self.encoder_obs_group_shapes["seq:actions"] = OrderedDict(actions=[-self.algo_config.max_len, self.ac_dim])

        self.all_obs_modalities = deepcopy(self.obs_config.action_encoder.modalities)
        self.all_obs_modalities.update(self.obs_config.actor.modalities)


    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        self.nets = nn.ModuleDict()

        encoder = MIMO_TRANSENCODER(
            input_obs_group_shapes = self.encoder_obs_group_shapes,
            output_shapes = OrderedDict(
                latent = (1, OrderedDict(
                    info = self.latent_dim*2
                )),
            ),
            **self.algo_config.encoder,
            encoder_kwargs=self.obs_config.actor.encoder
        )

        decoder = MIMO_TRANSFORM_DECODER(
            input_obs_group_shapes = self.actor_obs_group_shapes,
            output_shapes = OrderedDict(
                hat = (-1, OrderedDict(
                    action = self.ac_dim,
                    is_pad = 1
                )),
            ),
            **self.algo_config.transformer,
            encoder_kwargs=self.obs_config.actor.encoder
        )

        n_parameters = sum(p.numel() for model in [encoder, decoder] for p in model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters/1e6,))

        self.nets["encoder"] = encoder
        self.nets["decoder"] = decoder # CVAE decoder
        self.nets = self.nets.to(self.device)
        self.imgnormalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """
        input_batch = OrderedDict()

        for group, modalities in self.all_obs_modalities.items():
            if(group.startswith("seq:")):
                input_batch[group] = OrderedDict({
                    k:batch["obs"][k][:,:,...] 
                    for kk in modalities.values()
                    for k in kk
                })
                input_batch[group]["is_pad"] = torch.full( #TODO: also return pad_mask from dataset
                    tuple(input_batch[group].values())[0].shape[:2], False, dtype=torch.bool, device=self.device
                )
            else:
                input_batch[group] = OrderedDict({
                    k:batch["obs"][k][:,0,...] 
                    for kk in modalities.values()
                    for k in kk
                })

        input_batch["seq:actions"] = OrderedDict(
            actions = self._pre_action(batch["actions"]), # batch_size, seq_length, 7
            is_pad = ~batch["pad_mask"][:,:,0]
        )
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(YCY, self).train_on_batch(batch, epoch, validate=validate)
            predictions = self._forward_training(batch)
            losses = self._compute_losses(predictions, batch)

            info["predictions"] = TensorUtils.detach(predictions)
            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                step_info = self._train_step(losses)
                info.update(step_info)
        return info

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """


        # project action sequence to embedding dim, and concat with a CLS token
        # query model
        encoder_output = self.nets["encoder"](**batch)
        latent_info = encoder_output["latent_info"]
        mu = latent_info[:, :, :self.latent_dim].squeeze(1)
        logvar = latent_info[:, :, self.latent_dim:].squeeze(1)
        latent_sample = reparametrize(mu, logvar)

        decoder_input = batch
        decoder_input["latent"] = OrderedDict(style=latent_sample)
        predictions = self.nets["decoder"](**decoder_input)
        predictions["mu"] = mu
        predictions["logvar"] = logvar

        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        total_kld, dim_wise_kld, mean_kld = kl_divergence(predictions["mu"], predictions["logvar"])

        actions = batch["seq:actions"]["actions"]
        is_pad = batch["seq:actions"]["is_pad"].to(dtype=torch.bool)
        # Assume a_hat is at the last of the sequence from the decoder prediction
        a_hat = predictions["hat_action"][...,-actions.shape[-2]:,:]
        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.algo_config.loss.kl_weight
        return loss_dict

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """

        # gradient step
        info = OrderedDict()
        # TODO Change to use backprop_for_loss
        self.optimizers["encoder"].zero_grad()
        self.optimizers["decoder"].zero_grad()
        losses["loss"].backward()
        encoder_grad_norms = 0.
        for p in self.nets["encoder"].parameters():
            # only clip gradients for parameters for which requires_grad is True
            if p.grad is not None:
                encoder_grad_norms += p.grad.data.norm(2).pow(2).item()
        decoder_grad_norms = 0.
        for p in self.nets["decoder"].parameters():
            # only clip gradients for parameters for which requires_grad is True
            if p.grad is not None:
                decoder_grad_norms += p.grad.data.norm(2).pow(2).item()

        self.optimizers["encoder"].step()
        self.optimizers["decoder"].step()
        info["encoder_grad_norms"] = encoder_grad_norms
        info["decoder_grad_norms"] = decoder_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(YCY, self).log_info(info)
        log["Loss"] = info["losses"]["loss"].item()
        if "l1" in info["losses"]:
            log["l1"] = info["losses"]["l1"].item()
        if "kl" in info["losses"]:
            log["kl"] = info["losses"]["kl"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training
        input_batch = {}
        for group, modalities in self.obs_config.actor.modalities.items():
            input_batch[group] = OrderedDict({
                k:obs_dict[k].unsqueeze(1)
                for kk in modalities.values()
                for k in kk
            })
            if len(input_batch[group])==0:
                continue
            input_batch[group]["is_pad"] = torch.full(
                tuple(input_batch[group].values())[0].shape[:2], False, dtype=torch.bool, device=self.device
            )
        predictions = self.nets["decoder"](seq_step = self._step_count, incremental_state = self._incremental_state, **input_batch)
        action = predictions["hat_action"][:,-1,:]
        self._step_count += 1
        return self._post_action(action)


    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._incremental_state = {}
        self._step_count = 0
        ## forward in the decoder to push the zero latent first
        self.nets["decoder"](
            incremental_state=self._incremental_state,
            latent=OrderedDict(
                style=torch.zeros(1, 1, self.latent_dim, device=self.device)
            )
        )


    def _update_stack_obs(self, k, obs):
        self._stacked_obs[k].push(obs)
        return self._stacked_obs[k].get_elements()
        
    def _pre_action(self, action):
        if(self.action_space_normalizer is not None):
            act_mean = self.action_space_normalizer.get("mean", 0.0)
            act_std = self.action_space_normalizer.get("std", 1.0)
            if(type(act_mean) == list):
                act_mean = torch.tensor(act_mean, dtype=torch.float32, device=action.device)
            if(type(act_std) == list):
                act_std = torch.tensor(act_std, dtype=torch.float32, device=action.device)
            action = (action - act_mean)/act_std
        return action
    
    def _post_action(self, action):
        if(self.action_space_normalizer is not None):
            act_mean = self.action_space_normalizer.get("mean", 0.0)
            act_std = self.action_space_normalizer.get("std", 1.0)
            if(type(act_mean) == list):
                act_mean = torch.tensor(act_mean, dtype=torch.float32, device=action.device)
            if(type(act_std) == list):
                act_std = torch.tensor(act_std, dtype=torch.float32, device=action.device)
            action = action * act_std + act_mean
        return action

