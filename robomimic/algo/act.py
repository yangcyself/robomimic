"""
Implementation of ACT: Action Chunking with Transformers
"""
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
from robomimic.models.policy_nets import DETRVAEActor

from robomimic.models.transformer import TransformerEncoder, TransformerEncoderLayer
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, ActionChunkingAlgo
import torchvision.transforms as transforms


@register_algo_factory_func("act")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the act algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    return ACT, {}


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


class ACT(ActionChunkingAlgo):
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

        self.ac_dim = ac_dim
        self.device = device
        self.obs_key_shapes = obs_key_shapes

        self.nets = nn.ModuleDict()

        self._create_shapes(obs_config.action_encoder.modalities, obs_config.actor.modalities, obs_key_shapes)
        self._create_networks()
        self._create_optimizers()
        assert isinstance(self.nets, nn.ModuleDict)


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
        self.encoder_obs_group_shapes = OrderedDict()
        # We check across all modalitie specified in the config. store its corresponding shape internally
        for k in obs_key_shapes:
            for group, modality in encoder_obs_keys.items():
                modality_obs = [v for vv in modality.values() for v in vv] # flatten, vv are [low_dim, rgb, ...]
                if k in modality_obs:
                    if group not in self.encoder_obs_group_shapes:
                        self.encoder_obs_group_shapes[group] = OrderedDict()
                    self.encoder_obs_group_shapes[group][k] = obs_key_shapes[k]
            for group, modality in actor_obs_keys.items():
                modality_obs = [v for vv in modality.values() for v in vv] # flatten
                if k in modality_obs:
                    if group not in self.actor_obs_group_shapes:
                        self.actor_obs_group_shapes[group] = OrderedDict()
                    self.actor_obs_group_shapes[group][k] = obs_key_shapes[k]
        
        self.encoder_obs_group_shapes["seq:actions"] = OrderedDict(actions=[self.algo_config.chunk_size, self.ac_dim])
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.actor_obs_group_shapes["latent"] = OrderedDict(style=[self.latent_dim])


    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        self.nets = nn.ModuleDict()

        # From state
        # backbone = None # from state for now, no need for conv nets
        # From image
        # encoder = build_encoder(self.algo_config.encoder)
        model = DETRVAEActor(
            self.encoder_obs_group_shapes,
            self.actor_obs_group_shapes,
            latent_dim = self.latent_dim,
            action_dim = self.ac_dim,
            num_queries=self.algo_config.chunk_size,
            encoder_kwargs=self.algo_config.encoder,
            transformer_kwargs=self.algo_config.transformer,
            backbone_kwargs=self.algo_config.backbone,
            obs_encoder_kwargs=self.obs_config.actor.encoder
        )

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters/1e6,))

        self.nets["policy"] = model # CVAE decoder
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
        input_batch["cams"] = OrderedDict({k:v[:,0,:,:,:] for k,v in batch["obs"].items() if "_image" in k})
        input_batch["joints"] = OrderedDict({k:v[:,0,...] for k,v in batch["obs"].items() if k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]})
        input_batch["seq:actions"] = OrderedDict(
            actions = batch["actions"], # batch_size, seq_length, 7
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
            info = super(ACT, self).train_on_batch(batch, epoch, validate=validate)
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

        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](batch, True)
        predictions = OrderedDict(
            a_hat=a_hat,
            is_pad_hat=is_pad_hat,
            mu=mu,
            logvar=logvar,
        )
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
        actions = actions[:, :self.nets["policy"].num_queries]
        is_pad = batch["seq:actions"]["is_pad"].to(dtype=torch.bool)
        is_pad = is_pad[:, :self.nets["policy"].num_queries]

        loss_dict = dict()
        all_l1 = F.l1_loss(actions, predictions['a_hat'], reduction='none')
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
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["loss"],
        )
        info["policy_grad_norms"] = policy_grad_norms
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
        log = super(ACT, self).log_info(info)
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

        input_batch = OrderedDict()
        input_batch["cams"] = OrderedDict({k:v for k,v in obs_dict.items() if "_image" in k})
        input_batch["joints"] = OrderedDict({k:v for k,v in obs_dict.items() if k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]})

        a_hat,_,_ = self.nets["policy"](input_batch) # no action, sample from prior
        return a_hat[:,0,:] # TODO, average chunk this

