"""
Implementation of ACT: Action Chunking with Transformers
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.models.base_nets as BaseNets
import robomimic.models.obs_nets as ObsNets
from robomimic.models.policy_nets import DETRVAEActor
from robomimic.models.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer
import robomimic.models.vae_nets as VAENets
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo


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


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder

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


class ACT(PolicyAlgo):
    """
    Normal ACT training.
    """
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """

        # PolicyNets.ActorNetwork(
        #     obs_shapes=self.obs_shapes,
        #     goal_shapes=self.goal_shapes,
        #     ac_dim=self.ac_dim,
        #     mlp_layer_dims=self.algo_config.actor_layer_dims,
        #     encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        # )
        self.nets = nn.ModuleDict()

        state_dim = 14 # TODO hardcode
        # From state
        # backbone = None # from state for now, no need for conv nets
        # From image
        backbone = build_backbone(args)
        transformer = build_transformer(args)
        encoder = build_encoder(args)
        model = DETRVAEActor(
            backbone,
            transformer,
            encoder,
            state_dim=state_dim,
            num_queries=args.num_queries,
            camera_names=args.camera_names,
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
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, 0, :]
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
        image = batch["obs"]["image"]
        image = self.imgnormalize(image)
        qpos = batch["obs"]["qpos"]
        actions = batch["actions"]
        is_pad = batch["is_pad"]
        actions = actions[:, :self.nets["policy"].num_queries]
        is_pad = is_pad[:, :self.nets["policy"].num_queries]

        a_hat, is_pad_hat, (mu, logvar) = self.nets["policy"](qpos, image, env_state, actions, is_pad)
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

        actions = batch["actions"]
        actions = actions[:, :self.nets["policy"].num_queries]
        is_pad = batch["is_pad"]
        is_pad = is_pad[:, :self.nets["policy"].num_queries]

        loss_dict = dict()
        all_l1 = F.l1_loss(actions, a_hat, reduction='none')
        l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
        loss_dict['l1'] = l1
        loss_dict['kl'] = total_kld[0]
        loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.algo_config.kl_weight
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

        image = batch["obs"]["image"]
        image = self.imgnormalize(image)
        qpos = batch["obs"]["qpos"]
        a_hat = self.nets["policy"](qpos, image, None) # no action, sample from prior
        return a_hat

