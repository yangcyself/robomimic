"""
Config for BC algorithm.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.bc_config import BCConfig
from collections import OrderedDict


class ACTConfig(BaseConfig):
    ALGO_NAME = "act"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        # optimization parameters
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 1e-2  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # loss weights
        self.algo.loss.kl_weight = 10.0      # L2 loss weight
        self.algo.chunk_size = 100
        self.algo.latent_dim = 32

        # backbone settings
        self.algo.backbone.train_backbone = False       
        self.algo.backbone.return_interm_layers = False 
        self.algo.backbone.backbone = "resnet18"        
        self.algo.backbone.dilation = False
        self.algo.backbone.hidden_dim = 512
        self.algo.backbone.position_embedding = "sine"
        
        # transformer settings
        self.algo.transformer.d_model = 512           
        self.algo.transformer.dropout = 0.1                  
        self.algo.transformer.nhead = 8       
        self.algo.transformer.dim_feedforward = 2048
        self.algo.transformer.num_encoder_layers = 4
        self.algo.transformer.num_decoder_layers = 6
        self.algo.transformer.normalize_before = False

        # encoder settings
        self.algo.encoder.d_model = 512
        self.algo.encoder.dropout = 0.1           
        self.algo.encoder.nhead = 8           
        self.algo.encoder.dim_feedforward = 2048
        self.algo.encoder.num_encoder_layers = 4
        self.algo.encoder.normalize_before = False

    def observation_config(self):
        """
        Update from superclass so that value planner and actor each get their own obs config.
        """


        ## Set the encoders of the observations to default
        self.observation.action_encoder.encoder = BCConfig().observation.encoder
        self.observation.actor.encoder = BCConfig().observation.encoder

        self.observation.action_encoder.modalities = OrderedDict()             # modalities are not limited to specific groups
        self.observation.actor.modalities = OrderedDict()                      # modalities are not limited to specific groups
        

    @property
    def all_obs_keys(self):
        """
        Update from superclass to include modalities from value planner and actor.
        """
        # pool all modalities
        return sorted(tuple(set([
            k for group in [
                self.observation.action_encoder.modalities.values(),
                self.observation.actor.modalities.values(),
            ]
            for modality in group
            for obs_keys in modality.values()
            for k in obs_keys
        ])))


    @property
    def use_goals(self):
        """
        Update from superclass - value planner goal modalities determine goal-conditioning.
        """
        return "goal" in self.observation.actor.modalities \
            and sum([len(obs) for obs in self.observation.actor.modalities["goal"].values()]) > 0

