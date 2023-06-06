"""
Config for BC algorithm.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config.bc_config import BCConfig


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
        self.algo.camera_names = ["agentview_image"]


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
        self.algo.encoder.hidden_dim = 512
        self.algo.encoder.dropout = 0.1           
        self.algo.encoder.nheads = 8           
        self.algo.encoder.dim_feedforward = 2048
        self.algo.encoder.enc_layers = 4
        self.algo.encoder.pre_norm = False

    def observation_config(self):
        """
        Update from superclass so that value planner and actor each get their own obs config.
        """

        self.observation.action_encoder = BCConfig().observation

        ## Set the encoders of the observations to default
        self.observation.actor.encoder = BCConfig().observation.encoder

        ## TODO: See if this can be changed to list
        self.observation.actor.modalities.joints.low_dim = []            # specify low-dim observations for agent
        self.observation.actor.modalities.joints.rgb = []              # specify rgb image observations for agent
        self.observation.actor.modalities.joints.depth = []
        self.observation.actor.modalities.joints.scan = []
        self.observation.actor.modalities.latent.low_dim = []
        self.observation.actor.modalities.latent.rgb = []
        self.observation.actor.modalities.latent.depth = []
        self.observation.actor.modalities.latent.scan = []
        self.observation.actor.modalities.cams.low_dim = []
        self.observation.actor.modalities.cams.rgb = []
        self.observation.actor.modalities.cams.depth = []
        self.observation.actor.modalities.cams.scan = []
        self.observation.actor.modalities.goal.low_dim = []           # specify low-dim goal observations to condition agent on
        self.observation.actor.modalities.goal.rgb = []             # specify rgb image goal observations to condition agent on
        self.observation.actor.modalities.goal.depth = []
        self.observation.actor.modalities.goal.scan = []
        self.observation.actor.modalities.joints.do_not_lock_keys()
        self.observation.actor.modalities.goal.do_not_lock_keys()
        self.observation.actor.modalities.cams.do_not_lock_keys()


    @property
    def all_obs_keys(self):
        """
        Update from superclass to include modalities from value planner and actor.
        """
        # pool all modalities
        return sorted(tuple(set([
            obs_key for group in [
                self.observation.action_encoder.modalities.obs.values(),
                self.observation.actor.modalities.joints.values(),
                self.observation.actor.modalities.cams.values(),
            ]
            for modality in group
            for obs_key in modality
        ])))


    @property
    def use_goals(self):
        """
        Update from superclass - value planner goal modalities determine goal-conditioning.
        """
        return len(
            self.observation.actor.modalities.goal.low_dim +
            self.observation.actor.modalities.goal.rgb) > 0

