from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn

class TabularEncoder(nn.Module):
  """
  Main contrastive model used in SCARF. Consists of an encoder that takes the input and 
  creates an embedding of size {args.embedding_dim}.
  Also supports providing a checkpoint with trained weights to be loaded.
  """
  def __init__(self, args) -> None:
    super(TabularEncoder, self).__init__()
    self.args = args

    # Check if we are loading a pretrained model
    if args.checkpoint:
      loaded_chkpt = torch.load(args.checkpoint)
      original_args = loaded_chkpt['hyper_parameters']
      state_dict = loaded_chkpt['state_dict']
      self.input_size = original_args['input_size']
      
      if 'encoder_tabular.encoder.1.running_mean' in state_dict.keys():
        encoder_name = 'encoder_tabular.encoder.'
        self.encoder = self.build_encoder(original_args)
      elif 'encoder_projector_tabular.encoder.2.running_mean' in state_dict.keys():
        encoder_name = 'encoder_projector_tabular.encoder.'
        self.encoder = self.build_encoder_bn_old(original_args)
      else:
        encoder_name = 'encoder_projector_tabular.encoder.'
        self.encoder = self.build_encoder_no_bn(original_args)

      # Split weights
      state_dict_encoder = {}
      for k in list(state_dict.keys()):
        if k.startswith(encoder_name):
          state_dict_encoder[k[len(encoder_name):]] = state_dict[k]
        
      _ = self.encoder.load_state_dict(state_dict_encoder, strict=True)

      # Freeze if needed
      if args.finetune_strategy == 'frozen':
        for _, param in self.encoder.named_parameters():
          param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
        assert len(parameters)==0
    else:
      # Build architecture
      self.input_size = args.input_size
      self.encoder = self.build_encoder(args)
      self.encoder.apply(self.init_weights)

  def build_encoder(self, args: Dict) -> nn.Sequential:
    modules = [nn.Linear(self.input_size, args['embedding_dim'])]
    for _ in range(args['encoder_num_layers']-1):
      modules.extend([nn.BatchNorm1d(args['embedding_dim']), nn.ReLU(), nn.Linear(args['embedding_dim'], args['embedding_dim'])])
    return nn.Sequential(*modules)
  
  def build_encoder_no_bn(self, args: Dict) -> nn.Sequential:
    modules = [nn.Linear(self.input_size, args['embedding_dim'])]
    for _ in range(args['encoder_num_layers']-1):
      modules.extend([nn.ReLU(), nn.Linear(args['embedding_dim'], args['embedding_dim'])])
    return nn.Sequential(*modules)

  def build_encoder_bn_old(self, args: Dict) -> nn.Sequential:
    modules = [nn.Linear(args.input_size, args.embedding_dim)]
    for _ in range(args.encoder_num_layers-1):
      modules.extend([nn.ReLU(), nn.BatchNorm1d(args.embedding_dim), nn.Linear(args.embedding_dim, args.embedding_dim)])
    return nn.Sequential(*modules)

  def init_weights(self, m: nn.Module, init_gain = 0.02) -> None:
    """
    Initializes weights according to desired strategy
    """
    if isinstance(m, nn.Linear):
      if self.args.init_strat == 'normal':
        nn.init.normal_(m.weight.data, 0, 0.001)
      elif self.args.init_strat == 'xavier':
        nn.init.xavier_normal_(m.weight.data, gain=init_gain)
      elif self.args.init_strat == 'kaiming':
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif self.args.init_strat == 'orthogonal':
        nn.init.orthogonal_(m.weight.data, gain=init_gain)
      if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias.data, 0.0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Passes input through encoder and projector. 
    Output is ready for loss calculation.
    """
    x = self.encoder(x)
    return x