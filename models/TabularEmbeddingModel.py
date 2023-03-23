from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn

class TabularEmbeddingModel(nn.Module):
  """
  Embedding model for tabular trained with MLP backbone.
  """
  def __init__(self, args):
    super(TabularEmbeddingModel, self).__init__()

    self.keep_projector = args.keep_projector

    # Load weights
    loaded_chkpt = torch.load(args.checkpoint)
    original_args = loaded_chkpt['hyper_parameters']
    state_dict = loaded_chkpt['state_dict']

    # Build architecture
    self.encoder = self.build_encoder(original_args)
    self.projector = self.build_projector(original_args)

    # Split weights
    state_dict_encoder = {}
    state_dict_projector = {}
    for k in list(state_dict.keys()):
      if k.startswith('encoder_projector_tabular.encoder.'):
        state_dict_encoder[k[len('encoder_projector_tabular.encoder.'):]] = state_dict[k]
      if k.startswith('encoder_projector_tabular.projector.'):
        state_dict_projector[k[len('encoder_projector_tabular.projector.'):]] = state_dict[k]

    _ = self.encoder.load_state_dict(state_dict_encoder, strict=True)
    _ = self.projector.load_state_dict(state_dict_projector, strict=True)


  def build_encoder(self, original_args: Dict) -> nn.Sequential:
    modules = [nn.Linear(original_args['input_size'], original_args['embedding_dim'])]
    for _ in range(original_args['encoder_num_layers']-1):
      modules.extend([nn.ReLU(), nn.Linear(original_args['embedding_dim'], original_args['embedding_dim'])])
    return nn.Sequential(*modules)

  def build_projector(self, original_args: Dict) -> nn.Sequential:
    modules = [nn.ReLU(), nn.Linear(original_args['embedding_dim'], original_args['projection_dim'])]
    for _ in range(original_args['projector_num_layers']-1):
      modules.extend([nn.ReLU(), nn.Linear(original_args['projection_dim'], original_args['projection_dim'])])
    return nn.Sequential(*modules)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    embeddings = self.encoder(x)

    if self.keep_projector:
      embeddings = self.projector(embeddings)
    
    return embeddings