import torch
import torch.nn as nn
import torchvision.models as models
from lightly.models.modules import SimCLRProjectionHead

class ResnetEmbeddingModel(nn.Module):
  """
  Embedding model for imaging trained with ResNet backbone.
  """
  def __init__(self, args) -> None:
    super(ResnetEmbeddingModel, self).__init__()

    self.keep_projector = args.keep_projector

    # Load weights
    checkpoint = torch.load(args.checkpoint)
    original_args = checkpoint['hyper_parameters']
    state_dict = checkpoint['state_dict']

    # Load architecture
    if original_args['model'] == 'resnet18':
      model = models.resnet18(pretrained=False, num_classes=100)
      pooled_dim = 512
    elif original_args['model'] == 'resnet50':
      model = models.resnet50(pretrained=False, num_classes=100)
      pooled_dim = 2048
    else:
      raise Exception('Invalid architecture. Please select either resnet18 or resnet50.')

    self.backbone = nn.Sequential(*list(model.children())[:-1])
    self.projection_head = SimCLRProjectionHead(pooled_dim, original_args['embedding_dim'], original_args['projection_dim'])

    # Remove prefix and fc layers
    state_dict_encoder = {}
    state_dict_projector = {}
    for k in list(state_dict.keys()):
      if k.startswith('encoder_imaging.'):
        state_dict_encoder[k[len('encoder_imaging.'):]] = state_dict[k]
      if k.startswith('projection_head_imaging.'):
        state_dict_projector[k[len('projection_head_imaging.'):]] = state_dict[k]

    log = self.backbone.load_state_dict(state_dict_encoder, strict=True)
    assert len(log.missing_keys) == 0
    log = self.projection_head.load_state_dict(state_dict_projector, strict=True)
    assert len(log.missing_keys) == 0

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    embeddings = self.backbone(x).squeeze()
    
    if self.keep_projector:
      embeddings = self.projection_head(embeddings)

    return embeddings
