import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

class ImagingModel(nn.Module):
  """
  Evaluation model for imaging trained with ResNet encoder.
  """
  def __init__(self, args) -> None:
    super(ImagingModel, self).__init__()

    if args.checkpoint:
      # Load weights
      checkpoint = torch.load(args.checkpoint)
      original_args = checkpoint['hyper_parameters']
      state_dict = checkpoint['state_dict']
      self.pooled_dim = 2048 if original_args['model']=='resnet50' else 512

      if 'encoder_imaging.0.weight' in state_dict:
        self.bolt_encoder = False
        self.encoder_name = 'encoder_imaging.'
        self.create_imaging_model(original_args)
      else:
        encoder_name_dict = {'clip' : 'encoder_imaging.', 'remove_fn' : 'encoder_imaging.', 'supcon' : 'encoder_imaging.', 'byol': 'online_network.encoder.', 'simsiam': 'online_network.encoder.', 'swav': 'model.', 'barlowtwins': 'network.encoder.'}
        self.bolt_encoder = True
        self.encoder = torchvision_ssl_encoder(original_args['model'])
        self.encoder_name = encoder_name_dict[original_args['loss']]

      # Remove prefix and fc layers
      state_dict_encoder = {}
      for k in list(state_dict.keys()):
        if k.startswith(self.encoder_name) and not 'projection_head' in k and not 'prototypes' in k:
          state_dict_encoder[k[len(self.encoder_name):]] = state_dict[k]

      log = self.encoder.load_state_dict(state_dict_encoder, strict=True)
      assert len(log.missing_keys) == 0

      # Freeze if needed
      if args.finetune_strategy == 'frozen':
        for _, param in self.encoder.named_parameters():
          param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
        assert len(parameters)==0
    else:
      self.bolt_encoder = True
      self.pooled_dim = 2048 if args.model=='resnet50' else 512
      self.encoder = torchvision_ssl_encoder(args.model)

    self.classifier = nn.Linear(self.pooled_dim, args.num_classes)

  def create_imaging_model(self, args):
    if args['model'] == 'resnet18':
      model = models.resnet18(pretrained=False, num_classes=100)
      self.pooled_dim = 512
    elif args['model'] == 'resnet50':
      model = models.resnet50(pretrained=False, num_classes=100)
      self.pooled_dim = 2048
    else:
      raise Exception('Invalid architecture. Please select either resnet18 or resnet50.')
    self.encoder = nn.Sequential(*list(model.children())[:-1])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.bolt_encoder:
      x = self.encoder(x)[0]
    else:
      x = self.encoder(x).squeeze()
    x = self.classifier(x)
    return x