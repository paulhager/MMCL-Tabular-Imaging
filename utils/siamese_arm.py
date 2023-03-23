from torch import nn

from pl_bolts.utils.self_supervised import torchvision_ssl_encoder


class MLP(nn.Module):
    def __init__(self, input_dim=2048, hidden_size=4096, output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=False),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, output_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class SiameseArm(nn.Module):
    def __init__(self, encoder="resnet50", encoder_out_dim=2048, projector_hidden_size=4096, projector_out_dim=256, predictor=True, predictor_hidden_size=None):
        super().__init__()
        self.predictor = predictor

        if predictor_hidden_size is None:
          predictor_hidden_size = projector_hidden_size

        if isinstance(encoder, str):
            encoder = torchvision_ssl_encoder(encoder)
        # Encoder
        self.encoder = encoder
        print(self.encoder)
        # Projector
        self.projector = MLP(encoder_out_dim, projector_hidden_size, projector_out_dim)
        print(self.projector)
        # Predictor
        if self.predictor:
          self.predictor = MLP(projector_out_dim, predictor_hidden_size, projector_out_dim)
          print(self.predictor)



    def forward(self, x):
        y = self.encoder(x)[0]
        z = self.projector(y)
        h = None
        if self.predictor:
          h = self.predictor(z)
        return y, z, h
