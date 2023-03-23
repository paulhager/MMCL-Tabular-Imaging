import torch
import numpy as np

class RandomGamma(torch.nn.Module):
  """
  Applies gamma transform to images. 
  
  Raises pixel values in image to the power of a random number between {low} and {high}.
  If pixels are negative, takes the absolute value of all pixels before applying transform.
  """
  def __call__(self, pic: torch.Tensor, low=0.25, high=1.75) -> torch.Tensor:
    ran = np.random.uniform(low=low,high=high)
    transformed_tensors = self.power(pic,ran)
    return transformed_tensors
  
  def __repr__(self) -> str:
    return self.__class__.__name__ + '()'
  
  def power(self, tensor: torch.Tensor, gamma: float) -> torch.tensor:
    if tensor.min() < 0:
      output = tensor.sign() * tensor.abs() ** gamma
    else:
      output = tensor ** gamma
    return output