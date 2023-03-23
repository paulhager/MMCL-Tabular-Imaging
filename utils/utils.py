from typing import List, Tuple
from os.path import join
import os
import sys
from torch import nn

import torch
from torchvision import transforms

def create_logdir(name: str, resume_training: bool, wandb_logger):
  basepath = join(os.path.dirname(os.path.abspath(sys.argv[0])),'runs', name)
  run_name = wandb_logger.experiment.name
  logdir = join(basepath,run_name)
  if os.path.exists(logdir) and not resume_training:
    raise Exception(f'Run {run_name} already exists. Please delete the folder {logdir} or choose a different run name.')
  os.makedirs(logdir,exist_ok=True)
  return logdir

def grab_image_augmentations(img_size: int, target: str, crop_scale_lower: float = 0.08) -> transforms.Compose:
  """
  Defines augmentations to be used with images during contrastive training and creates Compose.
  """
  if target.lower() == 'dvm':
    transform = transforms.Compose([
      transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
      transforms.RandomResizedCrop(size=(img_size,img_size), scale=(crop_scale_lower, 1.0), ratio=(0.75, 1.3333333333333333)),
      transforms.RandomHorizontalFlip(p=0.5),
      #transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])
  else:
    transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(45),
      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
      transforms.RandomResizedCrop(size=img_size, scale=(0.2,1)),
      transforms.Lambda(lambda x: x.float())
    ])
  return transform

def grab_soft_eval_image_augmentations(img_size: int) -> transforms.Compose:
  """
  Defines augmentations to be used during evaluation of contrastive encoders. Typically a less sever form of contrastive augmentations.
  """
  transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.RandomResizedCrop(size=img_size, scale=(0.8,1)),
    transforms.Lambda(lambda x: x.float())
  ])
  return transform

def grab_hard_eval_image_augmentations(img_size: int, target: str) -> transforms.Compose:
  """
  Defines augmentations to be used during evaluation of contrastive encoders. Typically a less sever form of contrastive augmentations.
  """
  if target.lower() == 'dvm':
    transform = transforms.Compose([
      transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8)], p=0.8),
      transforms.RandomGrayscale(p=0.2),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=29, sigma=(0.1, 2.0))],p=0.5),
      transforms.RandomResizedCrop(size=(img_size,img_size), scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.Resize(size=(img_size,img_size)),
      transforms.Lambda(lambda x : x.float())
    ])
  else:
    transform = transforms.Compose([
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(45),
      transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
      transforms.RandomResizedCrop(size=img_size, scale=(0.6,1)),
      transforms.Lambda(lambda x: x.float())
    ])
  return transform

def grab_wids(category: str):
  # boat
  wids_b = ['n02951358', 'n03447447', 'n04612504', 'n03344393', 'n03662601', 'n04273569'] 
  # domestic cat
  wids_c = ['n02123597', 'n02123159', 'n02123045', 'n02124075', 'n02123394']
  # domestic dog
  wids_d = ['n02102480', 'n02096585', 'n02093256', 'n02091831', 'n02086910', 'n02100735', 'n02102040', 'n02085936', 'n02097130', 'n02097047', 'n02106662', 'n02110958', 'n02097209', 'n02092002', 'n02107142', 'n02099712', 'n02093754', 'n02112018', 'n02105412', 'n02096437', 'n02105251', 'n02108089', 'n02108551', 'n02095889', 'n02113624', 'n02093428', 'n02088238', 'n02100877', 'n02099849', 'n02108422', 'n02098413', 'n02086240', 'n02107574', 'n02101556', 'n02099429', 'n02098105', 'n02087394', 'n02108000', 'n02106166', 'n02107683', 'n02091244', 'n02101388', 'n02111889', 'n02093647', 'n02102973', 'n02101006', 'n02109961', 'n02085782', 'n02091635', 'n02112706', 'n02090622', 'n02110063', 'n02113712', 'n02110341', 'n02086079', 'n02089973', 'n02112350', 'n02113799', 'n02105162', 'n02108915', 'n02104029', 'n02089867', 'n02098286', 'n02105505', 'n02110627', 'n02106550', 'n02105641', 'n02100583', 'n02090721', 'n02093859', 'n02113978', 'n02088466', 'n02095570', 'n02099267', 'n02099601', 'n02106030', 'n02112137', 'n02089078', 'n02092339', 'n02088632', 'n02102177', 'n02096051', 'n02096294', 'n02096177', 'n02093991', 'n02110185', 'n02111277', 'n02090379', 'n02111500', 'n02088364', 'n02088094', 'n02094114', 'n02104365', 'n02111129', 'n02109525', 'n02097658', 'n02113186', 'n02095314', 'n02113023', 'n02087046', 'n02094258', 'n02100236', 'n02097298', 'n02105855', 'n02085620', 'n02106382', 'n02091032', 'n02110806', 'n02086646', 'n02094433', 'n02091134', 'n02107312', 'n02107908', 'n02097474', 'n02091467', 'n02102318', 'n02105056', 'n02109047']

  if category == 'Boat':
    return wids_b
  elif category == 'DomesticCat':
    return wids_c
  elif category == 'DomesticDog':
    return wids_d
  else:
    raise ValueError('Category not recognized')

def grab_arg_from_checkpoint(args: str, arg_name: str):
  """
  Loads a lightning checkpoint and returns an argument saved in that checkpoints hyperparameters
  """
  if args.checkpoint:
    ckpt = torch.load(args.checkpoint)
    load_args = ckpt['hyper_parameters']
  else:
    load_args = args
  return load_args[arg_name]

def chkpt_contains_arg(ckpt_path: str, arg_name: str):
  """
  Checks if a checkpoint contains a given argument.
  """
  ckpt = torch.load(ckpt_path)
  return arg_name in ckpt['hyper_parameters']

def prepend_paths(hparams):
  db = hparams.data_base
  
  for hp in [
    'labels_train', 'labels_val', 
    'data_train_imaging', 'data_val_imaging', 
    'data_val_eval_imaging', 'labels_val_eval_imaging', 
    'train_similarity_matrix', 'val_similarity_matrix', 
    'data_train_eval_imaging', 'labels_train_eval_imaging',
    'data_train_tabular', 'data_val_tabular', 
    'data_val_eval_tabular', 'labels_val_eval_tabular', 
    'data_train_eval_tabular', 'labels_train_eval_tabular',
    'field_indices_tabular', 'field_lengths_tabular',
    'data_test_eval_tabular', 'labels_test_eval_tabular',
    'data_test_eval_imaging', 'labels_test_eval_imaging',
    ]:
    if hp in hparams and hparams[hp]:
      hparams['{}_short'.format(hp)] = hparams[hp]
      hparams[hp] = join(db, hparams[hp])

  return hparams

def re_prepend_paths(hparams):
  db = hparams.data_base
  
  for hp in [
    'labels_train', 'labels_val', 
    'data_train_imaging', 'data_val_imaging', 
    'data_val_eval_imaging', 'labels_val_eval_imaging', 
    'train_similarity_matrix', 'val_similarity_matrix', 
    'data_train_eval_imaging', 'labels_train_eval_imaging',
    'data_train_tabular', 'data_val_tabular', 
    'data_val_eval_tabular', 'labels_val_eval_tabular', 
    'data_train_eval_tabular', 'labels_train_eval_tabular',
    'field_indices_tabular', 'field_lengths_tabular',
    'data_test_eval_tabular', 'labels_test_eval_tabular',
    'data_test_eval_imaging', 'labels_test_eval_imaging',
    ]:
    if hp in hparams and hparams[hp]:
      hparams[hp] = join(db, hparams['{}_short'.format(hp)])

  return hparams

def cos_sim_collate(data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
  """
  Collate function to use when cosine similarity of embeddings is relevant. Takes the embeddings returned by the dataset and calculates the cosine similarity matrix for them.
  """
  view_1, view_2, labels, embeddings, thresholds = zip(*data)
  view_1 = torch.stack(view_1)
  view_2 = torch.stack(view_2)
  labels = torch.tensor(labels)
  threshold = thresholds[0]

  cos = torch.nn.CosineSimilarity(dim=0)
  cos_sim_matrix = torch.zeros((len(embeddings),len(embeddings)))
  for i in range(len(embeddings)):
      for j in range(i,len(embeddings)):
          val = cos(embeddings[i],embeddings[j]).item()
          cos_sim_matrix[i,j] = val
          cos_sim_matrix[j,i] = val

  if threshold:
    cos_sim_matrix = torch.threshold(cos_sim_matrix,threshold,0)

  return view_1, view_2, labels, cos_sim_matrix

def calc_logits_labels(out0, out1, temperature=0.1):
  out0 = nn.functional.normalize(out0, dim=1)
  out1 = nn.functional.normalize(out1, dim=1)

  logits = torch.matmul(out0, out1.T) / temperature
  labels = torch.arange(len(out0), device=out0.device)

  return logits, labels