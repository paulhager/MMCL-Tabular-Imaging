import os 

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data.sampler import WeightedRandomSampler

from datasets.ImageDataset import ImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from models.Evaluator import Evaluator
from models.Evaluator_regression import Evaluator_Regression
from utils.utils import grab_arg_from_checkpoint, grab_hard_eval_image_augmentations, grab_wids, create_logdir

def load_datasets(hparams):
  if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
    train_dataset = ImageDataset(hparams.data_train_eval_imaging, hparams.labels_train_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=True, live_loading=hparams.live_loading, task=hparams.task)
    val_dataset = ImageDataset(hparams.data_val_eval_imaging, hparams.labels_val_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task)
  elif hparams.datatype == 'tabular':
    train_dataset = TabularDataset(hparams.data_train_eval_tabular, hparams.labels_train_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
    val_dataset = TabularDataset(hparams.data_val_eval_tabular, hparams.labels_val_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
    hparams.input_size = train_dataset.get_input_size()
  elif hparams.datatype == 'imaging_and_tabular':
    train_dataset = ImagingAndTabularDataset(
      hparams.data_train_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate, 
      hparams.data_train_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_train_eval_imaging, hparams.img_size, hparams.live_loading, train=True, target=hparams.target
    )
    val_dataset = ImagingAndTabularDataset(
      hparams.data_val_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate, 
      hparams.data_val_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
      hparams.labels_val_eval_imaging, hparams.img_size, hparams.live_loading, train=False, target=hparams.target
    )
    hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception('argument datatype must be set to imaging, tabular, multimodal or imaging_and_tabular')
  return train_dataset, val_dataset


def evaluate(hparams, wandb_logger):
  """
  Evaluates trained contrastive models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  pl.seed_everything(hparams.seed)
  
  train_dataset, val_dataset = load_datasets(hparams)
  
  drop = ((len(train_dataset)%hparams.batch_size)==1)

  sampler = None
  if hparams.weighted_sampler:
    print('Using weighted random sampler(')
    weights_list = [hparams.weights[int(l)] for l in train_dataset.labels]
    sampler = WeightedRandomSampler(weights=weights_list, num_samples=len(weights_list), replacement=True)
  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size, sampler=sampler,
    pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=512,
    pin_memory=True, shuffle=False, persistent_workers=True)

  logdir = create_logdir('eval', hparams.resume_training, wandb_logger)

  if hparams.task == 'regression':
    model = Evaluator_Regression(hparams)
  else:
    model = Evaluator(hparams)
  
  mode = 'max'
  
  callbacks = []
  callbacks.append(ModelCheckpoint(monitor=f'eval.val.{hparams.eval_metric}', mode=mode, filename=f'checkpoint_best_{hparams.eval_metric}', dirpath=logdir))
  callbacks.append(EarlyStopping(monitor=f'eval.val.{hparams.eval_metric}', min_delta=0.0002, patience=int(10*(1/hparams.val_check_interval)), verbose=False, mode=mode))
  if hparams.use_wandb:
    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, accelerator="gpu", devices=1, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, val_check_interval=hparams.val_check_interval, limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, limit_test_batches=hparams.limit_test_batches)

  trainer.fit(model, train_loader, val_loader)
  
  wandb_logger.log_metrics({f'best.val.{hparams.eval_metric}': model.best_val_score})

  if hparams.test_and_eval:
    if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
      test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task)
      
      hparams.transform_test = test_dataset.transform_val.__repr__()
    elif hparams.datatype == 'tabular':
      test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
      hparams.input_size = test_dataset.get_input_size()
    elif hparams.datatype == 'imaging_and_tabular':
      test_dataset = ImagingAndTabularDataset(
        hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, 
        hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
        hparams.labels_test_eval_imaging, hparams.img_size, hparams.live_loading, train=False, target=hparams.target)
      hparams.input_size = test_dataset.get_input_size()
    else:
      raise Exception('argument dataset must be set to imaging, tabular or multimodal')
    
    drop = ((len(test_dataset)%hparams.batch_size)==1)

    test_loader = DataLoader(
      test_dataset,
      num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
      pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)

    model.freeze()

    trainer.test(model, test_loader, ckpt_path=os.path.join(logdir,f'checkpoint_best_{hparams.eval_metric}.ckpt'))
