import unittest
from hydra import compose, initialize
import multiprocessing as mp
import traceback

import run
import controller

class Process(mp.Process):
    def __init__(self, *args, **kwargs):
        mp.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = mp.Pipe()
        self._exception = None

    def run(self):
        try:
            mp.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))
            # raise e  # You can still rise this exception if you need to

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception
    


class TestFullRuns(unittest.TestCase):
  def run_process(self, target, hparams):
      p = Process(target=target, args=(hparams,))
      p.start()
      p.join()

      if p.exception:
        error, traceback = p.exception
        print(traceback)
        raise error
  
  #@unittest.skip
  def test_original_args(self) -> None:
    with initialize(version_base=None, config_path='./configs'):
      args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac_original', 'datatype=imaging', 'pretrain=True', 'dataset=cardiac_original', 'multitarget=[Infarction_original,CAD_original]', "low_data_splits=['','_0.1']"])
      self.run_process(controller.control, args)

  #@unittest.skip
  def test_imaging_pretraining(self) -> None:
    with initialize(version_base=None, config_path='./configs'):
      args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac', 'datatype=imaging', 'pretrain=True'])
      self.run_process(controller.control, args)

  #@unittest.skip
  def test_imaging_pretraining_byol(self) -> None:
    with initialize(version_base=None, config_path='./configs'):
      args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac', 'datatype=imaging', 'pretrain=True', 'loss=byol'])
      self.run_process(controller.control, args)

  #@unittest.skip
  def test_imaging_pretraining_simsiam(self) -> None:
    with initialize(version_base=None, config_path='./configs'):
      args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac', 'datatype=imaging', 'pretrain=True', 'loss=simsiam'])
      self.run_process(controller.control, args)

  #@unittest.skip
  def test_imaging_pretraining_barlowtwins(self) -> None:
    with initialize(version_base=None, config_path='./configs'):
      args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac', 'datatype=imaging', 'pretrain=True', 'loss=barlowtwins'])
      self.run_process(controller.control, args)

  #@unittest.skip
  def test_tabular_pretraining(self) -> None:
    with initialize(version_base=None, config_path='./configs'):
      args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac', 'datatype=tabular', 'pretrain=True'])
      self.run_process(controller.control, args)

  #@unittest.skip
  def test_multimodal_pretraining(self) -> None:
    with initialize(version_base=None, config_path='./configs'):
      args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac', 'datatype=multimodal', 'pretrain=True'])
      self.run_process(controller.control, args)
  
  #@unittest.skip
  def test_imaging_supervised(self) -> None:
    with initialize(version_base=None, config_path='./configs'):
      args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac', 'datatype=imaging'])
      self.run_process(controller.control, args)

  #@unittest.skip
  def test_tabular_supervised(self) -> None:
    with initialize(version_base=None, config_path='./configs'):
      args = compose(config_name='config', overrides=['+experiment=testing', 'paths=tower_cardiac', 'datatype=tabular'])
      self.run_process(controller.control, args)

      

if __name__ == '__main__':
  unittest.main()