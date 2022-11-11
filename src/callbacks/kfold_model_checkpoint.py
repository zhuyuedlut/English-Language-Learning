import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint

class KFoldModelCheckpoint(ModelCheckpoint):
    def __init__(self, fold: int, *args, **kwargs):
        super(KFoldModelCheckpoint, self).__init__(*args, **kwargs)
        self.fold = fold

    def _should_skip_saving_checkpoint(self, trainer: pl.Trainer) -> bool:
        return trainer.fit_loop.current_fold != self.fold or super()._should_skip_saving_checkpoint(trainer)