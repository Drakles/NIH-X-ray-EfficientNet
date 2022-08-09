from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from efficentnet_lightning_module import EfficennetLightning


TENSORBOARD_DIRECTORY = "logs/"
seed_everything(42)

if __name__ == '__main__':
    logger = TensorBoardLogger(TENSORBOARD_DIRECTORY, name="efficentnet_fine_tuning")

    model = EfficennetLightning()

    trainer = Trainer(gpus=1, fast_dev_run=True)
    trainer.fit(model)

    checkpoint_callback = ModelCheckpoint(
        filepath='model/weights.ckpt',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    trainer = Trainer(max_epochs=20,
                      logger=logger,
                      gpus=1,
                      accumulate_grad_batches=4,
                      deterministic=True,
                      early_stop_callback=True)

    trainer.fit(model)