import pytorch_lightning as pl

from util.config import ex
from util.dataset.lightning_dataset import Intent_CLS_DataModule
from util.model.Classifier import Intent_CLS_Module
import copy
import os


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    
    # Print config
    for key, val in _config.items():
        key_str = "{}".format(key) + (" " * (30 - len(key)))
        print(f"{key_str}   =   {val}")    
    
    pl.seed_everything(_config["seed"])   
    
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        verbose=True,
        monitor="val/accuracy",
        filename='epoch={epoch}-step={step}-val_acc={val/accuracy:.5f}',
        mode="max",
        save_last=True,
        auto_insert_metric_name=False
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )


    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    accumulate_grad_batches = max(_config["batch_size"] // (
        _config["per_gpu_batch_size"] * len(_config['gpus']) * _config["num_nodes"]
    ), 1)

    dm = Intent_CLS_DataModule(_config=_config)
    model = Intent_CLS_Module(_config=_config, num_labels=len(dm.train_labels_li))

    trainer = pl.Trainer(
        gpus=_config['gpus'],
        max_steps=_config["max_steps"],
        accelerator="ddp",
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=_config['val_check_interval']
        )

    trainer.fit(model, datamodule=dm)