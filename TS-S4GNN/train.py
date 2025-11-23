import numpy as np                                 
import os
import pickle
import torch
import json
import torch.nn as nn        
import torch.nn.functional as F              
import torch.optim as optim              
import pandas as pd    

from data.datamodules.datamodule_tuh import TUH_DataModule                      #加载数据集
from data.datamodules.datamodule_icbeb import ICBEB_DataModule
from data.datamodules.datamodule_dreem import Dreem_DataModule
from args import get_args
import torch
from model.feature_fusion import *
from model.graph_fusion import *    
from model.lstm import LSTMModel
from model.s4 import S4Model
from constants import *

import utils.utils as utils
from utils.schedulers import *
from tqdm import tqdm
from dotted_dict import DottedDict 
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, MultiStepLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    ModelCheckpoint,
)
import copy
from collections import OrderedDict
from json import dumps
import sys
import itertools
from pytorch_lightning.loggers import TensorBoardLogger

torch.autograd.set_detect_anomaly(True)             #异常检测

class PLModel(pl.LightningModule):
    def __init__(
        self,
        args,
        lr=1e-3,
        weight_decay=1e-3,
        optimizer_name="adamw",
        scheduler_name="cosine",
        steps_per_epoch=None,
        scaler=None,
        log_prefix="",
        **scheduler_kwargs,
    ):
        super().__init__()
        self.args = args
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.steps_per_epoch = steps_per_epoch
        self.scaler = scaler
        self.scheduler_kwargs = scheduler_kwargs
        self.log_prefix = log_prefix

        self._build_model()

    def _build_model(self):
        args = self.args
        undirected_graph = args.undirected_graph
        args.max_seq_len *= TUH_FREQUENCY

        if args.model_name.lower() == "s4":
            self.model = S4Model(
                d_input=args.num_nodes * args.input_dim,
                d_output=args.output_dim,
                d_model=args.hidden_dim,
                d_state=args.state_dim,
                n_layers=args.num_temporal_layers,
                dropout=args.dropout,
                prenorm=args.prenorm,
                l_max=args.max_seq_len,
                l_output=args.output_seq_len,
                bidirectional=args.bidirectional,
                postact=args.postact,
                add_decoder=True,
                pool=False,
                temporal_pool=args.temporal_pool,
            )
        elif args.model_name.lower() == "lstm":
            self.model = LSTMModel(
                input_dim=args.num_nodes * args.input_dim,
                hidden_dim=args.hidden_dim,
                num_rnn_layers=args.num_temporal_layers,
                output_dim=args.output_dim,
                output_seq_len=args.output_seq_len,
                temporal_pool=args.temporal_pool,
                dropout=args.dropout,
                add_decoder=True,
            )
        
        elif args.model_name.lower() == "graph_fusion":
            self.model = Graph_Fusion(                    
                input_dim=args.input_dim,                 #输入特征维度 = 1
                num_nodes=args.num_nodes, 
                dropout=args.dropout,
                g_conv=args.g_conv,                    
                num_gnn_layers=args.num_gcn_layers,
                hidden_dim=args.hidden_dim,
                max_seq_len=args.max_seq_len,
                resolution=args.resolution,               #时间间隔r内学习唯一的图结构
                num_temporal_layers=args.num_temporal_layers,
                state_dim=args.state_dim,
                channels=args.channels, 
                temporal_model=args.temporal_model, 
                bidirectional=args.bidirectional,
                temporal_pool=args.temporal_pool,
                prenorm=args.prenorm,
                postact=args.postact,
                metric=args.graph_learn_metric,
                adj_embed_dim=args.adj_embed_dim,
                gin_mlp=args.gin_mlp,
                train_eps=args.train_eps,
                prune_method=args.prune_method,
                edge_top_perc=args.edge_top_perc,
                thresh=args.thresh,
                graph_pool=args.graph_pool,
                activation_fn=args.activation_fn,
                num_classes=args.output_dim,   
                undirected_graph=undirected_graph,
                use_prior=args.use_prior,
                K=args.knn,
                regularizations=args.regularizations,
                residual_weight=args.residual_weight,
                decay_residual_weight=args.decay_residual_weight,
            )
        elif args.model_name.lower() == "feature_fusion":
            self.model = Fusion(                    
                input_dim=args.input_dim,
                num_nodes=args.num_nodes,
                dropout=args.dropout,
                g_conv=args.g_conv,
                num_gnn_layers=args.num_gcn_layers,
                hidden_dim=args.hidden_dim,
                max_seq_len=args.max_seq_len,
                resolution=args.resolution,
                num_temporal_layers=args.num_temporal_layers,
                state_dim=args.state_dim,
                channels=args.channels, 
                temporal_model=args.temporal_model,
                bidirectional=args.bidirectional,
                temporal_pool=args.temporal_pool,
                prenorm=args.prenorm,       
                postact=args.postact,     
                metric=args.graph_learn_metric,      
                adj_embed_dim=args.adj_embed_dim,     
                gin_mlp=args.gin_mlp,
                train_eps=args.train_eps,
                prune_method=args.prune_method,          
                edge_top_perc=args.edge_top_perc,        
                thresh=args.thresh,
                graph_pool=args.graph_pool,
                activation_fn=args.activation_fn,
                num_classes=args.output_dim,
                undirected_graph=undirected_graph,
                K=args.knn,
                regularizations=args.regularizations,
                residual_weight=args.residual_weight,
                decay_residual_weight=args.decay_residual_weight,
            )
        
        else:
            raise NotImplementedError

    def _aggregate_regularization_losses(self, reg_loss_dict):
        reg_loss = 0.0
        for k in self.args.regularizations:
            if k == "feature_smoothing":
                reg_loss = (
                    reg_loss + self.args.feature_smoothing_weight * reg_loss_dict[k] 
                )
            elif k == "degree":
                reg_loss = reg_loss + self.args.degree_weight * reg_loss_dict[k]             
            elif k == "sparse":
                reg_loss = reg_loss + self.args.sparse_weight * reg_loss_dict[k]
            else:
                raise NotImplementedError()
        return reg_loss
#定义模型在训练时的单步操作 
    def training_step(self, batch, batch_idx):                          
        logits, y, cls_loss, reg_loss, _, _, _, _ = self._shared_step(   #调用 _shared_step 方法，获取模型输出和损失。返回logits：模型的输出。y：真实标签。cls_loss：分类损失。reg_loss：正则化损失。
            batch, batch_idx=batch_idx 
        )

        log_dict = {}          
        if (self.args.model_name == "graph_fusion"):
            loss = cls_loss + 10*reg_loss                                   
            log_dict["{}train/reg_loss".format(self.log_prefix)] = reg_loss.item()
        elif (self.args.model_name == "feature_fusion"):
            loss = cls_loss + 10*reg_loss
            log_dict["{}train/reg_loss".format(self.log_prefix)] = reg_loss.item()
        else:
            loss = cls_loss

        log_dict["{}train/cls_loss".format(self.log_prefix)] = cls_loss.item()
        log_dict["{}train/loss".format(self.log_prefix)] = loss.item()

        self.log_dict(                 #记录日志
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
            logger=True,
        )

        return loss                    

    def validation_step(self, batch, batch_idx):                          #定义模型在验证时的单步操作

        logits, y, cls_loss, reg_loss, file_names, _, _, _ = self._shared_step(batch)
        log_dict = {}
        if self.args.model_name == "graph_fusion":
            log_dict["val/reg_loss"] = reg_loss.item()
        if self.args.model_name == "feature_fusion":
            log_dict["val/reg_loss"] = reg_loss.item()    
        log_dict["val/cls_loss"] = cls_loss.item()
        self.log_dict(                 #记录日志
            log_dict,                  
            on_step=True,  
            on_epoch=True,       
            prog_bar=True,   
            logger=True,
        )
        return {                        
            "labels": y,                         
            "logits": logits,                            
            "reg_loss": reg_loss,                         
            "cls_loss":cls_loss,                           
            "file_names": file_names,                 
        }
                                               
    def validation_epoch_end(self, outputs):                   #验证周期结束时调用，处理所有批次的验证结果。
        logits = torch.cat([output["logits"] for output in outputs]).squeeze()                
        labels = torch.cat([output["labels"] for output in outputs]).squeeze()

        log_dict = {}

       
        if (self.args.model_name == "graph_fusion"):                  
            reg_loss = torch.mean(                              
                torch.stack([output["reg_loss"] for output in outputs])
            )
        if (self.args.model_name == "feature_fusion"):                  
            reg_loss = torch.mean(                       
                torch.stack([output["reg_loss"] for output in outputs])
            )
        if self.args.task == "classification":
            # classification
            if num_classes == 1:                           #二分类任务
                cls_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    pos_weight=torch.FloatTensor(self.args.pos_weight).to( 
                        self.device
                    ) if (self.args.pos_weight is not None) else None,
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)
            if num_classes == 4:                           #4分类任务
                self.class_weights = torch.tensor([2.0, 8.0, 10.0, 20.0]).to(self.device)  
                cls_loss = F.cross_entropy(
                    logits,
                    labels.long(),
                    weight=self.class_weights  # 应用类别权重
                )
                probs = torch.softmax(logits, dim=-1).cpu().numpy()  
                preds = np.argmax(probs, axis=-1) 
                print("当前 batch 的真实标签:", labels.cpu().numpy()) 
                print("当前 batch 的预测结果:", preds)
   
            scores_dict, _ = utils.eval_dict(                  
                y_pred=preds,
                y=labels.cpu().numpy(),
                y_prob=probs,
                average=self.args.metric_avg,                
                metrics=self.args.eval_metrics,                 
                find_threshold_on=self.args.find_threshold_on,   
            )
        else:
            raise NotImplementedError


        if (self.args.model_name == "graph_fusion"):                  #计算总损失
            loss = cls_loss + 10*reg_loss
            log_dict["{}val/reg_loss".format(self.log_prefix)] = reg_loss.item()
        elif (self.args.model_name == "feature_fusion"):
            loss = cls_loss + 10*reg_loss
            log_dict["{}val/reg_loss".format(self.log_prefix)] = reg_loss.item()
        else:
            loss = cls_loss

        log_dict["{}val/cls_loss".format(self.log_prefix)] = cls_loss.item()
        log_dict["{}val/loss".format(self.log_prefix)] = loss.item()
        for k, v in scores_dict.items():
            log_dict["{}val/{}".format(self.log_prefix, k)] = v

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):        #定义模型在测试时的单步操作


        prefix = ["val", "test"][dataloader_idx]

        (
            logits,
            y,
            cls_loss,
            reg_loss, 
            file_names,
            raw_attn_weight, 
            adj_mat_learned, 
            features
        ) = self._shared_step(batch)
        log_dict = {}
        if self.args.model_name == "graph_fusion":
            log_dict[f"{prefix}/reg_loss"] = reg_loss.item()
        if self.args.model_name == "feature_fusion":
            log_dict[f"{prefix}/reg_loss"] = reg_loss.item()
        log_dict[f"{prefix}/cls_loss"] = cls_loss.item()

        # 记录指标
        self.log_dict(
            log_dict,
            on_step=False,  
            on_epoch=True, 
            prog_bar=True, 
            logger=True,   
        )
        return {
            "labels": y,
            "logits": logits,
            "prefix": prefix,
            "file_names": file_names,
            "raw_attn_weight": raw_attn_weight,
            "adj_mat_learned": adj_mat_learned,
            "features": features,
        }

    def test_epoch_end(self, outputs):                             
        thresholds = None
        find_threshold = False
        find_threshold_on = None

        for curr_outputs in outputs: 
            logits = torch.cat([output["logits"] for output in curr_outputs]).squeeze()
            labels = torch.cat([output["labels"] for output in curr_outputs]).squeeze() #合并
            file_names = [output["file_names"] for output in curr_outputs]
            prefix = [output["prefix"] for output in curr_outputs][0]

            if self.args.task == "classification":
                # classification
                if num_classes == 1:                        #二分类任务
                    cls_loss = F.binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        pos_weight=torch.FloatTensor(self.args.pos_weight).to(
                            self.device
                        ) if (self.args.pos_weight is not None) else None,
                    )
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                if num_classes == 4:                        #4分类任务
                    cls_loss = F.cross_entropy(
                        logits,
                        labels.long(),
                    )
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=-1)
        
                scores_dict, thresholds = utils.eval_dict(
                    y_pred=preds,
                    y=labels.cpu().numpy(),
                    y_prob=probs,
                    average=self.args.metric_avg,
                    metrics=self.args.eval_metrics,
                    find_threshold_on=find_threshold_on,
                    thresholds=thresholds,
                )
            else:
                raise NotImplementedError

            # log
            res_str = "{} - ".format(prefix)
            for k, v in scores_dict.items():
                res_str = res_str + "{}: {:.4f}; ".format(k, v)
            print(res_str)

            self.log_dict(
                {
                    "{}{}/best_{}".format(
                        self.log_prefix, prefix, self.args.metric_name
                    ): scores_dict[self.args.metric_name]
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

            # save scores 保存测试结果到pkl文件中
            with open(
                os.path.join(self.args.save_dir, "{}_scores.pkl".format(prefix)), "wb"
            ) as pf:
                pickle.dump(scores_dict, pf)

            # save outputs  保存模型输出
            if self.args.save_output:
                outputs_dict = {
                    "logits": logits,
                    "labels": labels,
                    "file_names": file_names,
                }
                with open(
                    os.path.join(self.args.save_dir, "{}_results.pkl".format(prefix)),
                    "wb",
                ) as pf:
                    pickle.dump(outputs_dict, pf)
            if self.args.save_attn_weights:
                raw_attn_weight = torch.cat(
                    [output["raw_attn_weight"] for output in curr_outputs]
                )
                adj_mat_learned = torch.cat(
                    [output["adj_mat_learned"] for output in curr_outputs]
                )
                features = torch.cat(
                    [output["features"] for output in curr_outputs]
                )
                outputs_dict = {}
                outputs_dict["raw_attn_weight"] = raw_attn_weight
                outputs_dict["adj_mat_learned"] = adj_mat_learned
                outputs_dict["features"] = features

                with open(
                    os.path.join(
                        self.args.save_dir, "{}_attention_weights.pkl".format(prefix)
                    ),
                    "wb",
                ) as pf:
                    pickle.dump(outputs_dict, pf)

    def _shared_step(self, batch, batch_idx=None):
        batch = batch.to(self.device)

        y = batch.y  
        if y.shape[-1] == 1: 
            y = y.view(-1)
        else:
            seq_len = None

        raw_attn_weight = []
        adj_mat_learned = []
        features = []
        reg_loss = None              
        if (self.args.model_name == "graph_fusion"):
            if self.args.save_attn_weights:
                logits, reg_loss_dict, raw_attn_weight, adj_mat_learned, features = self.model(
                    batch, 
                    return_attention=True, 
                    lengths=seq_len,
                    epoch=self.current_epoch, 
                    epoch_total=self.args.num_epochs,
                )
            else:
                logits, reg_loss_dict = self.model(
                    batch,
                    return_attention=False, 
                    lengths=seq_len, 
                    epoch=self.current_epoch, 
                    epoch_total=self.args.num_epochs,
                )

            reg_loss = self._aggregate_regularization_losses(reg_loss_dict)
        elif (self.args.model_name == "feature_fusion"):
            if self.args.save_attn_weights:
                logits, reg_loss_dict, raw_attn_weight, adj_mat_learned, features = self.model(
                    batch, 
                    return_attention=True, 
                    lengths=seq_len,
                    epoch=self.current_epoch, 
                    epoch_total=self.args.num_epochs,
                )
            else:
                logits, reg_loss_dict = self.model(
                    batch,
                    return_attention=False, 
                    lengths=seq_len, 
                    epoch=self.current_epoch, 
                    epoch_total=self.args.num_epochs,
                )

            reg_loss = self._aggregate_regularization_losses(reg_loss_dict)
        elif self.args.model_name == "s4" or self.args.model_name == "lstm":
            x = (
                batch.x.reshape(
                    -1,
                    self.args.num_nodes,
                    self.args.max_seq_len,
                    self.args.input_dim,
                )
                .transpose(1, 2)
                .reshape(
                    -1, self.args.max_seq_len, self.args.num_nodes * self.args.input_dim
                )
            )  # (batch, seq_len, num_nodes*input_dim)
            logits = self.model(x, lengths=seq_len)
        else:
            logits = self.model(batch, lengths=seq_len)
        if self.args.task == "classification": 
            if num_classes == 1
                cls_loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1),
                    y,
                    pos_weight=torch.FloatTensor(self.args.pos_weight).to(
                        self.device
                    ) if (self.args.pos_weight is not None) else None,
                )
            if num_classes == 4
                self.class_weights = torch.tensor([2.0, 8.0, 10.0, 20.0]).to(self.device)
    
                cls_loss = F.cross_entropy(
                    logits,
                    y.long(),
                    weight=self.class_weights
                )
            
        else:
            raise NotImplementedError

        return (
            logits,
            y,
            cls_loss,
            reg_loss,            
            batch.writeout_fn,
            raw_attn_weight,
            adj_mat_learned,
            features,
        )

    def _freeze_s4(self, model):
        print("Loading S4 pretrained weights...")
        with open(os.path.join(self.args.s4_pretrained_dir, "args.json"), "r") as jf:
            args = json.load(jf)
        args = DottedDict(args)
        checkpoint_file = [fn for fn in os.listdir(self.args.s4_pretrained_dir) if ".ckpt" in fn][0] 
        checkpoint_file = os.path.join(self.args.s4_pretrained_dir, checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        state_dict = checkpoint["state_dict"]
        state_dict = {k.split("model.")[-1]:v for k,v in state_dict.items()}
        # remove encoder and decoder
        state_dict = {k:v for k,v in state_dict.items() if ("encoder" not in k) and ("decoder" not in k)}
        # load state dict
        model.t_model.load_state_dict(state_dict, strict=False)
        # freeze pretrained
        for name, param in model.t_model.named_parameters():
            if "encoder" in name:
                continue
            param.requires_grad = False
        
        return model
    

    def configure_optimizers(self):                          #配置优化器和学习率调度器
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = optim.AdamW(
                params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError

        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)
        elif self.scheduler_name == "one_cycle":
            print("steps_per_epoch:", self.steps_per_epoch)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=self.args.num_epochs,
            )
        elif self.scheduler_name == "timm_cosine":
            scheduler = TimmCosineLRScheduler(optimizer, **self.scheduler_kwargs)
        else:
            raise NotImplementedError

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main(args):
    torch.autograd.set_detect_anomaly(True)                        #异常检测
    # 配置 TensorBoardLogger
    logger = TensorBoardLogger("logs", name="my_model", version=f"version_{args.rand_seed}")

    # random seed
    pl.seed_everything(args.rand_seed, workers=True)

    # Get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False
    )
    save_dir = args.save_dir
    # Save args
    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Build dataset
    print("Building dataset...")
    scaler = None
    if args.dataset == "tuh":                                      #癫痫检测数据集
        datamodule = TUH_DataModule(
            preproc_save_dir=args.preproc_dir,
            raw_data_path=args.raw_data_dir,
            seq_len=args.max_seq_len,
            num_nodes=args.num_nodes,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            adj_mat_dir=args.adj_mat_dir,
            standardize=True,
            balanced_sampling=args.balanced_sampling,
            pin_memory=True,
        )
    else:
        raise NotImplementedError
 
    if args.load_model_path is not None:
        pl_model = PLModel.load_from_checkpoint(
            args.load_model_path,
            args=args,
            lr=args.lr_init,
            weight_decay=args.l2_wd,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            steps_per_epoch=len(datamodule.train_dataloader()),
            scaler=scaler,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            cycle_decay=args.cycle_decay,
            warmup_lr_init=args.warmup_lr_init,
            warmup_t=args.warmup_t,
            cycle_limit=args.cycle_limit,
        )
    else:
        pl_model = PLModel(
            args,
            lr=args.lr_init,
            weight_decay=args.l2_wd,
            optimizer_name=args.optimizer,
            scheduler_name=args.scheduler,
            steps_per_epoch=len(datamodule.train_dataloader()),
            scaler=scaler,
            t_initial=args.t_initial,
            lr_min=args.lr_min,
            cycle_decay=args.cycle_decay,
            warmup_lr_init=args.warmup_lr_init,
            warmup_t=args.warmup_t,
            cycle_limit=args.cycle_limit,
        )

    if args.do_train:                                              
        checkpoint_callback = ModelCheckpoint(   
            monitor="val/{}".format(args.metric_name), 
            mode="max" if args.maximize_metric else "min",            #根据指标类型 选择最大化还是最小化指标
            dirpath=args.save_dir, 
            save_last=True, 
            save_top_k=1,                                             #保存最好的k个模型 1
            auto_insert_metric_name=False,
        )

        early_stopping_callback = EarlyStopping(                      #配置早停回调，用于在验证损失不再下降时提前停止训练
            monitor="val/loss", mode="min", patience=args.patience
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

        if not (args.gpus > 1):   
            trainer = pl.Trainer(
                logger=logger,           
                accelerator="gpu",
                max_epochs=args.num_epochs,
                max_steps=-1,    
                enable_progress_bar=True,
                callbacks=[  
                    checkpoint_callback,
                    early_stopping_callback,
                    lr_monitor,
                ],
                benchmark=False,
                num_sanity_val_steps=0,
                devices=args.gpu_id, 
                accumulate_grad_batches=args.accumulate_grad_batches,  
            )
        else:                                                           #配置多gpu训练
            trainer = pl.Trainer(
                logger=logger,
                accelerator="gpu",
                strategy=pl.strategies.DDPSpawnStrategy(
                    find_unused_parameters=False
                ),
                replace_sampler_ddp=False,
                max_epochs=args.num_epochs,
                max_steps=-1,
                enable_progress_bar=True,
                callbacks=[
                    checkpoint_callback,
                    early_stopping_callback,
                    lr_monitor,
                ],
                benchmark=False,
                num_sanity_val_steps=0,
                devices=torch.cuda.device_count(),
                accumulate_grad_batches=args.accumulate_grad_batches,
            )

        trainer.fit(pl_model, datamodule=datamodule)
        print("Training DONE.")

        # 使用最佳验证模型进行测试
        trainer.test(                   
            model=pl_model,
            ckpt_path="best",
            dataloaders=[datamodule.val_dataloader(), datamodule.test_dataloader()],
        )

    else:  
        trainer = pl.Trainer(
            logger=logger, 
            accelerator="gpu",
            devices=args.gpu_id,
        )

        trainer.test(
            model=pl_model,
            ckpt_path=args.load_model_path,
            dataloaders=[datamodule.val_dataloader(), datamodule.test_dataloader()],
        )

if __name__ == "__main__":
    num_classes = 1
    main(get_args())
