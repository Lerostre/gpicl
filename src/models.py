import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import pytorch_lightning as pl
from torchmetrics import Accuracy
from transformers import GPT2Config, GPT2ForTokenClassification

from typing import OrderedDict, Optional, Tuple

class LightningBase(pl.LightningModule):
    
    loss = nn.CrossEntropyLoss()
    
    def __init__(
        self, *args,
        optimizer=optim.SGD,
        optimizer_kwargs=dict(lr=0.1),
        scheduler=None,
        scheduler_kwargs=dict(),
        num_classes=10,
        input_size=28*28,
        **kwargs
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.num_classes = num_classes
        self.input_size = input_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        
        self.best_valid_acc = 0
    
    def step(self, batch, batch_idx, subset="train", logger=True, **log_params):
        scorer = Accuracy("multiclass", num_classes=self.num_classes).to(self.device)
        image, label = batch
        
        pred = self(image)
        loss = self.loss(pred, label)
        accuracy = scorer(pred, label)
        
        res = {"loss": loss, "accuracy": accuracy}
        if logger:
            self.log(f"{subset}_loss", loss, **log_params)
            self.log(f"{subset}_accuracy", accuracy, **log_params)
        if subset == "predict":
            res["pred"] = pred
            res["true"] = label
        return res
        
    def training_step(self, batch, batch_idx):
        return self.step(
            batch, batch_idx, "train",
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True
        )
    
    def validation_step(self, batch, batch_idx):
        return self.step(
            batch, batch_idx, "valid",
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True
        )
    
    def test_step(self, batch, batch_idx):
        return self.step(
            batch, batch_idx, "test",
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True,
        )
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, batch_idx, "predict", logger=False)

    def configure_optimizers(self):
        config = {}
        optimizer = self.optimizer(
            self.parameters(), **self.optimizer_kwargs
        )
        config["optimizer"] = optimizer
        
        if self.scheduler is not None:
            scheduler = self.scheduler(
                optimizer, **self.scheduler_kwargs
            )
            config["lr_scheduler"] = scheduler
            
        return config
    
    
class MLP(LightningBase):
    
    def __init__(
        self, *args, use_batch_norm=True,
        hid_size=256, n_layers=3, **kwargs
    ):
        super().__init__(*args, **kwargs)
        # self.save_hyperparameters()
        linear_layers = []
        for i in range(n_layers):
            in_size = self.input_size if i == 0 else hid_size
            out_size = self.num_classes if i == n_layers-1 else hid_size
            batch_norm = nn.BatchNorm1d(out_size) if use_batch_norm else nn.Identity()
            linear_block = nn.Sequential(
                nn.Linear(in_size, out_size),
                batch_norm,
                nn.ReLU(),
            )
            linear_layers.append((f"linear_block_{i}", linear_block))
        self.mlp = nn.Sequential(OrderedDict(linear_layers))
    
    def forward(self, *args, **kwargs):
        return self.mlp(*args, **kwargs)
    
    
class LSTM(LightningBase):
    
    def __init__(
        self, *args,
        input_size=794, hidden_size=256,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=batch_first,
                            bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size, self.num_classes)
        
    def forward(self, inputs):
        return self.linear(self.lstm(inputs)[0]).transpose(-2, -1)
    
    
class OPLSTM(LightningBase):
    
    def __init__(
        self, *args,
        input_size=794,
        hidden_size=256,
        batch_first=True,
        bidirectional=False,
        num_layers=1, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_layers = num_layers
        self.lstm_input_size = input_size
        self.lstm_hidden_size = hidden_size
        self.lstms = nn.ModuleList([
            nn.LSTM(input_size, hidden_size,
                    batch_first=batch_first,
                    bidirectional=bidirectional)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_size, self.num_classes)
        
    def forward(self, inputs):
    
        hx, H, B = None, None, None
        for lstm in self.lstms:
            lstm = lstm.to(self.device)
            if hx is None:
                num_directions = 2 if lstm.bidirectional else 1
                h_zeros = torch.zeros(
                    lstm.num_layers*num_directions,
                    inputs.shape[0], lstm.hidden_size,
                    dtype=inputs.dtype, device=inputs.device
                )
                c_zeros = h_zeros.clone()
                hx = (h_zeros, c_zeros)

            output, hx = lstm(inputs, hx) 
            H_new = torch.einsum("ijk,ijl->ikl", output, inputs)/output.size(1)
            if H is None:
                H = H_new
            else:
                H = H + H_new
            if B is None:
                B = output.squeeze(0).mean(dim=0)
            else:
                B = B + output.squeeze(0).mean(dim=0)

        return self.linear(inputs @ H.transpose(-2, -1) + B).transpose(-2, -1)
    
    def __repr__(self):
        repr_str = ",\n    ".join(
            f"{layer}: {module}" for layer, module in zip(
                [f"(lstm x {self.num_layers})", "(linear)"],
                [self.lstms[0].__repr__(), self.linear.__repr__()]
            )
        )
        return f"OPLSTM(\n    {repr_str}\n)"


class Transformer(LightningBase):
    
    def __init__(self, config=None, input_size=28*28+10, *args, **kwargs):
        if config is None:
            config = GPT2Config(
                num_labels=10, hidden_size=32, n_inner=1024,
                n_layer=4, n_head=8, n_positions=100,
            )
        super().__init__(*args, **kwargs)
        self.input_size = input_size
        self.backbone = GPT2ForTokenClassification(config)
        self.config = config
        self.projection = nn.Sequential(
            nn.Linear(self.input_size, self.config.hidden_size),
            nn.LayerNorm(self.config.hidden_size),
            nn.Dropout(p=self.config.attn_pdrop)
        )
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        attn_output = self.backbone(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=self.projection(input_ids),
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return attn_output[0].transpose(-2, -1)