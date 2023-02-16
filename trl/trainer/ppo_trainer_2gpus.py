import bitsandbytes as bnb

from trl.trainer.ppo_trainer import PPOTrainer

import logging
import warnings
from typing import List, Optional, Union

import torch
from accelerate import Accelerator
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer, PreTrainedTokenizerFast

from trl.core import (
    logprobs_from_logits,
)

from trl.models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper
from trl.trainer import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig


class PPOTrainer2GPU(PPOTrainer):
    """
    The PPOTrainer2GPU uses Proximal Policy Optimization to optimise language models on 2 GPUs.
    """

    def __init__(
        self,
        config: PPOConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: PreTrainedModelWrapper = None,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        data_collator=None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        frozen_layers=0.80,  # -1 no frozen layers
    ):
        """
        Initialize PPOTrainer.
        Overloading PPOTrainer init to move model and ref_model on different cuda devices:
            - Do not support num_shared_layers as PPOTRainer does (since the two models will be on separated devices)
            - Do not support custom optimizer since it must be created after having moved the model params to cuda

            - Model is copied to cuda:0 (trained model)
            - Reference Model is copied to cuda:1 (inference only)
            - handled automatic device_placement for models to avoid having them automatically moved to other devices
        """
        # skip PPOTrainer init to call its parent init directly
        super(PPOTrainer, self).__init__(config)

        # Step 0: check positional arguments validity
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        if not isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                f"tokenizer must be a PreTrainedTokenizer or PreTrainedTokenizerFast, got {type(tokenizer)}"
            )
        if not isinstance(model, PreTrainedModelWrapper):
            raise ValueError(
                f"model must be a PreTrainedModelWrapper, got {type(model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(log_with=config.log_with, **config.accelerator_kwargs)
        self.accelerator.init_trackers(config.tracker_project_name, config=config.to_dict(), **config.tracker_kwargs)

        if frozen_layers != -1:
            self.freeze_layers(model, frozen_layers=frozen_layers)

        self.model = model
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")

        if isinstance(ref_model, PreTrainedModelWrapper):
            self.ref_model = ref_model
        else:
            raise ValueError(
                f"ref_model must be a PreTrainedModelWrapper, got {type(ref_model)} - supported architectures are: {SUPPORTED_ARCHITECTURES}"
            )

        if not (isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast)):
            raise ValueError(
                "tokenizer must be a transformers.PreTrainedTokenizer or transformers.PreTrainedTokenizerFast"
            )
        self.tokenizer = tokenizer

        if dataset is not None and not (isinstance(dataset, torch.utils.data.Dataset) or isinstance(dataset, Dataset)):
            raise ValueError("dataloader must be a torch.utils.data.Dataset or datasets.Dataset")
        elif dataset is None:
            warnings.warn(
                "No dataset is provided. Make sure to set config.batch_size to the correct value before training.",
                UserWarning,
            )
        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            warnings.warn(
                "No dataset is provided. In a multi-GPU setting, this will lead to an error. You should",
                " prepare your dataloader yourself with `dataloader = ppo_trainer.accelerator.prepare(dataloader)`",
                " and using `torch.utils.data.DataLoader`, or pass a dataset to the `PPOTrainer`. Please ",
                " refer to the documentation for more details.",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        # Step 2: Move model's params to GPUs (must be done before creating the optimizer)
        self.device_model = self.accelerator.device
        self.device_ref_model = "cuda:1"
        if self.device_model == self.device_ref_model:
            raise ValueError(f"model and ref_model are on the same device: {self.device_model}."
                             f"Please change ref_model's device.")

        self.model.to(self.device_model)
        self.ref_model.to(self.device_ref_model)

        # Step 3: Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.learning_rate)

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                raise ValueError("lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler")

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        (
            self.model,
            self.ref_model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.ref_model, self.optimizer,
            self.data_collator, self.dataloader, self.lr_scheduler,
            device_placement=[
                False, False, False,
                True, True, True,
            ]
        )

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        # init the current step
        self.current_step = 0

        # init wandb on the main process:
        if self.accelerator.is_main_process and self.config.log_with == "wandb":
            import wandb

            wandb.watch(self.model, log="all")

        if self.config.forward_batch_size > 1 and (self.is_encoder_decoder or self.tokenizer.padding_side == "left"):
            # warn users that this is not well supported yet
            logging.warning(
                "Forward batch size > 1 is not well supported yet for encoder-decoder models and when using `tokenizer.padding_side='left'`. This can lead to unexpected behaviour."
                " therefore, we recommend using forward_batch_size=1."
            )

    def batched_forward_pass(
        self,
        queries: torch.Tensor,
        responses: torch.Tensor,
    ):
        """
        Calculate model outputs in multiple batches.

        Args:
            queries (`torch.LongTensor`):
                List of tensors containing the encoded queries, shape (`batch_size`, `query_length`)
            responses (`torch.LongTensor`):
                List of tensors containing the encoded responses, shape (`batch_size`, `response_length`)
        Returns:
            (tuple):
                - all_logprobs (`torch.FloatTensor`): Log probabilities of the responses, shape (`batch_size`, `response_length`)
                - all_ref_logprobs (`torch.FloatTensor`): Log probabilities of the responses, shape (`batch_size`, `response_length`)
                - all_values (`torch.FloatTensor`): Values of the responses, shape (`batch_size`, `response_length`)
        """
        if self.is_encoder_decoder:
            raise ValueError("(Stephane) encoder decoder models not supported yet.")

        bs = self.config.batch_size
        fbs = self.config.forward_batch_size
        all_logprobs = []
        all_ref_logprobs = []
        all_values = []

        for i in range(int(bs / fbs)):
            # -----------------------------------------------------------------
            # model forward pass (on self.model_device, most probably "cuda:0")
            # -----------------------------------------------------------------
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]

            input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])[
                "input_ids"
            ]

            input_kwargs = {
                "input_ids": input_ids,
            }

            with torch.no_grad():
                logits, _, v = self.model(**input_kwargs)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

            # -------------------------------------------------------------------------
            # ref_model forward pass (on self.ref_model_device, most probably "cuda:1")
            # -------------------------------------------------------------------------
            query_batch = [tensor.to(self.device_ref_model) for tensor in query_batch]
            response_batch = [tensor.to(self.device_ref_model) for tensor in response_batch]

            input_ids = self.data_collator([torch.cat([q, r]) for q, r in zip(query_batch, response_batch)])[
                "input_ids"
            ]

            input_kwargs = {
                "input_ids": input_ids,
            }

            with torch.no_grad():
                ref_logits, _, _ = self.ref_model(**input_kwargs)

            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], input_ids[:, 1:])

            ref_logprobs = ref_logprobs[0].to(self.device_model)    # move tensor to other device
            ref_logprobs = ref_logprobs.view(1, -1)                 # convert from 1d to 2d tensor

            # -------------------------------------------------------------------------
            for j in range(fbs):
                start = len(query_batch[j]) - 1
                end = len(query_batch[j]) + len(response_batch[j]) - 1

                if len(logprobs[j, start:end]) < 2:
                    raise ValueError("Responses are too short. Make sure they are at least 4 tokens long.")

                all_values.append(v[j, start:end])
                all_logprobs.append(logprobs[j, start:end])
                all_ref_logprobs.append(ref_logprobs[j, start:end])

        return all_logprobs, all_ref_logprobs, all_values

    def freeze_layers(self, model, frozen_layers=0.80):
        # Freeze the first 80% of the hidden layers of the model backbone
        logging.info(f"Freezing {frozen_layers} of model layers.")

        layers = model.pretrained_model.transformer.h
        num_layers = len(layers)
        num_unfrozen = int((1.0 - frozen_layers) * num_layers)
        for layer in layers[:-num_unfrozen]:
            layer.requires_grad_(False)
