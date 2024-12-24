"""
- This training setup for SAEs is a little bit wired:
    - We are finetuning a Sparse Autoencoder
    - Instruction tokens are masked
    - Hence the number of activations is unknown a priori
    - Activations are however shuffled with each batch
"""



from __future__ import annotations
import json
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
import wandb
from simple_parsing import ArgumentParser
from transformer_lens.hook_points import HookedRootModule

from sae_lens import logger
from sae_lens.config import HfDataset, LanguageModelSAERunnerConfig
from sae_lens.load_model import load_model
from sae_lens.training.geometric_median import compute_geometric_median
from sae_lens.training.sae_trainer import SAETrainer
from sae_lens.training.training_sae import TrainingSAE, TrainingSAEConfig



import contextlib
import json
import os
import warnings
from collections.abc import Generator, Iterator
from typing import Any, Literal, cast

import datasets
import numpy as np
import torch
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from jaxtyping import Float
from requests import HTTPError
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule
from transformers import AutoTokenizer, PreTrainedTokenizerBase
import re

from sae_lens import logger
from sae_lens.config import (
    DTYPE_MAP,
    CacheActivationsRunnerConfig,
    HfDataset,
    LanguageModelSAERunnerConfig,
)
from sae_lens.sae import SAE
from sae_lens.tokenization_and_batching import concat_and_batch_sequences

DEBUG = False

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO: Make an activation store config class to be consistent with the rest of the code.
class ActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    model: HookedRootModule
    dataset: HfDataset
    cached_activations_path: str | None
    cached_activation_dataset: Dataset | None = None
    tokens_column: Literal["tokens", "input_ids", "text", "problem"]
    hook_name: str
    hook_layer: int
    hook_head_index: int | None
    _dataloader: Iterator[Any] | None = None
    _storage_buffer: torch.Tensor | None = None
    device: torch.device

    @classmethod
    def from_cache_activations(
        cls,
        model: HookedRootModule,
        cfg: CacheActivationsRunnerConfig,
    ) -> ActivationsStore:
        """
        Public api to create an ActivationsStore from a cached activations dataset.
        """
        return cls(
            cached_activations_path=cfg.new_cached_activations_path,
            dtype=cfg.dtype,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.model_batch_size,  # get_buffer
            train_batch_size_tokens=cfg.model_batch_size,  # dataloader
            seqpos_slice=(None,),
            device=torch.device(cfg.device),  # since we're sending these to SAE
            # NOOP
            prepend_bos=False,
            hook_head_index=None,
            dataset=cfg.dataset_path,
            streaming=False,
            model=model,
            normalize_activations="none",
            model_kwargs=None,
            autocast_lm=False,
            dataset_trust_remote_code=None,
        )

    @classmethod
    def from_config(
        cls,
        model: HookedRootModule,
        cfg: LanguageModelSAERunnerConfig | CacheActivationsRunnerConfig,
        override_dataset: HfDataset | None = None,
    ) -> ActivationsStore:
        if isinstance(cfg, CacheActivationsRunnerConfig):
            return cls.from_cache_activations(model, cfg)

        cached_activations_path = cfg.cached_activations_path
        # set cached_activations_path to None if we're not using cached activations
        if (
            isinstance(cfg, LanguageModelSAERunnerConfig)
            and not cfg.use_cached_activations
        ):
            cached_activations_path = None

        if override_dataset is None and cfg.dataset_path == "":
            raise ValueError(
                "You must either pass in a dataset or specify a dataset_path in your configutation."
            )

        return cls(
            model=model,
            dataset=override_dataset or cfg.dataset_path,
            streaming=cfg.streaming,
            hook_name=cfg.hook_name,
            hook_layer=cfg.hook_layer,
            hook_head_index=cfg.hook_head_index,
            context_size=cfg.context_size,
            d_in=cfg.d_in,
            n_batches_in_buffer=cfg.n_batches_in_buffer,
            total_training_tokens=cfg.training_tokens,
            store_batch_size_prompts=cfg.store_batch_size_prompts,
            train_batch_size_tokens=cfg.train_batch_size_tokens,
            prepend_bos=cfg.prepend_bos,
            normalize_activations=cfg.normalize_activations,
            device=torch.device(cfg.act_store_device),
            dtype=cfg.dtype,
            cached_activations_path=cached_activations_path,
            model_kwargs=cfg.model_kwargs,
            autocast_lm=cfg.autocast_lm,
            dataset_trust_remote_code=cfg.dataset_trust_remote_code,
            seqpos_slice=cfg.seqpos_slice,
        )

    @classmethod
    def from_sae(
        cls,
        model: HookedRootModule,
        sae: SAE,
        context_size: int | None = None,
        dataset: HfDataset | str | None = None,
        streaming: bool = True,
        store_batch_size_prompts: int = 8,
        n_batches_in_buffer: int = 8,
        train_batch_size_tokens: int = 4096,
        total_tokens: int = 10**9,
        device: str = "cpu",
    ) -> ActivationsStore:
        return cls(
            model=model,
            dataset=sae.cfg.dataset_path if dataset is None else dataset,
            d_in=sae.cfg.d_in,
            hook_name=sae.cfg.hook_name,
            hook_layer=sae.cfg.hook_layer,
            hook_head_index=sae.cfg.hook_head_index,
            context_size=sae.cfg.context_size if context_size is None else context_size,
            prepend_bos=sae.cfg.prepend_bos,
            streaming=streaming,
            store_batch_size_prompts=store_batch_size_prompts,
            train_batch_size_tokens=train_batch_size_tokens,
            n_batches_in_buffer=n_batches_in_buffer,
            total_training_tokens=total_tokens,
            normalize_activations=sae.cfg.normalize_activations,
            dataset_trust_remote_code=sae.cfg.dataset_trust_remote_code,
            dtype=sae.cfg.dtype,
            device=torch.device(device),
            seqpos_slice=sae.cfg.seqpos_slice,
        )

    def __init__(
        self,
        model: HookedRootModule,
        dataset: HfDataset | str,
        streaming: bool,
        hook_name: str,
        hook_layer: int,
        hook_head_index: int | None,
        context_size: int,
        d_in: int,
        n_batches_in_buffer: int,
        total_training_tokens: int,
        store_batch_size_prompts: int,
        train_batch_size_tokens: int,
        prepend_bos: bool,
        normalize_activations: str,
        device: torch.device,
        dtype: str,
        cached_activations_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        autocast_lm: bool = False,
        dataset_trust_remote_code: bool | None = None,
        seqpos_slice: tuple[int | None, ...] = (None,),
    ):
        self.model = model
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.dataset = (
            load_dataset(
                dataset,
                split="train",
                streaming=streaming,
                trust_remote_code=dataset_trust_remote_code,  # type: ignore
            )
            if isinstance(dataset, str)
            else dataset
        )

        if isinstance(dataset, (Dataset, DatasetDict)):
            self.dataset = cast(Dataset | DatasetDict, self.dataset)
            n_samples = len(self.dataset)

            if n_samples < total_training_tokens:
                warnings.warn(
                    f"The training dataset contains fewer samples ({n_samples}) than the number of samples required by your training configuration ({total_training_tokens}). This will result in multiple training epochs and some samples being used more than once."
                )

        self.hook_name = hook_name
        self.hook_layer = hook_layer
        self.hook_head_index = hook_head_index
        self.context_size = context_size
        self.d_in = d_in
        self.n_batches_in_buffer = n_batches_in_buffer
        self.half_buffer_size = n_batches_in_buffer // 2
        self.total_training_tokens = total_training_tokens
        self.store_batch_size_prompts = store_batch_size_prompts
        self.train_batch_size_tokens = train_batch_size_tokens
        self.prepend_bos = prepend_bos
        self.normalize_activations = normalize_activations
        self.device = torch.device(device)
        self.dtype = DTYPE_MAP[dtype]
        self.cached_activations_path = cached_activations_path
        self.autocast_lm = autocast_lm
        self.seqpos_slice = seqpos_slice

        self.n_dataset_processed = 0

        self.estimated_norm_scaling_factor = None

        # Check if dataset is tokenized
        dataset_sample = next(iter(self.dataset))

        # check if it's tokenized
        if "tokens" in dataset_sample:
            self.is_dataset_tokenized = True
            self.tokens_column = "tokens"
        elif "input_ids" in dataset_sample:
            self.is_dataset_tokenized = True
            self.tokens_column = "input_ids"
        elif "text" in dataset_sample:
            self.is_dataset_tokenized = False
            self.tokens_column = "text"
        elif "problem" in dataset_sample:
            self.is_dataset_tokenized = False
            self.tokens_column = "problem"
        else:
            raise ValueError(
                "Dataset must have a 'tokens', 'input_ids', 'text', or 'problem' column."
            )
        if self.is_dataset_tokenized:
            ds_context_size = len(dataset_sample[self.tokens_column])
            if ds_context_size < self.context_size:
                raise ValueError(
                    f"""pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}.
                    The context_size {ds_context_size} is expected to be larger than or equal to the provided context size {self.context_size}."""
                )
            if self.context_size != ds_context_size:
                warnings.warn(
                    f"""pretokenized dataset has context_size {ds_context_size}, but the provided context_size is {self.context_size}. Some data will be discarded in this case.""",
                    RuntimeWarning,
                )
            # TODO: investigate if this can work for iterable datasets, or if this is even worthwhile as a perf improvement
            if hasattr(self.dataset, "set_format"):
                self.dataset.set_format(type="torch", columns=[self.tokens_column])  # type: ignore

            if (
                isinstance(dataset, str)
                and hasattr(model, "tokenizer")
                and model.tokenizer is not None
            ):
                validate_pretokenized_dataset_tokenizer(
                    dataset_path=dataset, model_tokenizer=model.tokenizer
                )
        else:
            warnings.warn(
                "Dataset is not tokenized. Pre-tokenizing will improve performance and allows for more control over special tokens. See https://jbloomaus.github.io/SAELens/training_saes/#pretokenizing-datasets for more info."
            )

        self.iterable_sequences = self._iterate_tokenized_sequences()

        self.cached_activation_dataset = self.load_cached_activation_dataset()

        # TODO add support for "mixed loading" (ie use cache until you run out, then switch over to streaming from HF)

    def _iterate_raw_dataset(
        self,
    ) -> Generator[torch.Tensor | list[int] | str, None, None]:
        """
        Helper to iterate over the dataset while incrementing n_dataset_processed
        """
        for row in self.dataset:
            # typing datasets is difficult
            yield row[self.tokens_column]  # type: ignore
            self.n_dataset_processed += 1

    def _iterate_raw_dataset_tokens(self) -> Generator[torch.Tensor, None, None]:
        """
        Helper to create an iterator which tokenizes raw text from the dataset on the fly
        """
        for row in self._iterate_raw_dataset():
            tokens = (
                self.model.to_tokens(
                    row,
                    truncate=False,
                    move_to_device=False,  # we move to device below
                    prepend_bos=False,
                )
                .squeeze(0)
                .to(self.device)
            )
            assert (
                len(tokens.shape) == 1
            ), f"tokens.shape should be 1D but was {tokens.shape}"
            yield tokens

    def _iterate_tokenized_sequences(self) -> Generator[torch.Tensor, None, None]:
        """
        Generator which iterates over full sequence of context_size tokens
        """
        # If the datset is pretokenized, we will slice the dataset to the length of the context window if needed. Otherwise, no further processing is needed.
        # We assume that all necessary BOS/EOS/SEP tokens have been added during pretokenization.
        if self.is_dataset_tokenized:
            for row in self._iterate_raw_dataset():
                yield torch.tensor(
                    row[
                        : self.context_size
                    ],  # If self.context_size = None, this line simply returns the whole row
                    dtype=torch.long,
                    device=self.device,
                    requires_grad=False,
                )
        # If the dataset isn't tokenized, we'll tokenize, concat, and batch on the fly
        else:
            tokenizer = getattr(self.model, "tokenizer", None)
            bos_token_id = None if tokenizer is None else tokenizer.bos_token_id
            yield from concat_and_batch_sequences(
                tokens_iterator=self._iterate_raw_dataset_tokens(),
                context_size=self.context_size,
                begin_batch_token_id=(bos_token_id if self.prepend_bos else None),
                begin_sequence_token_id=None,
                sequence_separator_token_id=(
                    bos_token_id if self.prepend_bos else None
                ),
            )

    def load_cached_activation_dataset(self) -> Dataset | None:
        """
        Load the cached activation dataset from disk.

        - If cached_activations_path is set, returns Huggingface Dataset else None
        - Checks that the loaded dataset has current has activations for hooks in config and that shapes match.
        """
        if self.cached_activations_path is None:
            return None

        assert self.cached_activations_path is not None  # keep pyright happy
        # Sanity check: does the cache directory exist?
        assert os.path.exists(
            self.cached_activations_path
        ), f"Cache directory {self.cached_activations_path} does not exist. Consider double-checking your dataset, model, and hook names."

        # ---
        # Actual code
        activations_dataset = datasets.load_from_disk(self.cached_activations_path)
        activations_dataset.set_format(
            type="torch", columns=[self.hook_name], device=self.device, dtype=self.dtype
        )
        self.current_row_idx = 0  # idx to load next batch from
        # ---

        assert isinstance(activations_dataset, Dataset)

        # multiple in hooks future
        if not set([self.hook_name]).issubset(activations_dataset.column_names):
            raise ValueError(
                f"loaded dataset does not include hook activations, got {activations_dataset.column_names}"
            )

        if activations_dataset.features[self.hook_name].shape != (
            self.context_size,
            self.d_in,
        ):
            raise ValueError(
                f"Given dataset of shape {activations_dataset.features[self.hook_name].shape} does not match context_size ({self.context_size}) and d_in ({self.d_in})"
            )

        return activations_dataset

    def set_norm_scaling_factor_if_needed(self):
        if self.normalize_activations == "expected_average_only_in":
            self.estimated_norm_scaling_factor = self.estimate_norm_scaling_factor()

    def apply_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor is None:
            raise ValueError(
                "estimated_norm_scaling_factor is not set, call set_norm_scaling_factor_if_needed() first"
            )
        return activations * self.estimated_norm_scaling_factor

    def unscale(self, activations: torch.Tensor) -> torch.Tensor:
        if self.estimated_norm_scaling_factor is None:
            raise ValueError(
                "estimated_norm_scaling_factor is not set, call set_norm_scaling_factor_if_needed() first"
            )
        return activations / self.estimated_norm_scaling_factor

    def get_norm_scaling_factor(self, activations: torch.Tensor) -> torch.Tensor:
        return (self.d_in**0.5) / activations.norm(dim=-1).mean()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, n_batches_for_norm_estimate: int = int(1e3)):
        norms_per_batch = []
        for _ in tqdm(
            range(n_batches_for_norm_estimate), desc="Estimating norm scaling factor"
        ):
            # temporalily set estimated_norm_scaling_factor to 1.0 so the dataloader works
            self.estimated_norm_scaling_factor = 1.0
            acts = self.next_batch()
            self.estimated_norm_scaling_factor = None
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        return np.sqrt(self.d_in) / mean_norm

    def shuffle_input_dataset(self, seed: int, buffer_size: int = 1):
        """
        This applies a shuffle to the huggingface dataset that is the input to the activations store. This
        also shuffles the shards of the dataset, which is especially useful for evaluating on different
        sections of very large streaming datasets. Buffer size is only relevant for streaming datasets.
        The default buffer_size of 1 means that only the shard will be shuffled; larger buffer sizes will
        additionally shuffle individual elements within the shard.
        """
        if isinstance(self.dataset, IterableDataset):
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
        else:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.iterable_dataset = iter(self.dataset)

    def reset_input_dataset(self):
        """
        Resets the input dataset iterator to the beginning.
        """
        self.iterable_dataset = iter(self.dataset)

    @property
    def storage_buffer(self) -> torch.Tensor:
        if self._storage_buffer is None:
            self._storage_buffer = self.get_buffer(self.half_buffer_size)

        return self._storage_buffer

    @property
    def dataloader(self) -> Iterator[Any]:
        if self._dataloader is None:
            self._dataloader = self.get_data_loader()
        return self._dataloader

    def get_batch_tokens(
        self, batch_size: int | None = None, raise_at_epoch_end: bool = False
    ):
        """
        Streams a batch of tokens from a dataset.

        If raise_at_epoch_end is true we will reset the dataset at the end of each epoch and raise a StopIteration. Otherwise we will reset silently.
        """
        if not batch_size:
            batch_size = self.store_batch_size_prompts
        sequences = []
        # the sequences iterator yields fully formed tokens of size context_size, so we just need to cat these into a batch
        for _ in range(batch_size):
            try:
                sequences.append(next(self.iterable_sequences))
            except StopIteration:
                self.iterable_sequences = self._iterate_tokenized_sequences()
                if raise_at_epoch_end:
                    raise StopIteration(
                        f"Ran out of tokens in dataset after {self.n_dataset_processed} samples, beginning the next epoch."
                    )
                sequences.append(next(self.iterable_sequences))

        return torch.stack(sequences, dim=0).to(_get_model_device(self.model))

    def apply_masks_to_activations(self, activations: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        """
        Apply masks to activations.
        Arguments:
            activations: The activations tensor of shape (batches, context, num_layers, d_in).
            masks: The list of masks corresponding to each example.
        Returns:
            Filtered activations tensor.
        """
        batch_size, context_size, d_in = activations.shape
        filtered_activations = []
        for idx, mask in enumerate(masks):
            # Select activations for the current example based on the mask
            mask = mask.bool()
            example_activations = activations[idx]  # Get activations for this example
            filtered_activation = example_activations[mask]
            filtered_activations.append(filtered_activation)
        cat_filtered_activations = torch.cat(filtered_activations,dim = 0)
        number_of_full_batches = len(cat_filtered_activations) // context_size
        cat_filtered_activations = cat_filtered_activations[:number_of_full_batches * context_size]
        final_acts = cat_filtered_activations.reshape(number_of_full_batches, context_size, d_in)
        return final_acts

    @torch.no_grad()
    def get_activations_until_batch_size(self, target_size: int) -> torch.Tensor:
        """
        Keeps loading activations until we have enough non-masked tokens to fill the target batch size.
        
        Args:
            target_size: The desired number of examples in the final batch
        Returns:
            Tensor of shape (target_size, context_size, num_layers, d_in)
        """
        collected_activations = []
        collected_size = 0
        
        while collected_size < target_size:
            # Get next batch of tokens
            batch_tokens = self.get_batch_tokens().to(_get_model_device(self.model))
            
            # Get raw activations
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast_lm):
                layerwise_activations_cache = self.model.run_with_cache(
                    batch_tokens,
                    names_filter=[self.hook_name],
                    stop_at_layer=self.hook_layer + 1,
                    prepend_bos=False,
                    **self.model_kwargs,
                )[1]

            layerwise_activations = layerwise_activations_cache[self.hook_name][:, slice(*self.seqpos_slice)]
            n_batches, n_context = layerwise_activations.shape[:2]

            # Create examples and masks
            batch_examples = []
            for tokens in batch_tokens:
                batch_examples.append({
                    "prompt": self.model.tokenizer.decode(tokens),
                    "tokens": tokens.tolist(),
                })

            # Generate masks for the current batch
            masks = create_masks_for_batch(
                batch_examples, 
                self.model, 
                start_keys=["start_token"], 
                end_keys=["end_token"]
            )

            # Apply masks and reshape activations
            filtered_acts = self.apply_masks_to_activations(layerwise_activations, masks)
            
            collected_activations.append(filtered_acts)
            collected_size += filtered_acts.shape[0]

        # Concatenate all collected activations
        final_activations = torch.cat(collected_activations, dim=0)
        
        # Trim to exact size needed
        final_activations = final_activations[:target_size]
        
        return final_activations

    def apply_masks_to_activations(self, activations: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        """
        Apply masks to activations and reshape to maintain batch structure.
        
        Args:
            activations: Tensor of shape (batch_size, context_size, d_in)
            masks: List of masks of shape (context_size,) for each example
        Returns:
            Filtered activations tensor of shape (N, context_size, d_in) where N might be smaller than batch_size
        """
        batch_size, context_size, d_in = activations.shape
        filtered_activations = []
        
        for idx, mask in enumerate(masks):
            # Select activations for the current example based on the mask
            mask = mask.bool()
            example_activations = activations[idx]  # Shape: (context_size, d_in)
            filtered_activation = example_activations[mask]  # Shape: (masked_size, d_in)
            filtered_activations.append(filtered_activation)
        
        # Concatenate all filtered activations
        cat_filtered_activations = torch.cat(filtered_activations, dim=0)
        
        # Reshape to maintain batch structure
        # We'll create full context-sized batches
        number_of_full_batches = len(cat_filtered_activations) // context_size
        if number_of_full_batches == 0:
            # If we don't have enough for even one batch, pad with zeros
            padded_size = context_size
            padded_activations = torch.zeros(padded_size, d_in, device=cat_filtered_activations.device)
            padded_activations[:len(cat_filtered_activations)] = cat_filtered_activations
            return padded_activations.unsqueeze(0)  # Add batch dimension
        
        # Trim to full batches and reshape
        cat_filtered_activations = cat_filtered_activations[:number_of_full_batches * context_size]
        return cat_filtered_activations.reshape(number_of_full_batches, context_size, d_in)

    @torch.no_grad()
    def get_buffer(
        self,
        n_batches_in_buffer: int,
        raise_on_epoch_end: bool = False,
        shuffle: bool = True,
    ) -> torch.Tensor:
        """
        Modified get_buffer to use the new activation collection method.
        """
        context_size = self.context_size
        training_context_size = len(range(context_size)[slice(*self.seqpos_slice)])
        d_in = self.d_in
        total_size = self.store_batch_size_prompts * n_batches_in_buffer
        num_layers = 1

        if self.cached_activation_dataset is not None:
            return self._load_buffer_from_cached(
                total_size, context_size, num_layers, d_in, raise_on_epoch_end
            )

        # Get activations until we have enough to fill the buffer
        new_buffer = self.get_activations_until_batch_size(total_size)
        
        # Reshape and normalize as before
        new_buffer = new_buffer.reshape(-1, num_layers, d_in)
        if shuffle:
            new_buffer = new_buffer[torch.randperm(new_buffer.shape[0])]

        if self.normalize_activations == "expected_average_only_in":
            new_buffer = self.apply_norm_scaling_factor(new_buffer)

        return new_buffer


    @torch.no_grad()
    def get_activations(self, batch_tokens: torch.Tensor):
        """
        Returns activations of shape (batches, context, num_layers, d_in)

        d_in may result from a concatenated head dimension.
        """

        # Setup autocast if using
        if self.autocast_lm:
            autocast_if_enabled = torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.autocast_lm,
            )
        else:
            autocast_if_enabled = contextlib.nullcontext()

        with autocast_if_enabled:
            layerwise_activations_cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_name],
                stop_at_layer=self.hook_layer + 1,
                prepend_bos=False,
                **self.model_kwargs,
            )[1]

        layerwise_activations = layerwise_activations_cache[self.hook_name][
            :, slice(*self.seqpos_slice)
        ]
        print(layerwise_activations.shape)

        n_batches, n_context = layerwise_activations.shape[:2]
    
        # Create a list of examples from batch tokens
        batch_examples = []
        for tokens in batch_tokens:
            # Assuming tokens is a tensor or list of token IDs
            batch_examples.append({
                "prompt": self.model.tokenizer.decode(tokens),  # Decode tokens to get the prompt
                "tokens": tokens.tolist(),  # Store as list
            })

        # Generate masks for the current batch
        start_argument_keys = ["**Argument**", "**Argument:**", "## Argument", "##  Argument"]
        end_argument_keys = ["**End Argument**", "**End Argument:**", "## End Argument", "\nEnd Argument"]

        masks = create_masks_for_batch(batch_examples, self.model, start_argument_keys, end_argument_keys)


        # Apply the masks to the activations
        layerwise_activations = self.apply_masks_to_activations(layerwise_activations, masks)

        stacked_activations = torch.zeros((n_batches, n_context, 1, self.d_in))

        if self.hook_head_index is not None:
            stacked_activations[:, :, 0] = layerwise_activations[
                :, :, self.hook_head_index
            ]
        elif layerwise_activations.ndim > 3:  # if we have a head dimension
            try:
                stacked_activations[:, :, 0] = layerwise_activations.view(
                    n_batches, n_context, -1
                )
            except RuntimeError as e:
                logger.error(f"Error during view operation: {e}")
                logger.info("Attempting to use reshape instead...")
                stacked_activations[:, :, 0] = layerwise_activations.reshape(
                    n_batches, n_context, -1
                )
        else:
            stacked_activations[:, :, 0] = layerwise_activations

        return stacked_activations

    def _load_buffer_from_cached(
        self,
        total_size: int,
        context_size: int,
        num_layers: int,
        d_in: int,
        raise_on_epoch_end: bool,
    ) -> Float[torch.Tensor, "(total_size context_size) num_layers d_in"]:
        """
        Loads `total_size` activations from `cached_activation_dataset`

        The dataset has columns for each hook_name,
        each containing activations of shape (context_size, d_in).

        raises StopIteration
        """
        assert self.cached_activation_dataset is not None
        # In future, could be a list of multiple hook names
        hook_names = [self.hook_name]
        assert set(hook_names).issubset(self.cached_activation_dataset.column_names)

        if self.current_row_idx > len(self.cached_activation_dataset) - total_size:
            self.current_row_idx = 0
            if raise_on_epoch_end:
                raise StopIteration

        new_buffer = []
        for hook_name in hook_names:
            # Load activations for each hook.
            # Usually faster to first slice dataset then pick column
            _hook_buffer = self.cached_activation_dataset[
                self.current_row_idx : self.current_row_idx + total_size
            ][hook_name]
            assert _hook_buffer.shape == (total_size, context_size, d_in)
            new_buffer.append(_hook_buffer)

        # Stack across num_layers dimension
        # list of num_layers; shape: (total_size, context_size, d_in) -> (total_size, context_size, num_layers, d_in)
        new_buffer = torch.stack(new_buffer, dim=2)
        assert new_buffer.shape == (total_size, context_size, num_layers, d_in)

        self.current_row_idx += total_size
        return new_buffer.reshape(total_size * context_size, num_layers, d_in)



    def get_data_loader(
        self,
    ) -> Iterator[Any]:
        """
        Return a torch.utils.dataloader which you can get batches from.

        Should automatically refill the buffer when it gets to n % full.
        (better mixing if you refill and shuffle regularly).

        """

        batch_size = self.train_batch_size_tokens

        try:
            new_samples = self.get_buffer(
                self.half_buffer_size, raise_on_epoch_end=True
            )
        except StopIteration:
            warnings.warn(
                "All samples in the training dataset have been exhausted, we are now beginning a new epoch with the same samples."
            )
            self._storage_buffer = (
                None  # dump the current buffer so samples do not leak between epochs
            )
            try:
                new_samples = self.get_buffer(self.half_buffer_size)
            except StopIteration:
                raise ValueError(
                    "We were unable to fill up the buffer directly after starting a new epoch. This could indicate that there are less samples in the dataset than are required to fill up the buffer. Consider reducing batch_size or n_batches_in_buffer. "
                )

        # 1. # create new buffer by mixing stored and new buffer
        mixing_buffer = torch.cat(
            [new_samples, self.storage_buffer],
            dim=0,
        )

        mixing_buffer = mixing_buffer[torch.randperm(mixing_buffer.shape[0])]

        # 2.  put 50 % in storage
        self._storage_buffer = mixing_buffer[: mixing_buffer.shape[0] // 2]

        # 3. put other 50 % in a dataloader
        return iter(
            DataLoader(
                # TODO: seems like a typing bug?
                cast(Any, mixing_buffer[mixing_buffer.shape[0] // 2 :]),
                batch_size=batch_size,
                shuffle=True,
            )
        )

    def next_batch(self):
        """
        Get the next batch from the current DataLoader.
        If the DataLoader is exhausted, refill the buffer and create a new DataLoader.
        """
        try:
            # Try to get the next batch
            return next(self.dataloader)
        except StopIteration:
            # If the DataLoader is exhausted, create a new one
            self._dataloader = self.get_data_loader()
            return next(self.dataloader)

    def state_dict(self) -> dict[str, torch.Tensor]:
        result = {
            "n_dataset_processed": torch.tensor(self.n_dataset_processed),
        }
        if self._storage_buffer is not None:  # first time might be None
            result["storage_buffer"] = self._storage_buffer
        if self.estimated_norm_scaling_factor is not None:
            result["estimated_norm_scaling_factor"] = torch.tensor(
                self.estimated_norm_scaling_factor
            )
        return result

    def save(self, file_path: str):
        """save the state dict to a file in safetensors format"""
        save_file(self.state_dict(), file_path)


def validate_pretokenized_dataset_tokenizer(
    dataset_path: str, model_tokenizer: PreTrainedTokenizerBase
) -> None:
    """
    Helper to validate that the tokenizer used to pretokenize the dataset matches the model tokenizer.
    """
    try:
        tokenization_cfg_path = hf_hub_download(
            dataset_path, "sae_lens.json", repo_type="dataset"
        )
    except HfHubHTTPError:
        return
    if tokenization_cfg_path is None:
        return
    with open(tokenization_cfg_path) as f:
        tokenization_cfg = json.load(f)
    tokenizer_name = tokenization_cfg["tokenizer_name"]
    try:
        ds_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # if we can't download the specified tokenizer to verify, just continue
    except HTTPError:
        return
    if ds_tokenizer.get_vocab() != model_tokenizer.get_vocab():
        raise ValueError(
            f"Dataset tokenizer {tokenizer_name} does not match model tokenizer {model_tokenizer}."
        )


def _get_model_device(model: HookedRootModule) -> torch.device:
    if hasattr(model, "W_E"):
        return model.W_E.device
    if hasattr(model, "cfg") and hasattr(model.cfg, "device"):
        return model.cfg.device
    return next(model.parameters()).device


def create_masks_for_batch(batch, model, start_keys, end_keys):
    """
    Creates masks for a batch of examples based on argument positions with robust matching.
    
    Args:
        batch: List of examples, each containing 'prompt' and 'tokens'
        model: Language model with tokenizer
        start_keys: List of strings that mark argument start
        end_keys: List of strings that mark argument end
        
    Returns:
        List of masks (torch.Tensor) for each example in the batch
    """
    # Get special token IDs
    pad_token_id = model.tokenizer.pad_token_id
    eos_token_id = model.tokenizer.eos_token_id
    bos_token_id = model.tokenizer.bos_token_id
    
    masks = []
    
    for batch_idx, example in enumerate(batch):
        prompt = example["prompt"]
        input_ids = example["tokens"]
        
        # Find all start and end positions
        start_positions = [m.start() for key in start_keys for m in re.finditer(re.escape(key), prompt)]
        end_positions = [m.start() for key in end_keys for m in re.finditer(re.escape(key), prompt)]
        
        # Sort positions
        start_positions.sort()
        end_positions.sort()
        
        # Log and handle potential position issues
        if len(start_positions) == 0:
            #print(f"No start positions found in batch example {batch_idx}")
            masks.append(torch.zeros(len(input_ids), dtype=torch.float))
            continue
            
        if len(end_positions) == 0:
            print(f"No end positions found in batch example {batch_idx}")
            masks.append(torch.zeros(len(input_ids), dtype=torch.float))
            continue
            
        if len(start_positions) > 1:
            print(f"Multiple start positions found in batch example {batch_idx}, using first.")
            start_positions = [start_positions[0]]
            
        if len(end_positions) > 1:
            print(f"Multiple end positions found in batch example {batch_idx}, using first.")
            end_positions = [end_positions[0]]
        
        # Extract and tokenize the argument
        argument = prompt[start_positions[0]:end_positions[0]]
        tokenized_argument = model.to_tokens(argument, prepend_bos=False).squeeze().tolist()
        
        # Initialize mask
        mask = torch.zeros(len(input_ids), dtype=torch.float)
        
        # Attempt to create mask with sliding window approach
        success = False
        left_boundary, right_boundary = 0, len(tokenized_argument)
        max_attempts = 10
        attempt = 0
        
        while not success and attempt <= max_attempts:
            # Get current token window
            current_tokens = tokenized_argument[left_boundary:right_boundary]
            
            # Search for the current token window in input_ids
            for j in range(len(input_ids) - len(current_tokens) + 1):
                if input_ids[j:j + len(current_tokens)] == current_tokens:
                    # Set mask values for matched tokens
                    mask[j:j + len(current_tokens)] = 1
                    success = True
                    break
            
            if not success:
                # Adjust boundaries alternately
                if attempt % 2 == 0:
                    left_boundary += 1
                else:
                    right_boundary -= 1
                    
            attempt += 1
        
        if not success:
            print(f"Could not find a match for batch example {batch_idx} after {max_attempts} attempts")
            
        # Mask out special tokens
        for idx, token_id in enumerate(input_ids):
            if token_id in (pad_token_id, eos_token_id, bos_token_id):
                mask[idx] = 0
        
        masks.append(mask)
    
    return masks




class InterruptedException(Exception):
    pass


def interrupt_callback(sig_num: Any, stack_frame: Any):  # noqa: ARG001
    raise InterruptedException()


class SAETrainingRunner:
    """
    Class to run the training of a Sparse Autoencoder (SAE) on a TransformerLens model.
    """

    cfg: LanguageModelSAERunnerConfig
    model: HookedRootModule
    sae: TrainingSAE
    activations_store: ActivationsStore

    def __init__(
        self,
        cfg: LanguageModelSAERunnerConfig,
        override_dataset: HfDataset | None = None,
        override_model: HookedRootModule | None = None,
        override_sae: TrainingSAE | None = None,
    ):
        if override_dataset is not None:
            logger.warning(
                f"You just passed in a dataset which will override the one specified in your configuration: {cfg.dataset_path}. As a consequence this run will not be reproducible via configuration alone."
            )
        if override_model is not None:
            logger.warning(
                f"You just passed in a model which will override the one specified in your configuration: {cfg.model_name}. As a consequence this run will not be reproducible via configuration alone."
            )

        self.cfg = cfg

        if override_model is None:
            self.model = load_model(
                self.cfg.model_class_name,
                self.cfg.model_name,
                device=self.cfg.device,
                model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
            )
        else:
            self.model = override_model

        self.activations_store = ActivationsStore.from_config(
            self.model,
            self.cfg,
            override_dataset=override_dataset,
        )

        if override_sae is None:
            if self.cfg.from_pretrained_path is not None:
                self.sae = TrainingSAE.load_from_pretrained(
                    self.cfg.from_pretrained_path, self.cfg.device
                )
            else:
                self.sae = TrainingSAE(
                    TrainingSAEConfig.from_dict(
                        self.cfg.get_training_sae_cfg_dict(),
                    )
                )
                self._init_sae_group_b_decs()
        else:
            self.sae = override_sae

    def run(self):
        """
        Run the training of the SAE.
        """

        if self.cfg.log_to_wandb:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )

        trainer = SAETrainer(
            model=self.model,
            sae=self.sae,
            activation_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
        )

        self._compile_if_needed()
        sae = self.run_trainer_with_interruption_handling(trainer)

        if self.cfg.log_to_wandb:
            wandb.finish()

        return sae

    def _compile_if_needed(self):
        # Compile model and SAE
        #  torch.compile can provide significant speedups (10-20% in testing)
        # using max-autotune gives the best speedups but:
        # (a) increases VRAM usage,
        # (b) can't be used on both SAE and LM (some issue with cudagraphs), and
        # (c) takes some time to compile
        # optimal settings seem to be:
        # use max-autotune on SAE and max-autotune-no-cudagraphs on LM
        # (also pylance seems to really hate this)
        if self.cfg.compile_llm:
            self.model = torch.compile(
                self.model,
                mode=self.cfg.llm_compilation_mode,
            )  # type: ignore

        if self.cfg.compile_sae:
            backend = "aot_eager" if self.cfg.device == "mps" else "inductor"

            self.sae.training_forward_pass = torch.compile(  # type: ignore
                self.sae.training_forward_pass,
                mode=self.cfg.sae_compilation_mode,
                backend=backend,
            )  # type: ignore

    def run_trainer_with_interruption_handling(self, trainer: SAETrainer):
        try:
            # signal handlers (if preempted)
            signal.signal(signal.SIGINT, interrupt_callback)
            signal.signal(signal.SIGTERM, interrupt_callback)

            # train SAE
            sae = trainer.fit()

        except (KeyboardInterrupt, InterruptedException):
            logger.warning("interrupted, saving progress")
            checkpoint_name = str(trainer.n_training_tokens)
            self.save_checkpoint(trainer, checkpoint_name=checkpoint_name)
            logger.info("done saving")
            raise

        return sae

    # TODO: move this into the SAE trainer or Training SAE class
    def _init_sae_group_b_decs(
        self,
    ) -> None:
        """
        extract all activations at a certain layer and use for sae b_dec initialization
        """

        if self.cfg.b_dec_init_method == "geometric_median":
            layer_acts = self.activations_store.storage_buffer.detach()[:, 0, :]
            # get geometric median of the activations if we're using those.
            median = compute_geometric_median(
                layer_acts,
                maxiter=100,
            ).median
            self.sae.initialize_b_dec_with_precalculated(median)  # type: ignore
        elif self.cfg.b_dec_init_method == "mean":
            layer_acts = self.activations_store.storage_buffer.detach().cpu()[:, 0, :]
            self.sae.initialize_b_dec_with_mean(layer_acts)  # type: ignore

    @staticmethod
    def save_checkpoint(
        trainer: SAETrainer,
        checkpoint_name: str,
        wandb_aliases: list[str] | None = None,
    ) -> None:
        base_path = Path(trainer.cfg.checkpoint_path) / checkpoint_name
        base_path.mkdir(exist_ok=True, parents=True)

        trainer.activations_store.save(
            str(base_path / "activations_store_state.safetensors")
        )

        if trainer.sae.cfg.normalize_sae_decoder:
            trainer.sae.set_decoder_norm_to_unit_norm()

        weights_path, cfg_path, sparsity_path = trainer.sae.save_model(
            str(base_path),
            trainer.log_feature_sparsity,
        )

        # let's over write the cfg file with the trainer cfg, which is a super set of the original cfg.
        # and should not cause issues but give us more info about SAEs we trained in SAE Lens.
        config = trainer.cfg.to_dict()
        with open(cfg_path, "w") as f:
            json.dump(config, f)

        if trainer.cfg.log_to_wandb:
            # Avoid wandb saving errors such as:
            #   ValueError: Artifact name may only contain alphanumeric characters, dashes, underscores, and dots. Invalid name: sae_google/gemma-2b_etc
            sae_name = trainer.sae.get_name().replace("/", "__")

            # save model weights and cfg
            model_artifact = wandb.Artifact(
                sae_name,
                type="model",
                metadata=dict(trainer.cfg.__dict__),
            )
            model_artifact.add_file(str(weights_path))
            model_artifact.add_file(str(cfg_path))
            wandb.log_artifact(model_artifact, aliases=wandb_aliases)

            # save log feature sparsity
            sparsity_artifact = wandb.Artifact(
                f"{sae_name}_log_feature_sparsity",
                type="log_feature_sparsity",
                metadata=dict(trainer.cfg.__dict__),
            )
            sparsity_artifact.add_file(str(sparsity_path))
            wandb.log_artifact(sparsity_artifact)


def _parse_cfg_args(args: Sequence[str]) -> LanguageModelSAERunnerConfig:
    if len(args) == 0:
        args = ["--help"]
    parser = ArgumentParser()
    parser.add_arguments(LanguageModelSAERunnerConfig, dest="cfg")
    return parser.parse_args(args).cfg


# moved into its own function to make it easier to test
def _run_cli(args: Sequence[str]):
    cfg = _parse_cfg_args(args)
    SAETrainingRunner(cfg=cfg).run()

total_training_steps = 200  # probably we should do more

batch_size = 1024
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = 0
lr_decay_steps = total_training_steps // 5  # 20% of training
l1_warm_up_steps = total_training_steps // 20  # 5% of training

if __name__ == "__main__":
    from sae_lens import HookedSAETransformer
    
    model = HookedSAETransformer.from_pretrained("gemma-2-2b-it")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-scope-2b-pt-res-canonical",
        sae_id = "layer_10/width_16k/canonical",
    )
    cfg = LanguageModelSAERunnerConfig(
            is_dataset_tokenized=True,
            architecture="jumprelu",
            d_in=2304,
            d_sae=16384,
            apply_b_dec_to_input=False,
            context_size=768,
            store_batch_size_prompts = 2,
            n_batches_in_buffer=2,           
            train_batch_size_tokens = 512,
            model_name="gemma-2-2b-it",
            hook_name="blocks.10.hook_resid_post",
            hook_layer=10,
            hook_head_index=None,
            prepend_bos=False,
            dataset_path="gboxo/control_dataset_gemma2_tokenized_padded",
            dataset_trust_remote_code=True,
            dtype="float32",
            training_tokens=total_training_tokens,
            log_to_wandb=True,  # always use wandb unless you are just testing code.
            #act_store_device=cuda:3,
            device=device,
            #device="cuda:2"
            #model_from_pretrained_kwargs={"n_devices": 2},
            wandb_project="Finetuning SAE",
            wandb_log_frequency=10,
            eval_every_n_wandb_logs=5,
            n_checkpoints=5,
            )
    # TrainingSAE
    train_sae = TrainingSAE(
            TrainingSAEConfig.from_dict(
                cfg.get_training_sae_cfg_dict(),
            )
        )
    sae_state_dict = sae.state_dict()
    sae_state_dict["log_threshold"] = sae_state_dict["threshold"].log() 
    del sae_state_dict["threshold"]
    train_sae.load_state_dict(sae_state_dict)
    del sae

    """
    activations_store = ActivationsStore.from_config(
            model=model,
            cfg=cfg,
            )

    sparse_autoencoder.activations_store = activations_store
    """
    # SAETrainer
    sparse_autoencoder = SAETrainingRunner(
            cfg=cfg,
            override_model=model,
            override_sae=train_sae,
            )
    
    sparse_autoencoder.run()


