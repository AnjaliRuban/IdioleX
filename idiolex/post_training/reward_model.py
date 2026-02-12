"""Reward model for post-training using IdioleX embeddings.

This module wraps a trained IdioleX model to provide reward
signals for post-training.
"""

from collections.abc import Callable
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.centering import MeanCenterer
from src.layer_pool import LayerwiseAttention
from src.utils import average_pool, last_token_pool
from transformers import AutoConfig, AutoModel, AutoTokenizer


class RewardModel(nn.Module):
    """Reward model using IdioleX embedding similarity.

    Computes rewards based on the cosine similarity between ground truth and
    completion embeddings from a trained IdioleX model.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        embedding_model: Union[nn.Module, None] = None,
        centering_model: Union[nn.Module, None] = None,
        max_length: int = 512,
    ) -> None:
        """Initialize the reward model.

        Args:
            model: Trained encoder model.
            tokenizer: Tokenizer for the encoder.
            embedding_model: Optional layerwise attention model.
            centering_model: Optional mean centering model.
            max_length: Maximum sequence length for tokenization.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.centering_model = centering_model
        self.max_length = max_length

        # Freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if self.embedding_model is not None:
            self.embedding_model.eval()
            for param in self.embedding_model.parameters():
                param.requires_grad = False

    def encode(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Encode a list of texts into embeddings.

        Args:
            texts: List of text strings.
            device: Device to run computation on.

        Returns:
            Normalized embeddings of shape [batch_size, hidden_dim].
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Get model outputs
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Pool embeddings
        if self.embedding_model is not None:
            embeddings = average_pool(
                tokens=input_ids,
                embeddings=self.embedding_model(
                    hidden_states=outputs.hidden_states,
                    attention_mask=attention_mask,
                ),
                attention_mask=attention_mask,
            )
        else:
            embeddings = last_token_pool(
                outputs.last_hidden_state,
                attention_mask=attention_mask,
            )

        # Apply centering and normalize
        if self.centering_model is not None:
            embeddings = self.centering_model.apply(embeddings)
        else:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def forward(
        self,
        prompts: list[str],
        completions: list[str],
        ground_truth: list[str],
    ) -> torch.Tensor:
        """Compute reward scores for ground truth-completion pairs.

        The reward is the cosine similarity between the ground truth and
        completion embeddings, measuring stylistic/dialectal consistency.

        Args:
            prompts: List of prompt texts. (Not used in current implementation.)
            completions: List of completion texts.
            ground_truth: List of ground truth completion texts.

        Returns:
            Reward scores of shape [batch_size].
        """
        assert len(ground_truth) == len(
            completions
        ), "Number of ground truth and completions must match"

        device = next(self.model.parameters()).device

        # Encode prompts and completions
        completion_embeddings = self.encode(completions, device)
        ground_truth_embeddings = self.encode(ground_truth, device)

        # Compute cosine similarity as reward
        rewards = torch.sum(ground_truth_embeddings * completion_embeddings, dim=-1)

        return rewards

    def get_reward_function(self) -> Callable:
        """Get a reward function compatible with TRL trainers.

        Returns:
            Callable that takes prompts, completions, and returns rewards.
        """

        def reward_fn(
            prompts: list[str],
            completions: list[str],
            ground_truth: list[str],
            **kwargs,
        ) -> list[float]:
            """Compute rewards for post-training.

            Args:
                prompts: List of prompt texts. (Not used in current implementation.)
                completions: List of completion texts.
                ground_truth: List of ground truth completion texts.
                **kwargs: Additional arguments (ignored).

            Returns:
                List of reward scores.
            """
            with torch.no_grad():
                rewards = self.forward(prompts, completions, ground_truth)
            return rewards.cpu().tolist()

        return reward_fn

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Union[torch.device, str] = "cpu",
    ) -> "RewardModel":
        """Load a reward model from a training checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            device: Device to load the model on.

        Returns:
            Initialized RewardModel.
        """
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location=device
        )
        args = checkpoint["args"]

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)

        config = AutoConfig.from_pretrained(args.base_model)
        config.output_hidden_states = True

        model = AutoModel.from_config(config)
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)

        # Load optional components
        embedding_model = None
        if args.layerwise_pooling and checkpoint.get("embedding_model"):
            num_layers = model.config.num_hidden_layers + 1
            embedding_model = LayerwiseAttention(num_layers=num_layers).to(device)
            embedding_model.load_state_dict(checkpoint["embedding_model"])

        centering_model = None
        if args.mean_center and checkpoint.get("centering_model"):
            centering_model = MeanCenterer(dim=config.hidden_size).to(device)
            centering_model.load_state_dict(checkpoint["centering_model"])

        return cls(
            model=model,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            centering_model=centering_model,
        )
