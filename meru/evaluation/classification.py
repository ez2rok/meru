# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from loguru import logger

from meru import lorentz as L
from meru.evaluation.catalog import DatasetCatalog
from meru.evaluation.class_names import CLASS_NAMES
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer


class ZeroShotClassificationEvaluator:
    """
    Evaluate trained models for zero-shot image classification, wherein the entire
    model is transferred to the downstream task without additional training. This
    protocol is similar to CLIP: the classifier weights are constructed by encoding
    text prompts of class labels using text encoder.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        datasets_and_prompts: dict[str, list[str]],
        data_dir: str | Path,
        num_workers: int = 4,
        image_size: int = 224,
    ):
        """
        Args:
            datasets_and_prompts: Dictionary mapping between dataset name and
                a list of prompt templates to fill using its class names. Add
                a single `{}` in prompt to fill with class name. Datasets
                should be among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog.
            num_workers: Number of CPU works to parallelize data loading for
                extracting features.
            image_size: Resize and crop images to this size for evaluation. We
                resize the smaller image edge (keeping aspect ratio same) using
                bicubic interpolation, and take a square center crop.
        """
        self._datasets_and_prompts = datasets_and_prompts
        self._data_dir = Path(data_dir).resolve()
        self._num_workers = num_workers

        self._image_transform = T.Compose(
            [
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )

    @torch.inference_mode()
    def __call__(self, model: MERU | CLIPBaseline) -> dict[str, float]:
        model = model.eval()
        tokenizer = Tokenizer()

        # Collect results per task in this dict:
        results_dict = {}

        for dname, prompts in self._datasets_and_prompts.items():
            logger.info(f"Zero-shot classification evaluation for {dname}:")
            # ----------------------------------------------------------------
            # Make zero-shot classifier using class name and prompts.
            # ----------------------------------------------------------------
            class_names = CLASS_NAMES[dname]

            # Collect text features of each class.
            all_class_feats: list[torch.Tensor] = []

            for name in class_names:
                # Fill prompt templates with class name and tokenize them.
                class_prompts = [_pt.format(name) for _pt in prompts]

                class_prompt_tokens = tokenizer(class_prompts)
                class_feats = model.encode_text(class_prompt_tokens, project=False)

                if isinstance(model, MERU):
                    # Ensemble in the tangent space, then project to Hyperboloid.
                    class_feats = class_feats.mean(dim=0)
                    class_feats = class_feats * model.textual_alpha.exp()
                    class_feats = L.exp_map0(class_feats, model.curv.exp())
                else:
                    # Ensemble prompt features: normalize -> average -> normalize.
                    class_feats = F.normalize(class_feats, dim=-1)
                    class_feats = class_feats.mean(dim=0)
                    class_feats = F.normalize(class_feats, dim=-1)

                all_class_feats.append(class_feats)

            # shape: (num_classes, embed_dim)
            classifier = torch.stack(all_class_feats, dim=0)
            # ----------------------------------------------------------------

            # Extract image features and labels from the test split of required dataset.
            loader = DataLoader(
                DatasetCatalog.build(
                    dname, self._data_dir, "test", self._image_transform
                ),
                batch_size=128,
                num_workers=self._num_workers,
            )
            image_feats, labels = _encode_dataset(loader, model, project=True)

            # Features returned by this function will be on CPU, move to device:
            image_feats = image_feats.to(model.device)

            # Measure model performance according to accuracy metric of the dataset.
            acc_meter = MulticlassAccuracy(DatasetCatalog.NUM_CLASSES[dname])

            # Evaluate in small batches of 256 instances.
            for _feats, _labels in zip(image_feats.split(256), labels.split(256)):
                # Compute pairwise similarity depending on model type:
                if isinstance(model, MERU):
                    scores = L.pairwise_inner(_feats, classifier, model.curv.exp())
                else:
                    scores = _feats @ classifier.T

                acc_meter(scores.cpu(), _labels)

            accuracy = acc_meter.compute() * 100.0
            results_dict[dname] = accuracy

            logger.info(
                f"Zero-shot classification: {dname}, {len(image_feats)} images, "
                f"{len(class_names)} classes [acc.: {accuracy:.1f}%] "
            )

        return results_dict


class LinearProbeClassificationEvaluator:
    """
    Evaluate trained models for image classification by training logistic regression
    classifiers on top of frozen features from image encoders. Hyperparameter search
    is same as CLIP. For both MERU and CLIP, we evaluate the underlying Euclidean
    features before the projection layer.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        datasets: list[str],
        data_dir: str | Path,
        num_workers: int = 4,
        image_size: int = 224,
        tune_hyperparams: bool = False,
        outdir: str | Path = "embeddings",
    ):
        """
        Args:
            datasets: List of dataset names to evaluate on, these names should be
                among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog.
            num_workers: Number of CPU works to parallelize data loading for
                extracting features.
            image_size: Resize and crop images to this size for evaluation. We
                resize the smaller image edge (keeping aspect ratio same) using
                bicubic interpolation, and take a square center crop.
            tune_hyperparams: If True, perform a hyperparameter sweep over cost
                values for logistic regression using the validation set.
            outdir: Directory to save extracted image features and labels for
                each dataset.
        """
        self._datasets = datasets
        self._data_dir = Path(data_dir).resolve()
        self._num_workers = num_workers
        self.tune_hyperparams = tune_hyperparams
        
        if isinstance(outdir, str):
            outdir = Path(outdir)
        self.outdir = outdir

        self._image_transform = T.Compose(
            [
                T.Resize(image_size, T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
            ]
        )
        self._sk_kwargs = {"random_state": 0, "max_iter": 1000}

    @torch.inference_mode()
    def __call__(
        self,
        model: MERU | CLIPBaseline,
        save: bool = False,
        ) -> dict[str, float]:
        
        # get model from torch DistributedDataParallel object
        # https://github.com/huggingface/transformers/issues/18974#issuecomment-1242985539
        _model = model
        if isinstance(model, DistributedDataParallel):
            _model = model.module
        
        # Make output directory.
        model_name = _model.__class__.__name__.lower()
        model_size = 'small'
        embd_dim = str(_model.visual_proj.weight.shape[0]).zfill(4)
        run_name = f'{model_name}_vit_{model_size}_{embd_dim}'
        outdir = self.outdir / run_name
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Remove projection layer. Now `.encode_image()` will always give Euclidean
        # representations directly from the image encoder, regardless of model type.
        model = model.eval()
        model.visual_proj = torch.nn.Identity()

        # Collect results per task in this dict:
        results_dict = {}
        
        for dname in self._datasets:
            logger.info(f"Linear probe classification evaluation for {dname}:")
            image_feats_path = outdir / f"{dname}_image_features.pth"
            labels_path = outdir / f"{dname}_image_labels.pth"
            
            if image_feats_path.exists() and labels_path.exists():
                logger.info(f"Loading image features and labels from {outdir}")
                image_feats = torch.load(image_feats_path)
                labels = torch.load(labels_path)
            else:
                # Extract image features, labels from [train, val, test] splits.
                logger.info('Computing image and label features.')
                image_feats, labels = {}, {}
                for split in ["train", "val", "test"]:
                    dataset, sampler = DatasetCatalog.build(
                            dname, self._data_dir, split, self._image_transform
                        )
                    loader = DataLoader(
                        dataset,
                        sampler=sampler,
                        batch_size=64, # lowered from 128 to fit in memory
                        num_workers=self._num_workers,
                    )
                    image_feats[split], labels[split] = _encode_dataset(
                        loader, _model, project=False
                    )
                    
                if save:
                    # Save image features and labels to disk.
                    logger.info(f"Saving image features and labels to {outdir}")
                    torch.save(image_feats, image_feats_path)
                    torch.save(labels, labels_path)

            logger.info(
                f"{dname} split sizes: train ({len(labels['train'])}), "
                f"val ({len(labels['val'])}), test ({len(labels['test'])})"
            )
            
            # Find best cost value for logistic regression via hyperparameter tuning.
            best_cost = 1.0            
            if self.tune_hyperparams:
                best_cost = _get_best_cost(image_feats, labels, self._sk_kwargs)
        
            # Train a classifier on (train+val) split with best cost.
            logger.info(f"Training classifier on (train+val) split with best cost.")
            final_classifier = LogisticRegression(C=best_cost, **self._sk_kwargs)
            final_classifier.fit(
                torch.cat([image_feats["train"], image_feats["val"]]),
                torch.cat([labels["train"], labels["val"]]),
            )
            logger.info(f"Fit classifier.")
            logits = torch.as_tensor(final_classifier.predict_proba(image_feats["test"]))
            predictions = torch.argmax(logits, dim=-1)
            final_accuracy = accuracy_score(
                predictions, labels["test"]
                ) * 100.0
            logger.info(f"Evaluation done, {dname} test acc = {final_accuracy:.3f}")

            results_dict[dname] = final_accuracy

        return results_dict


def _encode_dataset(
    data_loader: DataLoader,
    model: MERU | CLIPBaseline,
    project: bool,
):
    """
    Extract image features and labels for a given dataset using the given model.

    Args:
        data_loader: PyTorch dataset or dataloader that serves instances/batches
            of `(image, label)` tuples.
        model: Model that implements `encode_image` method to extract features.
        project: Input argument to `model.encode_image`.
    """

    # Collect batches of extracted image features and labels (as-is from loader).
    all_image_feats, all_labels = [], []

    for images, labels in tqdm(data_loader, desc=f"Extracting image feats"):
        with torch.inference_mode():
            image_feats = model.encode_image(images.to(model.device), project)

        all_image_feats.append(image_feats.cpu())
        all_labels.append(labels)

    return torch.cat(all_image_feats, dim=0), torch.cat(all_labels, dim=0)


def _get_best_cost(
    image_feats: dict[str, torch.Tensor],
    labels: dict[str, torch.Tensor],
    _sk_kwargs: dict[str, int],
) -> float:
    """
    Perform a hyperparameter sweep over cost values for logistic
    regression using the validation set to get the best cost. We follow CLIP and
    perform a parametric binary search in two passes.
    
    Args:
        image_feats: Tensor of shape (num_instances, embed_dim) containing
            image features extracted from the model.
        labels: Tensor of shape (num_instances,) containing integer labels.
        _sk_kwargs: Keyword arguments for
        `sklearn.linear_model.LogisticRegression`.        
    """
   
    costs = [1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]
    logger.info(f"First pass: searching best cost among {costs}...")

    best_cost, best_result = -1, -1

    for cost in costs:
        classifier = LogisticRegression(C=cost, **_sk_kwargs)
        classifier.fit(image_feats["train"], labels["train"])

        logits = torch.as_tensor(classifier.predict_proba(image_feats["val"]))
        predictions = torch.argmax(logits, dim=-1)
        result = accuracy_score(predictions, labels["val"]) * 100.0
        logger.info(f"Cost = {cost:.6f}, Val acc = {result:.3f}")

        # Update best searched cost so far.
        if result > best_result:
            best_cost, best_result = cost, result

    logger.info(f"Best cost from first sweep: {best_cost:.6f}")
    logger.info(f"Second pass: perform binary search around best cost...")

    # Interval width around peak cost value from first sweep. For example,
    # if the best cost is 1000, then interval will be [100, 10000]
    delta = 10.0

    for step in range(1, 5):
        # Check accuracies on the extremes of left-half and right-half
        # interval around the current best cost.
        lc, rc = best_cost / delta, best_cost * delta
        logger.info(
            f"Search step {step}, costs [left = {lc:.6f}, right = {rc:.6f}]"
        )

        for cost in [lc, rc]:
            classifier = LogisticRegression(C=cost, **_sk_kwargs)
            classifier.fit(image_feats["train"], labels["train"])

            logits = torch.as_tensor(classifier.predict_proba(image_feats["val"]))
            predictions = torch.argmax(logits, dim=-1)
            result = accuracy_score(predictions, labels["val"]) * 100.0
            logger.info(f"Cost = {cost:.6f}, Val acc = {result:.3f}")

            # Update best searched cost so far.
            if result > best_result:
                best_cost, best_result = cost, result

        # Half the search interval in log-space, for next search step.
        delta = delta**0.5

    logger.info(f"Best cost = {best_cost:.6f}, Val acc = {best_result:.3f}")
    return best_cost