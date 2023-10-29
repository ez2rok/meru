# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from meru.config import LazyCall as L
from meru.evaluation.classification import ZeroShotClassificationEvaluator


evaluator = L(ZeroShotClassificationEvaluator)(
    datasets_and_prompts={
        "cifar100": [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a black and white photo of a {}.",
            "a low contrast photo of a {}.",
            "a high contrast photo of a {}.",
            "a bad photo of a {}.",
            "a good photo of a {}.",
            "a photo of a small {}.",
            "a photo of a big {}.",
            "a photo of the {}.",
            "a blurry photo of the {}.",
            "a black and white photo of the {}.",
            "a low contrast photo of the {}.",
            "a high contrast photo of the {}.",
            "a bad photo of the {}.",
            "a good photo of the {}.",
            "a photo of the small {}.",
            "a photo of the big {}.",
        ],
    },
    data_dir="datasets/eval",
    image_size=224,
    num_workers=1,
)
