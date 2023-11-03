# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from meru.config import LazyCall as L
from meru.evaluation.classification import ZeroShotClassificationEvaluator


evaluator = L(ZeroShotClassificationEvaluator)(
    datasets_and_prompts={
        "aircraft": [
            "a photo of a {}, a type of aircraft.",
            "a photo of the {}, a type of aircraft.",
        ],
    },
    data_dir="datasets/eval",
    image_size=224,
    num_workers=2,
)
