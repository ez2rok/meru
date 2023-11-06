# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from meru.config import LazyCall as L
from meru.evaluation.classification import ZeroShotClassificationEvaluator


evaluator = L(ZeroShotClassificationEvaluator)(
    datasets_and_prompts={
        "country211": [
            "a photo i took in {}.",
            "a photo i took while visiting {}.",
            "a photo from my home country of {}.",
            "a photo from my visit to {}.",
            "a photo showing the country of {}.",
        ],
    },
    data_dir="datasets/eval",
    image_size=224,
    num_workers=2,
)
