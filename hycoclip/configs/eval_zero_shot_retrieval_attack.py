#---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#---------------------------------------

from hycoclip.config import LazyCall as L
from hycoclip.evaluation.attack import ZeroShotRetrievalAttackEvaluator
import numpy as np

# 0.00314
# 0.00627
# 0.00941
# 0.01255
# 0.03137

evaluator = L(ZeroShotRetrievalAttackEvaluator)(
    datasets=["coco", "flickr30k"],
    data_dir="/path/to/dataset",
    image_size=224,
    eps=0.03137,
    norm=np.inf,
    method="agsm"
)
