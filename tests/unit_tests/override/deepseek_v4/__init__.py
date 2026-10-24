# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Package marker: keeps the V4 override tests' module names distinct from V4.1's.

``pytest``'s default prepend import mode keys a test module by its basename.  The V4 and
V4.1 override suites both have a ``test_lightning_indexer_fused.py`` and a
``test_smla_teacher.py``, so collecting both directories under one package-less tree raises
"import file mismatch".  This marker makes this directory a package, which is enough to
qualify its module names (the V4.1 directory stays package-less).
"""
