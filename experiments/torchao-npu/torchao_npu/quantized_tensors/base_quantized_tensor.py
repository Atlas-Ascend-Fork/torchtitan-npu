# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common base for the frozen, quantized NPU tensor subclasses."""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao_npu import TorchAONpuBaseTensor
from torchao_npu.quantization import _SUPPORTED_HP_DTYPES

aten = torch.ops.aten


class BaseQuantizedTensor(TorchAONpuBaseTensor):
    """Common base for frozen, quantized NPU tensor subclasses.

    Not intended to be instantiated directly.

    Op behavior inherited by subclasses:
        * Values are frozen: only in-place copy as a whole is supported.

        * ``__torch_function__`` is disabled, so every op routes straight to
          ``__torch_dispatch__``, bypassing the Python function layer.

        * Op support is a whitelist: aten ops below are customized
            - aten._to_copy.default
            - aten.is_pinned.default
            - aten.as_strided.default
    """

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __init_subclass__(cls, **kwargs):
        """``dtype`` is enforced in ``tensor_attribute_names``"""

        super().__init_subclass__(**kwargs)

        names = getattr(cls, "tensor_attribute_names", None)
        if names is None or "dtype" not in names:
            raise TypeError(f"{cls.__name__} must declare ``dtype`` in ``tensor_attribute_names``.")

    def is_contiguous(self, memory_format: torch.memory_format = torch.contiguous_format) -> bool:
        """Report whether every component is contiguous in ``memory_format``.

        A quantized tensor with no components at all is contiguous in every layout.

        Args:
            memory_format: The layout to test, ``torch.contiguous_format`` when omitted.

        Returns:
            Whether all of the tensor's components are contiguous in ``memory_format``.
        """
        # pyrefly: ignore [missing-attribute]
        names = [*self.tensor_data_names, *getattr(self, "optional_tensor_data_names", [])]
        return all(getattr(self, name).is_contiguous(memory_format=memory_format) for name in names)


implements = BaseQuantizedTensor.implements


@implements(aten._to_copy.default)
def _(func, types, args, kwargs):
    """The op behind ``Tensor.to``.

    For dtype, only values in `_SUPPORTED_HP_DTYPES` are supported.

    Raises:
        ValueError: If the requested dtype is not a supported logical dtype.
    """
    x = args[0]
    dtype = (kwargs or {}).pop("dtype", None)
    if not (dtype is None or dtype is x.dtype):
        if dtype not in _SUPPORTED_HP_DTYPES:
            raise ValueError(
                f"{type(x).__name__} only supports the logical dtypes {_SUPPORTED_HP_DTYPES} for ``.to``, got {dtype}."
            )

        data = [getattr(x, name) for name in x.tensor_data_names]
        attributes = [dtype if name == "dtype" else getattr(x, name) for name in x.tensor_attribute_names]
        optional_data = {name: getattr(x, name) for name in getattr(x, "optional_tensor_data_names", [])}
        optional_attributes = {name: getattr(x, name) for name in getattr(x, "optional_tensor_attribute_names", [])}

        args = (type(x)(*data, *attributes, **optional_data, **optional_attributes), *args[1:])

    return TorchAONpuBaseTensor._ATEN_OP_TABLE[TorchAONpuBaseTensor][func](func, types, args, kwargs)


@implements(aten.is_pinned.default)
def _(func, types, args, kwargs):
    """Report whether the quantized tensor's storage is pinned.

    A quantized tensor has no storage of its own: the answer is derived from its
    components. So the answer from components must be uniform. Otherwise, a
    ``RuntimeError`` is raised. A quantized tensor with no components at all holds
    no storage, so it reports ``False``.
    """
    x = args[0]
    names = [*x.tensor_data_names, *getattr(x, "optional_tensor_data_names", [])]
    if not names:
        return False

    first, *rest = names
    expected = getattr(x, first).is_pinned()

    if any(getattr(x, name).is_pinned() is not expected for name in rest):
        raise RuntimeError(f"{type(x).__name__} components must agree on ``is_pinned``.")

    return expected


@implements(aten.as_strided.default)
def _(func, types, args, kwargs):
    """
    Accept only the layout-preserving ``as_strided`` arguments, in which case
    this aten is a no-op.

    If the arguments imply a different layout, a ``NotImplementedError`` is raised.
    """
    x = args[0]
    size = args[1] if len(args) > 1 else kwargs.get("size")
    stride = args[2] if len(args) > 2 else kwargs.get("stride")
    storage_offset = args[3] if len(args) > 3 else kwargs.get("storage_offset")
    offset = x.storage_offset() if storage_offset is None else storage_offset

    if tuple(size) != tuple(x.shape) or tuple(stride) != tuple(x.stride()) or offset != x.storage_offset():
        raise NotImplementedError(
            f"{type(x).__name__} only supports the layout-preserving ``as_strided`` "
            f"(size={tuple(x.shape)}, stride={tuple(x.stride())}, storage_offset={x.storage_offset()}), "
            f"got size={tuple(size)}, stride={tuple(stride)}, storage_offset={offset}."
        )

    return return_and_correct_aliasing(func, args, kwargs, x._apply_fn_to_data(lambda t: t))
