# Pending upstream PR: https://github.com/pytorch/pytorch/pull/201621

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal mmap backport for trusted local DCP checkpoints from PyTorch PR #201621.

Remove this module after the pinned PyTorch dependency includes the PR.
"""

from __future__ import annotations

import collections
import io
import os
import zipfile

import torch
import torch.distributed._shard._utils as shard_utils
import torch.distributed.checkpoint.filesystem
import torch.distributed.checkpoint.planner
import torch.distributed.checkpoint.utils
import torch.futures
import torch.serialization

_ORIGINAL_INIT = torch.distributed.checkpoint.filesystem.FileSystemReader.__init__
_ORIGINAL_READ_DATA = torch.distributed.checkpoint.filesystem.FileSystemReader.read_data


class _UnsupportedZipRegion(Exception):  # noqa: N818
    """Compressed or encrypted storage. Caller falls back to torch.load."""


class _BoundedReader(torch.distributed.checkpoint.utils._ReaderView):
    """Correct SEEK_END for one blob. Not installed on the global _ReaderView."""

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int:
        if whence == os.SEEK_CUR:
            offset += self.tell()
        elif whence == os.SEEK_END:
            offset += self.len
        elif whence != os.SEEK_SET:
            raise ValueError(f"invalid whence: {whence}")
        if offset < 0:
            raise ValueError(f"negative seek value {offset}")
        self.base_stream.seek(self.offset + offset)
        return offset

    def read(self, size=-1):
        max_size = self.len - self.tell()
        if max_size <= 0:
            return b""
        if size < 0 or size > max_size:
            size = max_size
        return self.base_stream.read(size)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _load_zip_region(zip_view, overall_storage, storage_base_offset: int):
    position = zip_view.tell()
    view = _BoundedReader(zip_view.base_stream, zip_view.offset, zip_view.len)
    try:
        region_length = view.seek(0, os.SEEK_END)
        view.seek(0)
        with zipfile.ZipFile(view) as archive:
            for info in archive.infolist():
                if not info.filename.split("/", 1)[-1].startswith("data/"):
                    continue
                if info.flag_bits & 0x1 or info.compress_type != zipfile.ZIP_STORED:
                    raise _UnsupportedZipRegion
        view.seek(0)
        with torch.serialization._open_zipfile_reader(view) as zip_file:  # pyrefly: ignore [bad-argument-type]
            # Offsets from this reader are relative to the blob, matching the slice.
            region = overall_storage[storage_base_offset : storage_base_offset + region_length]
            return torch.serialization._load(
                zip_file,
                "cpu",
                torch.serialization._weights_only_unpickler,
                overall_storage=region,
                weights_only=True,
            )
    finally:
        zip_view.seek(position)


def _patched_init(self, path, *args, mmap: bool = False, **kwargs) -> None:
    _ORIGINAL_INIT(self, path, *args, **kwargs)
    self.mmap = mmap


def _patched_read_data(self, plan, planner):
    storage_data = self.storage_data
    if (
        not self.mmap
        or type(self.fs) is not torch.distributed.checkpoint.filesystem.FileSystem
        or any(storage_data[item.storage_index].transform_descriptors or () for item in plan.items)
    ):
        return _ORIGINAL_READ_DATA(self, plan, planner)

    per_file = collections.defaultdict(list)
    for item in plan.items:
        per_file[storage_data[item.storage_index].relative_path].append(item)

    for relative_path, reqs in per_file.items():
        path = self.fs.concat_path(self.path, relative_path)
        with self.fs.create_stream(path, "rb") as stream:
            mapped = None
            file_size = None
            for req in reqs:  # pyrefly: ignore [bad-assignment]
                item_md = storage_data[req.storage_index]
                blob = self._slice_file(stream, item_md)
                if req.type == torch.distributed.checkpoint.planner.LoadItemType.BYTE_IO:
                    planner.load_bytes(req, io.BytesIO(blob.read(-1)))
                    continue
                if file_size is None:
                    file_size = os.fstat(stream.fileno()).st_size
                if item_md.offset < 0 or item_md.length <= 0 or item_md.offset + item_md.length > file_size:
                    raise RuntimeError("DCP storage range is outside the backing file")
                if mapped is None:
                    mapped = torch.UntypedStorage.from_file(os.fspath(path), shared=False, nbytes=file_size)
                try:
                    tensor = _load_zip_region(blob, mapped, item_md.offset)
                except _UnsupportedZipRegion:
                    blob.seek(0)
                    tensor = torch.load(  # pyrefly: ignore [bad-argument-type]
                        blob, map_location="cpu", weights_only=True
                    )
                tensor = shard_utils.narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                target = planner.resolve_tensor(req).detach()
                if target.size() != tensor.size():
                    raise AssertionError(f"req {req.storage_index} mismatch sizes {target.size()} vs {tensor.size()}")
                target.copy_(tensor)
                planner.commit_tensor(req, target)

    fut = torch.futures.Future()
    fut.set_result(None)
    return fut


def apply() -> None:
    torch.distributed.checkpoint.filesystem.FileSystemReader.__init__ = _patched_init
    torch.distributed.checkpoint.filesystem.FileSystemReader.read_data = _patched_read_data


apply()
