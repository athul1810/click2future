"""Retain the published EMA tensors verbatim in a smaller inference archive.

The original checkpoint and its hash remain provenance. No dtype conversion,
quantization, tensor selection within the EMA state, or parameter update occurs.
"""

import argparse
import collections
import hashlib
import io
import json
import pickle
import sys
import zipfile
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare(source, output, expected_sha256):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError("Retain existing artifacts; select a new output path")
    original = source.stat()
    if digest(source) != expected_sha256:
        raise ValueError("Source checkpoint SHA-256 does not match")
    # FakeTensorMode reads the restricted metadata without mapping/dereferencing
    # a 61-GB Windows storage. Actual storage bytes go through Python's ZIP I/O.
    with FakeTensorMode():
        state = torch.load(source, map_location="cpu", weights_only=True)
    ema = state.get("generator_ema")
    if not isinstance(ema, dict) or not ema:
        raise ValueError("No released EMA generator state")
    if any(not isinstance(v, torch.Tensor) or v.layout != torch.strided for v in ema.values()):
        raise ValueError("Expected a dense tensor-only EMA state")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    storage_types = {
        torch.float32: torch.FloatStorage,
        torch.float64: torch.DoubleStorage,
        torch.float16: torch.HalfStorage,
        torch.bfloat16: torch.BFloat16Storage,
        torch.int64: torch.LongStorage,
        torch.int32: torch.IntStorage,
        torch.int16: torch.ShortStorage,
        torch.int8: torch.CharStorage,
        torch.uint8: torch.ByteStorage,
        torch.bool: torch.BoolStorage,
    }
    storages, proxies = {}, {}

    class Storage:
        def __init__(self, key, tensor):
            self.key, self.tensor = key, tensor

    class Tensor:
        def __init__(self, tensor, storage):
            self.tensor, self.storage = tensor, storage

        def __reduce__(self):
            t = self.tensor
            return torch._utils._rebuild_tensor_v2, (
                self.storage,
                t.storage_offset(),
                tuple(t.shape),
                tuple(t.stride()),
                False,
                collections.OrderedDict(),
            )

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, Storage):
                t = obj.tensor
                return (
                    "storage",
                    storage_types[t.dtype],
                    obj.key,
                    "cpu",
                    t.untyped_storage().nbytes() // t.element_size(),
                )
            return None

    for name, tensor in ema.items():
        if tensor.dtype not in storage_types:
            raise ValueError(f"Unsupported released storage dtype: {tensor.dtype}")
        storage = tensor.untyped_storage()
        offset = getattr(storage, "_checkpoint_offset", None)
        if offset is None:
            raise ValueError("PyTorch must expose checkpoint offsets in FakeTensorMode")
        if offset not in storages:
            storages[offset] = Storage(str(len(storages)), tensor)
        proxies[name] = Tensor(tensor, storages[offset])
    buffer = io.BytesIO()
    Pickler(buffer, protocol=2).dump({"generator_ema": proxies})
    tensor_hashes = {}
    with zipfile.ZipFile(source) as archive:
        offsets = {}
        with source.open("rb") as raw:
            for info in archive.infolist():
                raw.seek(info.header_offset + 26)
                sizes = raw.read(4)
                start = (
                    info.header_offset
                    + 30
                    + int.from_bytes(sizes[:2], "little")
                    + int.from_bytes(sizes[2:], "little")
                )
                offsets[start] = info
        writer = torch._C.PyTorchFileWriter(str(partial))
        try:
            metadata = buffer.getvalue()
            writer.write_record("data.pkl", metadata, len(metadata))
            writer.write_record("byteorder", sys.byteorder, len(sys.byteorder))
            for offset, storage in storages.items():
                info = offsets[offset]
                size = storage.tensor.untyped_storage().nbytes()
                if info.file_size != size or size > 1024**3:
                    raise ValueError("Unexpected storage size; original retained")
                data = archive.read(info)  # ZIP verifies the source CRC as well.
                tensor_hashes[storage.key] = hashlib.sha256(data).hexdigest()
                writer.write_record("data/" + storage.key, data, len(data))
                del data
            writer.write_end_of_file()
        finally:
            del writer
    with FakeTensorMode():
        restored = torch.load(partial, map_location="cpu", weights_only=True)
    if restored.keys() != {"generator_ema"} or restored["generator_ema"].keys() != ema.keys():
        raise ValueError("EMA archive keys changed")
    for name, tensor in ema.items():
        other = restored["generator_ema"][name]
        if (
            tensor.dtype != other.dtype
            or tensor.shape != other.shape
            or tensor.stride() != other.stride()
        ):
            raise ValueError(f"EMA tensor metadata changed: {name}")
    with zipfile.ZipFile(partial) as archive:
        prefix = archive.namelist()[0].split("/")[0]
        for key, expected in tensor_hashes.items():
            if hashlib.sha256(archive.read(prefix + "/data/" + key)).hexdigest() != expected:
                raise ValueError("EMA storage bytes changed")
    current = source.stat()
    if (original.st_size, original.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
        raise RuntimeError("Source changed during extraction")
    del other, restored
    partial.replace(output)
    record = {
        "status": "passed_exact_ema_extraction",
        "source_sha256": expected_sha256,
        "source_bytes": original.st_size,
        "output_sha256": digest(output),
        "output_bytes": output.stat().st_size,
        "ema_tensors": len(ema),
        "ema_tensor_bytes": sum(v.numel() * v.element_size() for v in ema.values()),
        "dtypes": sorted({str(v.dtype) for v in ema.values()}),
        "comparison": "all tensor names, shapes, dtypes and bytes equal",
        "omitted": [str(key) for key in state if key != "generator_ema"],
        "scope": "inference packaging only; original checkpoint retained separately",
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.source, arguments.output, arguments.sha256), indent=2))
