"""Host runtime for SAVA, implemented on the simulator's C API via ctypes.

This plays the role a vendor runtime/driver plays for real hardware: it owns
device memory allocation, uploads inputs, launches executables and reads back
results. A later iteration replaces it with an IREE HAL driver that calls the
same C API (see docs/roadmap.md).
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import sbin

_REPO = Path(__file__).resolve().parents[2]

_NP_DTYPES = {"f32": np.float32, "bf16": np.uint16, "i32": np.int32, "i8": np.int8}
_STATUS = {
    0: "OK", 1: "INVALID_ARGUMENT", 2: "OUT_OF_RANGE", 3: "BAD_EXECUTABLE",
    4: "EXECUTION", 5: "HAZARD", 6: "TIMEOUT",
}


class SimError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"[{_STATUS.get(status, status)}] {message}")
        self.status = _STATUS.get(status, str(status))


class _Config(ctypes.Structure):
    _fields_ = [
        ("dram_bytes", ctypes.c_uint64),
        ("sram_bytes", ctypes.c_uint32),
        ("strict_hazards", ctypes.c_int),
        ("max_instructions", ctypes.c_uint64),
        ("trace_path", ctypes.c_char_p),
    ]


_STAT_FIELDS = [
    "cycles", "instructions", "scalar_instructions", "wait_stall_cycles",
    "queue_stall_cycles", "dma_ops", "dma_busy_cycles", "dma_bytes_in", "dma_bytes_out",
    "mxu_ops", "mxu_busy_cycles", "mxu_macs", "vpu_ops", "vpu_busy_cycles", "vpu_elements",
]


class _Stats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in _STAT_FIELDS]


def find_library() -> Path:
    env = os.environ.get("SAVA_SIM_LIB")
    candidates = [Path(env)] if env else []
    candidates += [_REPO / "build" / "sim" / "libsava_sim.so",
                   _REPO / "build-debug" / "sim" / "libsava_sim.so"]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "libsava_sim.so not found; build the simulator (cmake --preset dev && "
        "cmake --build --preset dev) or set SAVA_SIM_LIB")


_lib = None


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(find_library()))
        lib.sava_config_init_default.argtypes = [ctypes.POINTER(_Config)]
        lib.sava_device_create.argtypes = [ctypes.POINTER(_Config)]
        lib.sava_device_create.restype = ctypes.c_void_p
        lib.sava_device_destroy.argtypes = [ctypes.c_void_p]
        lib.sava_device_last_error.argtypes = [ctypes.c_void_p]
        lib.sava_device_last_error.restype = ctypes.c_char_p
        for fn in (lib.sava_dram_write, lib.sava_dram_read):
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
        lib.sava_load_executable.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64]
        lib.sava_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        lib.sava_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Stats)]
        _lib = lib
    return _lib


@dataclass
class Stats:
    cycles: int
    instructions: int
    scalar_instructions: int
    wait_stall_cycles: int
    queue_stall_cycles: int
    dma_ops: int
    dma_busy_cycles: int
    dma_bytes_in: int
    dma_bytes_out: int
    mxu_ops: int
    mxu_busy_cycles: int
    mxu_macs: int
    vpu_ops: int
    vpu_busy_cycles: int
    vpu_elements: int

    def utilization(self, unit: str) -> float:
        return getattr(self, f"{unit}_busy_cycles") / self.cycles if self.cycles else 0.0

    def report(self, clock_ghz: float = 1.0) -> str:
        us = self.cycles / (clock_ghz * 1e3)
        gflops = 2 * self.mxu_macs / (self.cycles / clock_ghz) if self.cycles else 0.0
        return "\n".join([
            f"cycles        {self.cycles:>12}  ({us:.2f} us @ {clock_ghz} GHz)",
            f"instructions  {self.instructions:>12}  (scalar {self.scalar_instructions})",
            f"wait stalls   {self.wait_stall_cycles:>12}",
            f"dma  ops {self.dma_ops:>6}  util {100 * self.utilization('dma'):5.1f}%  "
            f"in {self.dma_bytes_in} B  out {self.dma_bytes_out} B",
            f"mxu  ops {self.mxu_ops:>6}  util {100 * self.utilization('mxu'):5.1f}%  "
            f"macs {self.mxu_macs}  ({gflops:.1f} GFLOP/s)",
            f"vpu  ops {self.vpu_ops:>6}  util {100 * self.utilization('vpu'):5.1f}%",
        ])


class Device:
    """One simulated SAVA device (DRAM + one core)."""

    def __init__(self, *, dram_bytes: int | None = None, strict_hazards: bool = True,
                 max_instructions: int = 0, trace: str | None = None):
        self._dev = None
        lib = _load()
        cfg = _Config()
        lib.sava_config_init_default(ctypes.byref(cfg))
        if dram_bytes is not None:
            cfg.dram_bytes = dram_bytes
        cfg.strict_hazards = int(strict_hazards)
        cfg.max_instructions = max_instructions
        self._trace = trace.encode() if trace else None
        cfg.trace_path = self._trace
        self._lib = lib
        self._dev = lib.sava_device_create(ctypes.byref(cfg))
        if not self._dev:
            raise MemoryError("cannot create SAVA device")
        self.dram_bytes = cfg.dram_bytes
        self._next = 0x1000  # bump allocator; address 0 is never handed out

    def close(self):
        if self._dev:
            self._lib.sava_device_destroy(self._dev)
            self._dev = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def _check(self, status: int):
        if status != 0:
            raise SimError(status, self._lib.sava_device_last_error(self._dev).decode())

    def alloc(self, nbytes: int, align: int = 4096) -> int:
        addr = (self._next + align - 1) & ~(align - 1)
        if addr + nbytes > self.dram_bytes:
            raise MemoryError("out of device DRAM")
        self._next = addr + nbytes
        return addr

    def write(self, addr: int, array: np.ndarray):
        data = np.ascontiguousarray(array)
        self._check(self._lib.sava_dram_write(self._dev, addr, data.ctypes.data, data.nbytes))

    def read(self, addr: int, dtype, shape) -> np.ndarray:
        out = np.empty(shape, dtype=dtype)
        self._check(self._lib.sava_dram_read(self._dev, addr, out.ctypes.data, out.nbytes))
        return out

    def load(self, image: bytes):
        self._check(self._lib.sava_load_executable(self._dev, image, len(image)))

    def run(self, args: Sequence[int]):
        arr = (ctypes.c_uint64 * len(args))(*args)
        self._check(self._lib.sava_run(self._dev, arr, len(args)))

    def stats(self) -> Stats:
        s = _Stats()
        self._check(self._lib.sava_get_stats(self._dev, ctypes.byref(s)))
        return Stats(**{name: getattr(s, name) for name in _STAT_FIELDS})


def run_executable(image: bytes, inputs: Sequence[np.ndarray], *, device: Device | None = None,
                   **device_kwargs) -> tuple[list[np.ndarray], Stats]:
    """Runs an .sbin image. `inputs` bind to the in/inout arguments in order.

    Returns the out/inout arguments (in order) and the performance counters.
    """
    exe = sbin.read(image)
    own = device is None
    dev = device or Device(**device_kwargs)
    try:
        dev.load(image)
        addrs, inputs = [], list(inputs)
        for a in exe.args:
            addr = dev.alloc(a.nbytes)
            addrs.append(addr)
            if a.kind != "out":
                if not inputs:
                    raise ValueError(f"missing input for argument '{a.name}'")
                x = np.asarray(inputs.pop(0))
                want = _NP_DTYPES[a.dtype]
                if x.dtype != want or tuple(x.shape) != a.shape:
                    raise ValueError(f"argument '{a.name}' expects {a.dtype}{list(a.shape)} "
                                     f"(numpy {np.dtype(want).name}), got {x.dtype}{list(x.shape)}")
                dev.write(addr, x)
        if inputs:
            raise ValueError("too many inputs")
        dev.run(addrs)
        outputs = [dev.read(addr, _NP_DTYPES[a.dtype], a.shape)
                   for a, addr in zip(exe.args, addrs) if a.kind != "in"]
        return outputs, dev.stats()
    finally:
        if own:
            dev.close()


def to_bf16(x: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bit patterns (uint16), round to nearest even."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = u + 0x7FFF + ((u >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def from_bf16(x: np.ndarray) -> np.ndarray:
    """bf16 bit patterns (uint16) -> f32."""
    return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
