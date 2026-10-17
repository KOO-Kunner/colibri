# SPDX-License-Identifier: MIT
"""ctypes 绑定到 libcolibri.so"""
import ctypes
from ctypes import (
    c_char_p, c_int, c_int32, c_size_t, c_uint64,
    c_float, c_bool, c_void_p, POINTER, Structure, CFUNCTYPE,
)
import os
from pathlib import Path


def _find_lib() -> str:
    env = os.environ.get("COLIBRI_LIB")
    if env and Path(env).exists():
        return env
    here = Path(__file__).resolve().parent
    root = here.parent.parent.parent
    candidates = [
        here / "libcolibri.so",
        here / "libcolibri.dylib",
        here / "colibri.dll",
        root / "build" / "libcolibri.so",
        root / "build" / "libcolibri.dylib",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RuntimeError(
        "libcolibri not found. Set COLIBRI_LIB env var, or run `make -j` in ~/colibri"
    )


_lib = ctypes.CDLL(_find_lib())


class SamplerParams(Structure):
    _fields_ = [
        ("mode",           c_int),
        ("seed",           c_uint64),
        ("temp",           c_float),
        ("top_k",          c_int32),
        ("top_p",          c_float),
        ("min_p",          c_float),
        ("repeat_penalty", c_float),
        ("repeat_last_n",  c_int32),
        ("n_vocab",        c_int32),
    ]


class EngineParams(Structure):
    _fields_ = [
        ("model_path", c_char_p),
        ("n_ctx",      c_int32),
        ("n_threads",  c_int),
        ("use_mmap",   c_bool),
        ("log_level",  c_int),
    ]


class GenParams(Structure):
    _fields_ = [
        ("prompt",      c_char_p),
        ("n_predict",   c_int32),
        ("sampler",     SamplerParams),
        ("echo_prompt", c_bool),
    ]


STREAM_FN = CFUNCTYPE(c_bool, c_char_p, c_int32, c_void_p)


_lib.clb_init.restype                   = c_int
_lib.clb_shutdown.restype               = None
_lib.clb_version_string.restype         = c_char_p

_lib.clb_engine_default_params.restype  = EngineParams
_lib.clb_gen_default_params.restype     = GenParams
_lib.clb_sampler_default_params.restype = SamplerParams

_lib.clb_engine_create.argtypes         = [POINTER(EngineParams),
                                           POINTER(c_void_p)]
_lib.clb_engine_create.restype          = c_int
_lib.clb_engine_free.argtypes           = [c_void_p]
_lib.clb_engine_generate.argtypes       = [c_void_p, POINTER(GenParams),
                                           STREAM_FN, c_void_p,
                                           POINTER(c_int32)]
_lib.clb_engine_generate.restype        = c_int

_lib.clb_engine_model.argtypes          = [c_void_p]
_lib.clb_engine_model.restype           = c_void_p
_lib.clb_engine_context.argtypes        = [c_void_p]
_lib.clb_engine_context.restype         = c_void_p

_lib.clb_status_str.argtypes            = [c_int]
_lib.clb_status_str.restype             = c_char_p

_lib.clb_log_set_level.argtypes         = [c_int]

_lib.clb_model_nbytes.argtypes          = [c_void_p]
_lib.clb_model_nbytes.restype           = c_size_t

_lib.clb_context_n_past.argtypes        = [c_void_p]
_lib.clb_context_n_past.restype         = c_int32
_lib.clb_context_n_ctx.argtypes         = [c_void_p]
_lib.clb_context_n_ctx.restype          = c_int32
