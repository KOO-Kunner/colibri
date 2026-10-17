# SPDX-License-Identifier: MIT
"""Colibri —— 纯 C 轻量大模型推理引擎的 Python 绑定"""
import ctypes
from typing import Optional, List

from ._ffi import (
    _lib, EngineParams, GenParams, SamplerParams, STREAM_FN,
)


class ColibriError(RuntimeError):
    def __init__(self, status: int):
        self.status = status
        msg = _lib.clb_status_str(status).decode()
        super().__init__(f"colibri error {status}: {msg}")


class Sampler:
    GREEDY = 0
    TEMP   = 1

    def __init__(self):
        p = _lib.clb_sampler_default_params()
        self.mode           = p.mode
        self.seed           = p.seed
        self.temp           = p.temp
        self.top_k          = p.top_k
        self.top_p          = p.top_p
        self.min_p          = p.min_p
        self.repeat_penalty = p.repeat_penalty
        self.repeat_last_n  = p.repeat_last_n
        self.n_vocab        = p.n_vocab

    def _to_c(self) -> SamplerParams:
        return SamplerParams(
            self.mode, self.seed, self.temp, self.top_k, self.top_p,
            self.min_p, self.repeat_penalty, self.repeat_last_n, self.n_vocab,
        )


class Colibri:
    def __init__(self, model_path: str, n_ctx: int = 0, n_threads: int = 0,
                 use_mmap: bool = True, log_level: int = 3):
        ep = EngineParams()
        ep.model_path = str(model_path).encode()
        ep.n_ctx      = n_ctx
        ep.n_threads  = n_threads
        ep.use_mmap   = use_mmap
        ep.log_level  = log_level

        h = ctypes.c_void_p()
        st = _lib.clb_engine_create(ctypes.byref(ep), ctypes.byref(h))
        if st != 0:
            raise ColibriError(st)
        self._handle = h

    def close(self):
        if self._handle is not None:
            _lib.clb_engine_free(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def generate(self, prompt: str, n_predict: int = 128,
                 sampler: Optional[Sampler] = None,
                 echo_prompt: bool = False) -> str:
        chunks: List[bytes] = []

        def _cb(piece_ptr, _tok, _user):
            if piece_ptr:
                chunks.append(piece_ptr)
            return True

        cb = STREAM_FN(_cb)

        gp = GenParams()
        gp.prompt      = prompt.encode()
        gp.n_predict   = n_predict
        gp.echo_prompt = echo_prompt
        gp.sampler     = (sampler or Sampler())._to_c()

        n_gen = ctypes.c_int32(0)
        st = _lib.clb_engine_generate(
            self._handle,
            ctypes.byref(gp),
            cb, None,
            ctypes.byref(n_gen),
        )
        if st != 0:
            raise ColibriError(st)
        return b"".join(chunks).decode("utf-8", errors="replace")

    @property
    def model_size(self) -> int:
        return _lib.clb_model_nbytes(_lib.clb_engine_model(self._handle))

    @property
    def n_ctx(self) -> int:
        return _lib.clb_context_n_ctx(_lib.clb_engine_context(self._handle))

    @property
    def n_past(self) -> int:
        return _lib.clb_context_n_past(_lib.clb_engine_context(self._handle))


def init():
    _lib.clb_init()


def shutdown():
    _lib.clb_shutdown()


def version() -> str:
    return _lib.clb_version_string().decode()


__all__ = ["Colibri", "Sampler", "ColibriError", "init", "shutdown", "version"]
