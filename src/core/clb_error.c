/* SPDX-License-Identifier: MIT */
#include "colibri/clb_error.h"

const char *clb_status_str(clb_status s)
{
    switch (s) {
    case CLB_OK:                return "success";
    case CLB_ERR_INVALID_ARG:   return "invalid argument";
    case CLB_ERR_OUT_OF_MEMORY: return "out of memory";
    case CLB_ERR_IO:            return "i/o error";
    case CLB_ERR_FORMAT:        return "invalid model format";
    case CLB_ERR_UNSUPPORTED:   return "unsupported operation";
    case CLB_ERR_NOT_FOUND:     return "not found";
    case CLB_ERR_CONTEXT_FULL:  return "context window exhausted";
    case CLB_ERR_TOKENIZE:      return "tokenization failed";
    case CLB_ERR_THREAD:        return "thread error";
    case CLB_ERR_STATE:         return "invalid state";
    case CLB_ERR_INTERNAL:      return "internal error";
    default:                    return "unknown error";
    }
}