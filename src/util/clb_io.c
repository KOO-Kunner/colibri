/* SPDX-License-Identifier: MIT */
#define _POSIX_C_SOURCE 200809L

#include "colibri/clb_alloc.h"
#include "colibri/clb_error.h"
#include "colibri/clb_log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

/* ---- mmap 句柄（内部使用） ---- */
typedef struct clb_mmap_handle {
    void   *addr;
    size_t  size;
#if defined(_WIN32)
    HANDLE  hFile;
    HANDLE  hMap;
#else
    int     fd;
#endif
} clb_mmap_handle;

/* --------------------------------------------------------------
 * 读取整个文件到堆内存
 * -------------------------------------------------------------- */
clb_status clb__io_read_all(const char *path, uint8_t **out, size_t *out_size)
{
    if (!path || !out || !out_size) return CLB_ERR_INVALID_ARG;

    FILE *f = fopen(path, "rb");
    if (!f) {
        CLB_LOGE("cannot open file: %s", path);
        return CLB_ERR_IO;
    }

    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return CLB_ERR_IO; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return CLB_ERR_IO; }
    rewind(f);

    uint8_t *buf = (uint8_t *)clb_malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return CLB_ERR_OUT_OF_MEMORY; }

    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) {
        clb_free(buf);
        return CLB_ERR_IO;
    }

    buf[sz] = 0;
    *out      = buf;
    *out_size = (size_t)sz;
    return CLB_OK;
}

/* --------------------------------------------------------------
 * mmap 打开文件（只读）
 * -------------------------------------------------------------- */
void *clb__io_mmap_open(const char *path, size_t *size, clb_status *st,
                        void **handle_out)
{
    if (!path || !size || !handle_out) {
        if (st) *st = CLB_ERR_INVALID_ARG;
        return NULL;
    }

    clb_mmap_handle *h = (clb_mmap_handle *)clb_calloc(1, sizeof(*h));
    if (!h) { if (st) *st = CLB_ERR_OUT_OF_MEMORY; return NULL; }

#if defined(_WIN32)
    h->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (h->hFile == INVALID_HANDLE_VALUE) {
        CLB_LOGE("CreateFile failed: %s", path);
        clb_free(h);
        if (st) *st = CLB_ERR_IO;
        return NULL;
    }

    LARGE_INTEGER li;
    if (!GetFileSizeEx(h->hFile, &li)) {
        CloseHandle(h->hFile);
        clb_free(h);
        if (st) *st = CLB_ERR_IO;
        return NULL;
    }
    h->size = (size_t)li.QuadPart;

    h->hMap = CreateFileMappingA(h->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!h->hMap) {
        CloseHandle(h->hFile);
        clb_free(h);
        if (st) *st = CLB_ERR_IO;
        return NULL;
    }

    h->addr = MapViewOfFile(h->hMap, FILE_MAP_READ, 0, 0, 0);
    if (!h->addr) {
        CloseHandle(h->hMap);
        CloseHandle(h->hFile);
        clb_free(h);
        if (st) *st = CLB_ERR_IO;
        return NULL;
    }
#else
    h->fd = open(path, O_RDONLY);
    if (h->fd < 0) {
        CLB_LOGE("open failed: %s", path);
        clb_free(h);
        if (st) *st = CLB_ERR_IO;
        return NULL;
    }

    struct stat sb;
    if (fstat(h->fd, &sb) != 0) {
        close(h->fd);
        clb_free(h);
        if (st) *st = CLB_ERR_IO;
        return NULL;
    }
    h->size = (size_t)sb.st_size;

    h->addr = mmap(NULL, h->size, PROT_READ, MAP_PRIVATE, h->fd, 0);
    if (h->addr == MAP_FAILED) {
        close(h->fd);
        clb_free(h);
        if (st) *st = CLB_ERR_IO;
        return NULL;
    }

#  if defined(MADV_SEQUENTIAL)
    madvise(h->addr, h->size, MADV_SEQUENTIAL);
#  endif
#endif

    *size      = h->size;
    *handle_out = h;
    if (st) *st = CLB_OK;
    return h->addr;
}

/* --------------------------------------------------------------
 * 关闭 mmap
 * -------------------------------------------------------------- */
void clb__io_mmap_close(void *handle)
{
    clb_mmap_handle *h = (clb_mmap_handle *)handle;
    if (!h) return;

#if defined(_WIN32)
    if (h->addr) UnmapViewOfFile(h->addr);
    if (h->hMap) CloseHandle(h->hMap);
    if (h->hFile != INVALID_HANDLE_VALUE) CloseHandle(h->hFile);
#else
    if (h->addr && h->addr != MAP_FAILED) munmap(h->addr, h->size);
    if (h->fd >= 0) close(h->fd);
#endif
    clb_free(h);
}