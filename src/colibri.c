/* SPDX-License-Identifier: MIT */
#include "colibri/colibri.h"
#include "colibri/clb_cpu.h"
#include "colibri/clb_log.h"

clb_status clb_init(void)
{
    clb_cpu_detect();
    CLB_LOGI("colibri %s initialized", CLB_VERSION_STRING);
    return CLB_OK;
}

void clb_shutdown(void) { }
const char *clb_version_string(void) { return CLB_VERSION_STRING; }