/*
 * Copyright (c) 2021 Open Grid Computing, Inc. All rights reserved.
 *
 * See the file COPYING at the top of the source tree for license
 * detail.
 */
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <ods/ods.h>

FILE *__ods_log_fp;
uint64_t __ods_log_mask;

void ods_log_file_set(FILE *fp)
{
	__ods_log_fp = fp;
}

void ods_log_mask_set(uint64_t mask)
{
	__ods_log_mask = mask;
}
