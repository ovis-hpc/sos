/* -*- c-basic-offset : 8 -*-
 * Copyright (c) 2022 Open Grid Computing, Inc. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the BSD-type
 * license below:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *      Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *      Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following
 *      disclaimer in the documentation and/or other materials provided
 *      with the distribution.
 *
 *      Neither the name of Open Grid Computing nor the names of any
 *      contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *      Modified source versions must be plainly marked as such, and
 *      must not be misrepresented as being the original software.
 *
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \page sos_part_reindex Rebuild all indices in the container partition
 *
 * sos_part_reindex -C <container> NAME
 *
 */
#define _GNU_SOURCE
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <sos/sos.h>

int verbose;

void usage(int argc, char *argv[])
{
	printf("sos_part_reindex -C <path> NAME\n");
	printf("    -C <path> The path to the container. Required for all options.\n");
	printf("    NAME      The name of the partition in the container.\n");
	exit(1);
}

const char *short_options = "C:";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{0,             0,                  0,  0}
};

struct reindex_cb_arg {
	FILE *fp;
	struct timespec start;
};

int __reindex_cb(sos_part_t part, void *arg, uint64_t obj_count)
{
	struct reindex_cb_arg *uarg = arg;
	struct timespec now;
	double dt;

	clock_gettime(CLOCK_REALTIME, &now);
	dt = (double)now.tv_sec * 1e9 + (double)now.tv_nsec;
	dt -= (double)uarg->start.tv_sec * 1e9 + (double)uarg->start.tv_nsec;
	dt /= 1e9;
       
	printf("%12ld %g objects/sec\n", obj_count, (double)obj_count / dt);
	return 0;
}

int main(int argc, char **argv)
{
	sos_t sos;
	int opt;
	char *part_name = NULL;
	char *path = NULL;
	while (0 < (opt = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (opt) {
		case 'C':
			path = strdup(optarg);
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}

	if (!path)
		usage(argc, argv);

	if (optind < argc) {
		part_name = strdup(argv[optind]);
	} else {
		printf("The partition name is required.\n\n");
		usage(argc, argv);
	}
	sos = sos_container_open(path, SOS_PERM_RW);
	if (!sos) {
		printf("Error %d opening the container %s.\n",
		       errno, argv[1]);
		exit(1);
	}
	sos_part_t part = sos_part_by_name(sos, part_name);
	if (!part) {
		printf("Error %d opening the partition %s\n", errno, part_name);
		exit(2);
	}
	struct reindex_cb_arg arg;
	arg.fp = stdout;
	clock_gettime(CLOCK_REALTIME, &arg.start);
	
	size_t count = sos_part_reindex(part, __reindex_cb, &arg, 10000);
	printf("%zu objects were reindexed.\n", count);

	sos_container_close(sos, SOS_COMMIT_ASYNC);
	return 0;
}
