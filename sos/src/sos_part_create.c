/*
 * Copyright (c) 2015 Open Grid Computing, Inc. All rights reserved.
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
 * \page sos_part_create Create a Partition
 *
 * sos_part_create -C <container> [-s <state>] [-p <path>] part_name
 *
 * @param "-C PATH" The *PATH* to the Container. This option is required.
 * @param "-s STATE" The *STATE* of the new partition. This paramter is optional.
 * The default initial state is *offline*. The STATE is one of *primary*, *active*,
 * or *offline*.
 * @param "-p PATH" The *PATH* to the parition. This parameter is optional. The
 * default path is the container path.
 * @param part_name The name of the partition.
 */
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <sos/sos.h>

const char *short_options = "p:d:o:s:";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"path",        required_argument,  0,  'p'},
	{"desc",        required_argument,  0,  'd'},
	{"store",	required_argument,  0,  's'},
	{"mode",	required_argument,  0,  'o'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos_part_create -p <path> -d <desc>\n");
	printf("    -p <path>	Optional partition path. The container path is used by default.\n");
	printf("    -d <desc>   A description of the partition's contents.\n");
	printf("    -s <name>   Storage strategy: MMOS or LSOS");
	printf("    -o <mode>   File creation mode bits in octal. See open(2)\n");
	exit(1);
}

int main(int argc, char **argv)
{
	char *path = NULL;
	char *desc = "";
	sos_perm_t o_perm = SOS_BE_MMOS;
	int o_mode = 0660;
	int o, rc = 0;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'p':
			path = strdup(optarg);
			break;
		case 'd':
			desc = strdup(optarg);
			break;
		case 'o':
			o_mode = strtol(optarg, NULL, 8);
			break;
		case 's':
			if (0 == strcasecmp(optarg, "mmos")) {
				o_perm = SOS_BE_MMOS;
			} else if (0 == strcasecmp(optarg, "lsos")) {
				o_perm = SOS_BE_LSOS;
			} else {
				fprintf(stderr, "Invalid storage policy '%s'\n", optarg);
				usage(argc, argv);
			}
		case '?':
		default:
			usage(argc, argv);
		}
	}
	if (!path) {
		printf("The partition path is a required argument.\n");
		usage(0, NULL);
	}
	sos_part_t part = sos_part_open(path, o_perm | SOS_PERM_CREAT, o_mode, desc);
	if (!part) {
		fprintf(stderr, "sos_part_create: The partition could not be created, error %s\n", strerror(errno));
		rc = errno;
	} else {
		sos_part_put(part);
	}
	return rc;
}
