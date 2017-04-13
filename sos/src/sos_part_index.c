/*
 * Copyright (c) 2017 Open Grid Computing, Inc. All rights reserved.
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
 * \section sos_part_index sos_part_index command
 *
 * \b NAME
 *
 * sos_part_index - Add objects in a partition to their schema indices
 *
 * \b SYNOPSIS
 *
 * sos_part_index -C <SRC-PATH> <PART-NAME>
 *
 * \b DESCRIPTION
 *
 *
 * Objects that have been exported to a partition may not be indexed
 * by the export process. This command can be used to ensure that all
 * objects in a partition have been added to their schema indices.
 *
 * The partition may not be in the OFFLINE state.
 *
 * See the \c sos_part_export command's -I option.
 *
 * \b -C SRC-PATH
 *
 * Specify the PATH to the source Container. This option is required.
 *
 * \b PART-NAME
 *
 * The name of the partition.
 */
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
	printf("sos_part_index -C <path> > <part-name>\n");
	printf("    -C <src-path> The path to the container.\n");
	printf("    <part-name>   The name of the partition to index.\n");
	exit(1);
}

const char *short_options = "C:E:";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"path",        required_argument,  0,  'C'},
	{0,             0,                  0,  0}
};

int main(int argc, char **argv)
{
	sos_t src_sos;
	int opt;
	char *part_name = NULL;
	char *src_path = NULL;
	while (0 < (opt = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (opt) {
		case 'C':
			src_path = strdup(optarg);
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}

	if (!src_path)
		usage(argc, argv);

	if (optind < argc)
		part_name = strdup(argv[optind]);
	else
		usage(argc, argv);

	sos_log_file_set(stderr);
	sos_log_mask_set(SOS_LOG_WARN);

	src_sos = sos_container_open(src_path, SOS_PERM_RW);
	if (!src_sos) {
		printf("Error %d opening the container %s.\n",
		       errno, src_path);
		exit(1);
	}
	sos_part_t part = sos_part_find(src_sos, part_name);
	if (!part) {
		printf("The partition named '%s' was not found.\n", part_name);
		exit(1);
	}
	if (sos_part_state(part) == SOS_PART_STATE_OFFLINE) {
		printf("You cannot export objects from an OFFLINE partition.\n");
		sos_part_put(part);
		exit(2);
	}
	int64_t count = sos_part_index(part);
	sos_part_put(part);
	if (count < 0)
		printf("Error %d encountered indexing objects.\n", -count);
	else
		printf("%d objects were indexed.\n", count);
	return 0;
}
