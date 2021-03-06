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
 * \section sos_part_delete sos_part_delete command
 *
 * \b NAME
 *
 * sos_part_delete - Delete a partition in a Container
 *
 * \b SYNOPSIS
 *
 * sos_part_delete -C <container> <name>
 *
 * \b DESCRIPTION
 *
 *  Delete a parition in a Container. The partition must be in the
 *  OFFLINE state to be deleted.
 *
 * \b -C=PATH
 *
 * Specify the PATH to the Container. This option is required.
 *
 * \b <name>
 *
 * Specify the name of the Partition. This paramter is required.
 *
 */
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <sos/sos.h>

const char *short_options = "C:s:";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos_part_delete -C <path> <name>\n");
	printf("    -C <path>   The path to the container.\n");
	printf("    <name>	The partition name.\n");
	exit(1);
}

int main(int argc, char **argv)
{
	char *cont_path = NULL;
	char *name = NULL;
	sos_part_t part;
	int o, rc;
	sos_t sos;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'C':
			cont_path = strdup(optarg);
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}
	if (!cont_path)
		usage(argc, argv);

	if (optind == argc)
		usage(argc, argv);
	name = strdup(argv[optind]);
	sos = sos_container_open(cont_path, SOS_PERM_RW);
	if (!sos) {
		printf("Error %d opening the container %s.\n",
		       errno, cont_path);
		exit(1);
	}
	part = sos_part_find(sos, name);
	if (!part) {
		printf("The specified partition does not exist.\n");
		exit(1);
	}
	rc = sos_part_delete(part);
	if (rc)
		perror("sos_part_delete");
	return 0;
}
