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
 * \page sos_cont_clone Create a thin copy of a container
 *
 * This command creates a new container with all of the schema from
 * the source container, but none of the partition data.
 *
 * sos_cont_clone -C <container> <clone-container-path>
 *
 * @param "-C PATH" The *PATH* to the source Container. This parameter is required.
 * @param  The *PATH* to the cloned container. This parameter is required.
 */
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <sos/sos.h>

const char *short_options = "C:c:";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{"clone",	required_argument,  0,  'c'},

	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos_cont_clone -C <path> -c <path>\n");
	printf("    -C <path>   The path to the container to clone.\n");
	printf("    -c <path>	The path to the new clone; that must not exist.\n");
	exit(1);
}

int main(int argc, char **argv)
{
	char *cont_path = NULL;
	char *clone_path = NULL;
	int o, rc;
	sos_t sos;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'C':
			cont_path = strdup(optarg);
			break;
		case 'c':
			clone_path = strdup(optarg);
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}
	if (!cont_path || !clone_path)
		usage(argc, argv);

	sos = sos_container_open(cont_path, SOS_PERM_RD);
	if (!sos) {
		printf("The container at %s could not be opened, error %m\n", cont_path);
		exit(1);
	}
	rc = sos_container_clone(sos, clone_path);
	if (rc) {
		printf("The clone at %s could not be created, error %s\n",
		       clone_path, strerror(rc));
		exit(1);
	}
	return 0;
}
