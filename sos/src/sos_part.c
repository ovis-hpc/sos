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

#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <sos/sos.h>

const char *short_options = "C:n:m:cdq";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{"name",        required_argument,  0,  'n'},
	{"create",	no_argument,	    0,  'c'},
	{"modify",      no_argument,        0,  'm'},
	{"delete",      no_argument,        0,  'd'},
	{"query",	no_argument,        0,  'q'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos_part\n");
	printf("    -C <path>   The path to the container. Required for all options.\n");
	printf("    -n <name>	The partition name. Required for all options except query.\n");
	printf("    -c          Create the partition. It's initial state is 'offline'.\n");
	printf("    -d          Delete the named partition. It must be in the inactive state.\n");
	printf("    -m <state>  Modify the state of a partition. Valid states are:\n"
	       "                    primary  - All new allocations go in this partition.\n"
	       "                    online   - Objects are accessible, the partition does not grow\n"
	       "                    offline  - Object references are invalid; the partition\n"
	       "                               may be moved or deleted.\n");
	printf("    -q          Query the container's partitions and their state.\n");
	exit(1);
}

void query(sos_t sos)
{
	sos_container_part_list(sos, stdout);
}

void create_part(sos_t sos, const char *name)
{
	int rc = sos_part_create(sos, name);
	if (rc)
		perror("sos_part_create: ");
}

void modify_part(sos_t sos, const char *name, const char *state)
{
	sos_part_t part;
	sos_part_iter_t iter;
	int rc;

	if (!state) {
		printf("A state string must be specified.\n");
		return;
	}

	iter = sos_part_iter_new(sos);
	if (!iter) {
		perror("sos_part_iter_new: ");
		return;
	}
	for (part = sos_part_first(iter); part; part = sos_part_next(iter)) {
		if (strcmp(sos_part_name_get(part), name)) {
			sos_part_put(part);
			continue;
		}

		if (0 == strcasecmp(state, "online")) {
			rc = sos_part_active_set(part, 1);
			if (rc) {
				errno = rc;
				perror("Error enabling the partition");
			}
		} else if (0 == strcasecmp(state, "offline")) {
			rc = sos_part_active_set(part, 0);
			if (rc) {
				errno = rc;
				perror("Error disabling the partition");
			}
		} else if (0 == strcasecmp(state, "primary")) {
			sos_part_primary_set(part);
		}
		else
			printf("The state string specified is not recognized.\n");
		sos_part_put(part);
		break;
	}
	sos_part_iter_free(iter);
}

#define CREATE	1
#define QUERY	2
#define MODIFY	4
#define DELETE	8

int main(int argc, char **argv)
{
	char *path = NULL;
	char *name = NULL;
	char *state = NULL;
	int o;
	int mode;
	int action = 0;
	sos_t sos;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'C':
			path = strdup(optarg);
			break;
		case 'n':
			name = strdup(optarg);
			break;
		case 'c':
			action |= CREATE;
			break;
		case 'm':
			action |= MODIFY;
			state = strdup(optarg);
			break;
		case 'q':
			action |= QUERY;
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}
	if (!path)
		usage(argc, argv);

	if (!action)
		action = QUERY;

	if (action == QUERY) {
		mode = O_RDONLY;
	} else {
		if (!name)
			usage(argc, argv);
		mode = O_RDWR;
	}
	sos = sos_container_open(path, mode);
	if (!sos) {
		printf("Error %d opening the container %s.\n",
		       errno, path);
		exit(1);
	}

	if (action & CREATE)
		create_part(sos, name);

	if (action & MODIFY)
		modify_part(sos, name, state);
	if (action & QUERY)
		query(sos);

	return 0;
}
