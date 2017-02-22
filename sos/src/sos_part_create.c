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
 *
 * \page sos_part_create sos_part_create command
 *
 * \b NAME
 *
 * sos_part_create - Create a partition
 *
 * \b SYNOPSIS
 *
 * sos_part_create -C <container> [-s <state>] [-p <path>] part_name
 *
 * \b DESCRIPTION
 *
 * Create a new partition.
 *
 * \b -C=PATH
 *
 * Specify the PATH to the Container. This option is required.
 *
 * \b -s=STATE
 *
 * Specify the state of the new partition. This paramter is optional.
 * The default initial state is OFFLINE.
 *
 * \b -p=PATH
 *
 * Specify the path to the parition. This parameter is optional. The
 * default path is the container path.
 *
 * \b part_name
 *
 * Specifies the name of the partition.
 *
 */
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <errno.h>
#include <sos/sos.h>

const char *short_options = "C:p:s:";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{"state",       no_argument,        0,  's'},
	{"path",        required_argument,  0,  'p'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos_part_create -C <path> [-p <path>] [-s <state>] <name>\n");
	printf("    -C <path>   The path to the container. Required for all options.\n");
	printf("    -p <path>	Optional partition path. The container path is used by default.\n");
	printf("    -s <state>  The initial state of a partition. Valid states are:\n"
	       "                primary  - All new allocations go in this partition.\n"
	       "                online   - Objects are accessible, the partition does not grow\n"
	       "                offline  - Object references are invalid; the partition\n"
	       "                           may be moved or deleted.\n"
	       "    <name>      The partition name.\n"
	       "     The default initial state is OFFLINE.\n");
	exit(1);
}

int create_part(sos_t sos, const char *name, const char *path)
{
	int rc = sos_part_create(sos, name, path);
	if (rc)
		perror("sos_part_create: ");
	return rc;
}

void modify_part(sos_t sos, const char *name, const char *state)
{
	sos_part_t part;
	sos_part_iter_t iter;
	int rc = ENOENT;

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
		if (strcmp(sos_part_name(part), name)) {
			sos_part_put(part);
			continue;
		}
		rc = 0;
		if (0 == strcasecmp(state, "active")) {
			rc = sos_part_state_set(part, SOS_PART_STATE_ACTIVE);
			if (rc) {
				errno = rc;
				perror("Error enabling the partition");
			}
		} else if (0 == strcasecmp(state, "offline")) {
			rc = sos_part_state_set(part, SOS_PART_STATE_OFFLINE);
			if (rc) {
				errno = rc;
				perror("Error disabling the partition");
			}
		} else if (0 == strcasecmp(state, "primary")) {
			sos_part_state_set(part, SOS_PART_STATE_PRIMARY);
		}
		else
			printf("The state string specified is not recognized.\n");
		sos_part_put(part);
		break;
	}
	sos_part_iter_free(iter);
	if (rc == ENOENT)
		printf("The partition was not found, did you forget to create it?\n");
}

int main(int argc, char **argv)
{
	char *cont_path = NULL;
	char *part_path = NULL;
	char *name = NULL;
	char *state = NULL;
	int o, rc;
	sos_t sos;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'C':
			cont_path = strdup(optarg);
			break;
		case 's':
			state = strdup(optarg);
			break;
		case 'p':
			part_path = strdup(optarg);
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}
	if (!cont_path || argc == optind)
		usage(argc, argv);
	name = strdup(argv[optind]);

	sos = sos_container_open(cont_path, SOS_PERM_RW);
	if (!sos) {
		printf("Error %d opening the container %s.\n",
		       errno, cont_path);
		exit(1);
	}

	rc = create_part(sos, name, part_path);
	if (rc)
		exit(1);
	if (state) {
		modify_part(sos, name, state);
		free(state);
	}
	free(cont_path);
	free(name);
	return 0;
}
