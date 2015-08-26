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
 * \page partition_overview Partitions
 * \section part_sect Partitions
 *
 * In order to faciliate management of the storage consumed by a
 * Container, a Container is divided up into one or more Partitions. A
 * Parition contains the objects that are created in the
 * Container. A Partition is in one of the following states:
 *
 * - \b Primary  The Partition is the target of all new
 *               object allocations and it's contents may
 *               be referred to by entries in one or more
 *               Indices.
 * - \b Active   The contents of the Partition are
 *               accessible and it's objects may be
 *               referred to by one or more Indices.
 * - \b Offline  The contents of the Partition are in the
 *               process of being migrated to backup or
 *               secondary storage. The objects that are
 *               part of the Partition are or have been
 *               removed from all Indices.
 *
 * A Container always has at least one partition called the "Default"
 * Partition. When a Container is newly created, this Partition is
 * automatically created as well.
 *
 * Typically an administrator will use the \ref sos_part to manage
 * partitions. This command can be used to create, modify and delete
 * Partitions. Suppose, for example, the administrator wishes to keep
 * 5 days worth of live data. One approach is to create partitions
 * named for the date. This date might look like the following
 * "2015-03-03", "2015-03-04", etc... The following
 * command creates a new partition with the name 2015-03-03:
 *
 *       sos_part -C MyContainer -c -n "2015-03-03"
 *
 * Then to activate and make the Parition the target of new
 * allocations, it can be made Primary as follows:
 *
 *       sos_part -C MyContainer -m primary -n "2015-03-03"
 *
 * Note that these two steps can be combined:
 *
 *       sos_part -C MyContainer -c -n "2015-03-03" -m primary
 *
 * Creates the Partition and makes it Primary in a single step.
 *
 * When a Partition's data is no longer needed, it may be moved to
 * secondary storage as follows:
 *
 *      sos_part -C MyContainer -m offline -n "2015-03-03"
 *      sos_part -C MyContainer -d -n "2015-03-03"
 *      mv ${SOS_ROOT}/MyContainer/2015-03-03 /secondary-storage/MyContainer/2015-03-03
 *
 * The list of Partitions defined in a Container can be queried as
 * follows:
 *
 *      tom@css:/SOS/import$ sos_part -C /NVME/0/SOS_ROOT/Test
 *      Partition Name       RefCount Status           Size     Modified         Accessed
 *      -------------------- -------- ---------------- -------- ---------------- ----------------
 *      00000000                    3 ONLINE                 1M 2015/08/25 13:49 2015/08/25 13:51
 *      00000001                    3 ONLINE                 2M 2015/08/25 11:54 2015/08/25 13:51
 *      00000002                    3 ONLINE                 2M 2015/08/25 11:39 2015/08/25 13:51
 *      00000003                    3 ONLINE PRIMARY         2M 2015/08/25 11:39 2015/08/25 13:51
 *
 *
 * \section sos_part sos_part command
 *
 * \b NAME
 *
 * sos_part - Manage a Container's Partitions
 *
 * \b SYNOPSIS
 *
 * sos_cmd -C <container> [OPTION]...
 *
 * \b DESCRIPTION
 *
 * Create, list, modify and destroy a Container's Partitions.
 *
 * \b -C=PATH
 *
 * Specify the PATH to the Container. This option is required.
 *
 * \b -n=NAME
 *
 * Specify the name of the Partition. This paramter is required for options c, d, and m.
 *
 * \b -c
 *
 * Create the partition and set it's initial state to 'offline'.
 *
 * \b -q
 *
 * Query the Container's Partitions and their state. This is the default option.
 *
 * \b -d
 *
 * Delete the named partition. The partition must be in the \c offline state.
 *
 * \b -m=STATE
 *
 * Modify the state of a partition. Valid values for the STATE parameter
 * are: "primary", "active", and "offline".
 *
 * If the "primary" STATE is requested, the current primary Partition
 * is made "active" and the specified partition is made primary.
 *
 * If the "active" STATE is requested and the named Partition is
 * Primary, an error is returned indicating the Partition is busy.
 *
 * If the "offline" STATE is requested, and the Partition is Primary,
 * an error is returned indicating the Partition is busy. Otherwise,
 * all keys referring to an Object in the named Partition are removed
 * from all Indices in the Container and the Paritition is moved to
 * the "offline" state.
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
		if (strcmp(sos_part_name(part), name)) {
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
