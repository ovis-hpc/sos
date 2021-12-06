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
 * \page sos_part_modify Modify the State of a Partition
 *
 * sos_part_modify -C <container> -s <state> -u <uid> -g <gid> -o <mode> <name>
 *
 * Modify the state of a partition. Valid values for the \c -s parameter
 * are: "primary", "active", and "offline".
 *
 * If the "primary" state is requested, the current primary Partition
 * is made "active" and the specified partition is made primary.
 *
 * If the "active" state is requested and the named Partition is
 * "primary", an error is returned indicating the Partition is busy.
 *
 * If the "offline" state is requested, and the Partition is "primary",
 * an error is returned indicating the Partition is busy. Otherwise,
 * all keys referring to an Object in the named Partition are removed
 * from all Indices in the Container and the Paritition is moved to
 * the "offline" state.

 * @param "-C PATH" The *PATH* to the Container.
 * @param "-s STATE" The *STATE* of the new partition. If not specified
 *                  The state is *offline*. Valid values are:
 *                    - *primary*
 *                    - *active*
 *                    - *offline*
 * @param "-u UID" The user-id to assign to the container
 * @param "-g GID" The group-id to assign to the container
 * @param "-o MODE" The access mode bits (see open(3))
 *
 */
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include <errno.h>
#include <sos/sos.h>

const char *short_options = "C:s:u:g:o:";

struct option long_options[] = {
	{"help",        no_argument,        0,  '?'},
	{"container",   required_argument,  0,  'C'},
	{"state",       required_argument,  0,  's'},
	{"uid",         required_argument,  0,  'u'},
	{"gid",         required_argument,  0,  'g'},
	{"mode",        required_argument,  0,  'o'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos_part_modify -C <path> -s <state> <name>\n"
	       "    -C <path>   The path to the container.\n"
	       "    -s <state>  Modify the state of a partition. Valid states are:\n"
	       "                primary  - All new allocations go in this partition.\n"
	       "                active   - Objects are accessible, the partition does not grow\n"
	       "                offline  - Object references are invalid; the partition\n"
	       "                           may be moved or deleted.\n"
	       "    -u <uid>    The user-id\n"
	       "    -g <gid>    The group-id\n"
	       "    -o <mode>   The acces mode mask (see open(3))\n"
	       "    <name>	The partition name.\n");
	exit(1);
}

void modify_part(sos_t sos, const char *name, const char *state, uid_t uid, gid_t gid, int mode)
{
	sos_part_t part;
	int rc = ENOENT;

	part = sos_part_by_name(sos, name);
	if (!part) {
		printf("The partition '%s' was not found.", name);
		return;
	}
	rc = 0;

	if (!state)
		goto skip_state;

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
	} else {
		printf("The state string '%s is invalid.\n", state);
	}
skip_state:
	sos_part_chown(part, uid, gid);
	if (mode)
		sos_part_chmod(part, mode);
	sos_part_put(part);
}

int main(int argc, char **argv)
{
	char *cont_path = NULL;
	char *name = NULL;
	char *state = NULL;
	int o;
	sos_t sos;
	uid_t uid = -1;
	gid_t gid = -1;
	int mode = 0;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'C':
			cont_path = strdup(optarg);
			break;
		case 's':
			state = strdup(optarg);
			break;
		case 'u':
			uid = strtol(optarg, NULL, 0);
			break;
		case 'g':
			gid = strtol(optarg, NULL, 0);
			break;
		case 'o':
			mode = strtol(optarg, NULL, 0);
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
	modify_part(sos, name, state, uid, gid, mode);
	return 0;
}
