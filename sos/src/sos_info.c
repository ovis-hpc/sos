/*
 * Copyright (c) 2014 Open Grid Computing, Inc. All rights reserved.
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
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <fcntl.h>
#include <sos/sos.h>

const char *short_options = "C:";

struct option long_options[] = {
	{"container",   required_argument,  0,  'C'},
	{"help",        no_argument,        0,  '?'},
	{0,             0,                  0,  0}
};

void usage(int argc, char *argv[])
{
	printf("sos_info -C <container>\n");
	exit(1);
}

int main(int argc, char *argv[])
{
	sos_t sos;
	char *path;
	int o, rc;
	while (0 < (o = getopt_long(argc, argv, short_options, long_options, NULL))) {
		switch (o) {
		case 'C':
			path = optarg;
			break;
		case '?':
		default:
			usage(argc, argv);
		}
	}
	if (!path)
		usage(argc, argv);

	rc = sos_container_open(path, O_RDWR, &sos);
	if (rc) {
		printf("Error %d opening the container.\n", rc);
		usage(argc, argv);
	}
	sos_container_info(sos, stdout);
	sos_container_close(sos, ODS_COMMIT_ASYNC);
	return 0;
}
