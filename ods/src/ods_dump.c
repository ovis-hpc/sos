/*
 * Copyright (c) 2012 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2012 Sandia Corporation. All rights reserved.
 * Under the terms of Contract DE-AC04-94AL85000, there is a non-exclusive
 * license for use of this work by or on behalf of the U.S. Government.
 * Export of this program may require a license from the United States
 * Government.
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
 *      Neither the name of Sandia nor the names of any contributors may
 *      be used to endorse or promote products derived from this software
 *      without specific prior written permission.
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

/*
 * Author: Tom Tucker tom at ogc dot us
 */

#include <sys/fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <ods/ods.h>

void usage(int argc, char *argv[])
{
	printf("usage: %s [-a] <ODS Name>\n"
	       "    -a Show allocated objects.\n"
	       "\n"
	       "    <ODS Name> is the canonical name of the\n"
	       "    ODS absent the filename extension.\n",
	       argv[0]);
	exit(1);
}

int print_fn(ods_t ods, ods_ref_t ref, void *arg)
{
	ods_obj_t obj = ods_ref_as_obj(ods, ref);
	printf("%p %zu\n", obj->as.ptr, obj->size);
	ods_obj_put(obj);
	return 0;
}

int main(int argc, char *argv[])
{
	extern int optind;
	ods_t ods;
	int c, show_alloc = 0;
	char *name = NULL;
	struct ods_obj_iter_pos_s pos;
	while ((c = getopt(argc, argv, "a")) > 0) {
		switch (c) {
		case 'a':
			show_alloc = 1;
			break;
		default:
			usage(argc, argv);
		}
	}
	if (optind < argc)
		name = argv[optind];
	else
		usage(argc, argv);

	ods = ods_open(name, O_RDONLY);
	if (!ods) {
		printf("Could not open the ODS %s\n", name);
		exit(1);
	}
	if (show_alloc) {
		ods_obj_iter_pos_init(&pos);
		ods_obj_iter(ods, &pos, print_fn, NULL);
	} else {
		ods_dump(ods, stdout);
	}
	ods_close(ods, ODS_COMMIT_ASYNC);

	return 0;
}
