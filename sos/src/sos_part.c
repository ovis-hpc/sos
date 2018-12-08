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
 * \page partitions Partitions
 * \section partitions Partitions
 *
 * In order to faciliate management of the storage consumed by a
 * Container, a Container is divided up into one or more Partitions. A
 * Parition contains the objects that are created in the
 * Container. A Partition is in one of the following states:
 *
 * - \b Primary  The Partition is the target of all new
 *               object allocations and its contents may
 *               be referred to by entries in one or more
 *               Indices.
 * - \b Active   The contents of the Partition are
 *               accessible and its objects may be
 *               referred to by one or more Indices.
 * - \b Offline  The contents of the Partition are not accessible.
 * - \b Busy     The Parition is being updated and cannot be changed.
 *
 * There are several commands available to manipulate
 * partitions.
 *
 * - \ref sos_part_create Create a new partition
 * - \ref sos_part_query Query the partitions in a container
 * - \ref sos_part_modify Modify the state of a partition
 * - \ref sos_part_move Move a partition to another storage location
 * - \ref sos_part_delete Destroy a partition
 *
 * There must be at least one partition in the container in the
 * 'primary' state in order for objects to be allocated and stored in
 * the container. For example:
 *
 *      sos_part_create -C theContainer -s primary "today"
 *
 * Use the sos_part_query command to see the new partition
 *
 *      sos_part_query -C theContainer
 *      Partition Name       RefCount Status           Size     Modified         Accessed         Path
 *      -------------------- -------- ---------------- -------- ---------------- ---------------- ----------------
 *      today                       2 PRIMARY               65K 2017/04/11 11:47 2017/04/11 11:47 /btrfs/test_data/theContainer
 *
 * Let's create another partition to contain tomorrow's data:
 *
 *      sos_part_create -C theContainer -s primary "tomorrow"
 *      sos_part_query -C theContainer -v
 *      Partition Name       RefCount Status           Size     Modified         Accessed         Path
 *      -------------------- -------- ---------------- -------- ---------------- ---------------- ----------------
 *      today                       2 PRIMARY               65K 2017/04/11 11:51 2017/04/11 11:51 /btrfs/test_data/theContainer
 *      tomorrow                    3 OFFLINE                                                     /btrfs/test_data/theContainer
 *
 * A typical use case for partitions is to group objects together by
 * date and then migrate objects from an older partitions to another
 * container on secondary storage.
 *
 * At midnight the administrator starts storing data in tomorrow's partition as follows:
 *
 *      sos_part_modify -C theContainer -s primary "tomorrow"
 *      sos_part_query -C theContainer -v
 *      Partition Name       RefCount Status           Size     Modified         Accessed         Path
 *      -------------------- -------- ---------------- -------- ---------------- ---------------- ----------------
 *      today                       2 PRIMARY               65K 2017/04/11 11:51 2017/04/11 11:51 /btrfs/test_data/theContainer
 *      tomorrow                    3 OFFLINE                                                     /btrfs/test_data/theContainer
 *
 * New object allocations will immediately start flowing into the new
 * primary partition. Objects in the original partition are still
 * accessible and indexed.
 *
 * The administrator then wants to migrate the data from the today
 * partition to another container in secondary storage. First create
 * the container to contain 'backups'.
 *
 *      sos_cmd -C /secondary/backupContainer -c
 *      sos_part_create -C theContainer -s primary "backup"
 *
 * Then export the contents of the today partition to the backup:
 *
 *      sos_part_export -C theContainer -E /secondary/backupContainer today
 *
 * All objects in the today partition are now in the
 * /secondary/backupContainer. They are also still in theContainer in
 * the today partition. To remove the today partition, delete the partition as follows:
 *
 *      sos_part_modify -C theContainer -s offline today
 *      sos_part_delete -C theContainer today
 *
 * There are API for manipulating Partitions from a program. In
 * general, only management applications should call these
 * functions. It is possible to corrupt and otherwise destroy the
 * object store by using these functions incorrectly.
 *
 * The Partition API include the following:
 *
 * - sos_part_create() Create a new partition
 * - sos_part_delete() Delete a partition
 * - sos_part_move() Move a parition to another storage location
 * - sos_part_copy() Copy a partition to another storage location
 * - sos_part_iter_new() Create a partition iterator
 * - sos_part_iter_free() Free a partition iterator
 * - sos_part_first() Return the first partition in the Container
 * - sos_part_next() Return the next partition in the Container
 * - sos_part_find() Find a partition by name
 * - sos_part_put() Drop a reference on the partition
 * - sos_part_stat() Return size and access information about a partition
 * - sos_part_state_set() Set the state of a parittion
 * - sos_part_name() Return the name of a partition
 * - sos_part_path() Return a partitions storage path
 */
#include <sys/queue.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/sendfile.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>

#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

static sos_part_t __sos_part_first(sos_part_iter_t iter);
static sos_part_t __sos_part_next(sos_part_iter_t iter);
static int __refresh_part_list(sos_t sos);

/*
 * New partition objects show up in the address space through:
 * __sos_part_create(), __sos_part_data_first(), and
 * __sos_part_data_next(). Each of these functions takes a reference
 * on the object on behalf of the caller.
 */
static ods_obj_t __sos_part_create(sos_t sos, char *tmp_path,
				   const char *part_name, const char *part_path)
{
	char real_path[PATH_MAX];
	int rc;
	struct stat sb;
	ods_ref_t head_ref, tail_ref;
	ods_obj_t part, new_part;

	/* Take the partition lock */
	ods_lock(sos->part_ods, 0, NULL);

	/* See if the specified name already exists in the filesystem */
	if (part_path == NULL)
		part_path = sos->path;
	else
		part_path = realpath(part_path, real_path);

	if (strlen(part_path) >= SOS_PART_PATH_LEN) {
		errno = E2BIG;
		goto err_1;
	}
	sprintf(tmp_path, "%s/%s", part_path, part_name);
	rc = stat(tmp_path, &sb);
	if (rc == 0 || (rc && errno != ENOENT)) {
		errno = EEXIST;
		goto err_1;
	}

	head_ref = SOS_PART_UDATA(sos->part_udata)->head;
	tail_ref = SOS_PART_UDATA(sos->part_udata)->tail;

	part = ods_ref_as_obj(sos->part_ods, head_ref);
	while (part) {
		ods_ref_t next_part;
		rc = strcmp(SOS_PART(part)->name, part_name);
		next_part = SOS_PART(part)->next;
		ods_obj_put(part);
		if (!rc) {
			errno = EEXIST;
			goto err_1;
		}
		if (next_part)
			part = ods_ref_as_obj(sos->part_ods, next_part);
		else
			part = NULL;
	}
	new_part = ods_obj_alloc(sos->part_ods, sizeof(struct sos_part_data_s));
	if (!new_part)
		goto err_1;

	/* Set up the new partition */
	strcpy(SOS_PART(new_part)->name, part_name);
	strcpy(SOS_PART(new_part)->path, part_path);
	SOS_PART(new_part)->part_id =
		ods_atomic_inc(&SOS_PART_UDATA(sos->part_udata)->next_part_id);
	SOS_PART(new_part)->prev = tail_ref;
	SOS_PART(new_part)->next = 0;
	SOS_PART(new_part)->ref_count = 1 + 1; /* create reference + caller ref */
	SOS_PART(new_part)->state = SOS_PART_STATE_OFFLINE;

	/* Insert it into the partition list */
	if (tail_ref) {
		ods_obj_t prev = ods_ref_as_obj(sos->part_ods, tail_ref);
		if (!prev)
			goto err_2;
		SOS_PART(prev)->next = ods_obj_ref(new_part);
		SOS_PART_UDATA(sos->part_udata)->tail = ods_obj_ref(new_part);
		if (!head_ref)
			SOS_PART_UDATA(sos->part_udata)->head = ods_obj_ref(new_part);
		ods_obj_put(prev);
	} else {
		SOS_PART_UDATA(sos->part_udata)->head = ods_obj_ref(new_part);
		SOS_PART_UDATA(sos->part_udata)->tail = ods_obj_ref(new_part);
	}
	ods_atomic_inc(&SOS_PART_UDATA(sos->part_udata)->gen);
	ods_unlock(sos->part_ods, 0);
	return new_part;
 err_2:
	ods_obj_delete(new_part);
	ods_obj_put(new_part);
 err_1:
	ods_unlock(sos->part_ods, 0);
	return NULL;
}

/*
 * This function uses the part reference. If the caller wants to
 * continue using it, it must take its own reference.
 */
static int __sos_open_partition(sos_t sos, sos_part_t part)
{
	char tmp_path[PATH_MAX];
	int rc;
	ods_t ods;

	sprintf(tmp_path, "%s/%s", sos_part_path(part), sos_part_name(part));
	assert(tmp_path[0] == '/');
	rc = __sos_make_all_dir(tmp_path, sos->o_mode);
	if (rc) {
		rc = errno;
		goto err_0;
	}
	sprintf(tmp_path, "%s/%s/objects", sos_part_path(part), sos_part_name(part));
 retry:
	ods = ods_open(tmp_path, sos->o_perm);
	if (!ods) {
		/* Create the ODS to contain the objects */
		rc = ods_create(tmp_path, sos->o_mode & ~(S_IXGRP|S_IXUSR|S_IXOTH));
		if (rc)
			goto err_0;
		goto retry;
	}
	part->obj_ods = ods;
	return 0;
 err_0:
	return rc;
}

struct iter_args {
	double start;
	double timeout;
	sos_part_t part;
	uint64_t count;
};
#define DUTY_CYCLE 250000

static int __unindex_callback_fn(ods_t ods, ods_obj_t obj, void *arg)
{
	sos_obj_ref_t ref;
	sos_obj_t sos_obj;
	struct iter_args *uarg = arg;
	sos_part_t part = uarg->part;
	sos_obj_data_t sos_obj_data = obj->as.ptr;
	sos_schema_t schema = sos_schema_by_id(part->sos, sos_obj_data->schema);
	if (!schema) {
		sos_warn("Object at %p is missing a valid schema id.\n", ods_obj_ref(obj));
		/* This is a garbage object that should not be here */
		return 0;
	}
	struct timeval tv;
	(void)gettimeofday(&tv, NULL);
	double now = ((double)tv.tv_sec * 1.0e6) + (double)tv.tv_usec;
	double dur = now - uarg->start;
	uarg->count++;
	if (now - uarg->start > uarg->timeout) {
		sos_info("Processed %ld objects in %f microseconds\n", uarg->count, dur);
		return 1;
	}
	ref.ref.ods = SOS_PART(part->part_obj)->part_id;
	ref.ref.obj = ods_obj_ref(obj);
	sos_obj = __sos_init_obj(part->sos, schema, obj, ref);
	sos_obj_remove(sos_obj);
	sos_obj_put(sos_obj);
	return 0;
}

void __unindex_part_objects(sos_t sos, sos_part_t part)
{
	int rc;
	struct iter_args uargs;
	struct ods_obj_iter_pos_s pos;

	/*
	 * Remove all objects in this partition from the indices
	 */
	ods_obj_iter_pos_init(&pos);
	do {
		struct timeval tv;
		(void)gettimeofday(&tv, NULL);
		uargs.start = (double)tv.tv_sec * 1.0e6 + (double)tv.tv_usec;
		uargs.timeout = DUTY_CYCLE;
		uargs.part = part;
		uargs.count = 0;
		rc = ods_obj_iter(part->obj_ods, &pos, __unindex_callback_fn, &uargs);
		if (rc) {
			usleep(1000000 - DUTY_CYCLE);
		}
	} while (rc);
}

void __make_part_offline(sos_t sos, sos_part_t part)
{
	SOS_PART(part->part_obj)->state = SOS_PART_STATE_OFFLINE;
	ods_atomic_inc(&SOS_PART_UDATA(sos->part_udata)->gen);
}

void __make_part_busy(sos_t sos, sos_part_t part)
{
	SOS_PART(part->part_obj)->state = SOS_PART_STATE_BUSY;
	ods_atomic_inc(&SOS_PART_UDATA(sos->part_udata)->gen);
}

/*
 * Iterate through all objects in the partition and call sos_obj_index
 * for each one. This function is used when partitions are moving from
 * OFFLINE --> ACTIVE or OFFLINE --> PRIMARY
 */
static int __reindex_callback_fn(ods_t ods, ods_obj_t obj, void *arg)
{
	struct timeval tv;
	int rc;
	sos_obj_ref_t ref;
	sos_obj_t sos_obj;
	struct iter_args *rarg = arg;
	sos_part_t part = rarg->part;
	sos_obj_data_t sos_obj_data = obj->as.ptr;
	sos_schema_t schema = sos_schema_by_id(part->sos, sos_obj_data->schema);
	if (!schema) {
		sos_warn("Object at %p is missing a valid schema id.\n", ods_obj_ref(obj));
		/* This is a garbage object that should not be here */
		return 0;
	}
	rc = gettimeofday(&tv, NULL);
	double now = ((double)tv.tv_sec * 1.0e6) + (double)tv.tv_usec;
	double dur = now - rarg->start;
	rarg->count++;
	if (now - rarg->start > rarg->timeout) {
		sos_info("Processed %ld objects in %f microseconds\n", rarg->count, dur);
		return 1;
	}
	ref.ref.ods = SOS_PART(part->part_obj)->part_id;
	ref.ref.obj = ods_obj_ref(obj);
	sos_obj = __sos_init_obj(part->sos, schema, obj, ref);
	rc = sos_obj_index(sos_obj);
	if (rc) {
		/* The object couldn't be indexed for some reason */
		sos_warn("The object of type '%s' at %p could not be indexed.\n",
			 sos_schema_name(schema), ref.ref.obj);
	}
	sos_obj_put(sos_obj);
	return 0;
}

static int __reindex_part_objects(sos_t sos, sos_part_t part)
{
	struct ods_obj_iter_pos_s pos;
	struct timeval tv;
	struct iter_args rargs;
	int rc;
	ods_obj_iter_pos_init(&pos);
	do {
		rc = gettimeofday(&tv, NULL);
		rargs.start = (double)tv.tv_sec * 1.0e6 + (double)tv.tv_usec;
		rargs.timeout = DUTY_CYCLE;
		rargs.part = part;
		rargs.count = 0;
		rc = ods_obj_iter(part->obj_ods, &pos, __reindex_callback_fn, &rargs);
		if (rc) {
			usleep(1000000 - DUTY_CYCLE);
		}
	} while (rc);
	return 0;
}

static void __make_part_active(sos_t sos, sos_part_t part)
{
	SOS_PART(part->part_obj)->state = SOS_PART_STATE_ACTIVE;
	ods_atomic_inc(&SOS_PART_UDATA(sos->part_udata)->gen);

	/* Open the partition and add it to the container */
	(void)__sos_open_partition(part->sos, part);
}

static void __make_part_primary(sos_t sos, sos_part_t part)
{
	ods_ref_t cur_ref;
	ods_obj_t cur_primary;

	/* Fix-up the current primary */
	cur_ref = SOS_PART_UDATA(sos->part_udata)->primary;
	if (cur_ref) {
		cur_primary = ods_ref_as_obj(sos->part_ods, cur_ref);
		if (!cur_primary) {
			sos_fatal("Could not get object for primary "
				  "partition reference %p at %s:%d",
				  cur_ref, __func__, __LINE__);
			return;
		}
		SOS_PART(cur_primary)->state = SOS_PART_STATE_ACTIVE;
		ods_obj_put(cur_primary);
	}
	/* Make part_obj primary */
	SOS_PART(part->part_obj)->state = SOS_PART_STATE_PRIMARY;
	ods_atomic_inc(&SOS_PART_UDATA(sos->part_udata)->gen);
	SOS_PART_UDATA(sos->part_udata)->primary = ods_obj_ref(part->part_obj);
	sos->primary_part = part;
}

sos_part_t __sos_part_find_by_ref(sos_t sos, ods_ref_t ref)
{
	sos_part_t part;
	sos_part_iter_t iter = sos_part_iter_new(sos);
	if (!iter)
		return NULL;

	for (part = __sos_part_first(iter); part; part = __sos_part_next(iter)) {
		if (ref == ods_obj_ref(part->part_obj))
			goto out;
		sos_part_put(part);
	}
 out:
	sos_part_iter_free(iter);
	return part;
}
/**
 * \brief Set the state of a partition
 *
 * Partition state transitions are limited by the current state. If
 * the current state is PRIMARY, there are no state changes
 * allowed. The only way to change the state of the PRIMARY partition
 * is to make another partition PRIMARY. When this occurs, the
 * partition is made ACTIVE.
 *
 * Any partition in the ACTIVE state can be made PRIMARY. Any
 * partition in the ACTIVE state can be made OFFLINE and an OFFLINE
 * partition can be made ACTIVE. Note that when a parition is made
 * OFFLINE, all Keys that refer to objects in that partition are
 * removed from all indices in the container.
 *
 * \param part The partition handle
 * \param new_state The desired state of the partition
 * \retval 0 The state was successfully changed
 * \retval EINVAL The specified state is invalid given the current
 * state of the partition.
 */
int sos_part_state_set(sos_part_t part, sos_part_state_t new_state)
{
	sos_t sos = part->sos;
	int rc = 0;
	sos_part_state_t cur_state;

	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ods, 0, NULL);
	cur_state = SOS_PART(part->part_obj)->state;
	if (cur_state == SOS_PART_STATE_BUSY
	    || cur_state == SOS_PART_STATE_PRIMARY) {
		rc = EBUSY;
		goto out;
	}
	__make_part_busy(sos, part);
	ods_unlock(sos->part_ods, 0);
	pthread_mutex_unlock(&sos->lock);

	switch (cur_state) {
	case SOS_PART_STATE_OFFLINE:
		switch (new_state) {
		case SOS_PART_STATE_ACTIVE:
		case SOS_PART_STATE_PRIMARY:
			rc = __sos_open_partition(sos, part);
			if (rc) {
				pthread_mutex_lock(&sos->lock);
				ods_lock(sos->part_ods, 0, NULL);
				__make_part_offline(sos, part);
				ods_unlock(sos->part_ods, 0);
				pthread_mutex_unlock(&sos->lock);
				return EINVAL;
			}
			__reindex_part_objects(sos, part);
			break;
		default:
			break;
		}
		break;
	case SOS_PART_STATE_ACTIVE:
		switch (new_state) {
		case SOS_PART_STATE_OFFLINE:
			__unindex_part_objects(sos, part);
			break;
		default:
			break;
		}
		break;
	default:
		break;
	}

	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ods, 0, NULL);
	assert(SOS_PART_STATE_BUSY == SOS_PART(part->part_obj)->state);

	switch (cur_state) {
	case SOS_PART_STATE_OFFLINE:
		switch (new_state) {
		case SOS_PART_STATE_BUSY:
			rc = EINVAL;
			break;
		case SOS_PART_STATE_OFFLINE:
			break;
		case SOS_PART_STATE_ACTIVE:
			__make_part_active(sos, part);
			__refresh_part_list(sos);
			break;
		case SOS_PART_STATE_PRIMARY:
			__make_part_primary(sos, part);
			__refresh_part_list(sos);
			break;
		default:
			rc = EINVAL;
			break;
		}
		break;
	case SOS_PART_STATE_ACTIVE:
		switch (new_state) {
		case SOS_PART_STATE_BUSY:
			rc = EINVAL;
			break;
		case SOS_PART_STATE_OFFLINE:
			__make_part_offline(sos, part);
			__refresh_part_list(sos);
			break;
		case SOS_PART_STATE_ACTIVE:
			break;
		case SOS_PART_STATE_PRIMARY:
			__make_part_primary(sos, part);
			break;
		default:
			rc = EINVAL;
			break;
		}
		break;
	default:
		assert(0);
	}
 out:
	ods_unlock(sos->part_ods, 0);
	pthread_mutex_unlock(&sos->lock);
	return rc;
}

static sos_part_t __sos_part_new(sos_t sos, ods_obj_t part_obj)
{
	sos_part_t part = calloc(1, sizeof(*part));
	if (!part)
		return NULL;
	part->ref_count = 1;
	part->sos = sos;
	part->part_obj = part_obj;
	return part;
}


static int __refresh_part_list(sos_t sos)
{
	int rc = 0;
	sos_part_t part;
	ods_obj_t part_obj;

	sos->primary_part = NULL;
	while (!TAILQ_EMPTY(&sos->part_list)) {
		part = TAILQ_FIRST(&sos->part_list);
		sos_part_put(part);
		TAILQ_REMOVE(&sos->part_list, part, entry);
	}
	sos->part_gn = SOS_PART_UDATA(sos->part_udata)->gen;
	for (part_obj = __sos_part_data_first(sos);
	     part_obj; part_obj = __sos_part_data_next(sos, part_obj)) {
		part = __sos_part_new(sos, part_obj);
		if (!part) {
			rc = ENOMEM;
			goto out;
		}
		TAILQ_INSERT_TAIL(&sos->part_list, part, entry);
		if (SOS_PART(part_obj)->state != SOS_PART_STATE_OFFLINE) {
			rc = __sos_open_partition(sos, part);
			if (rc)
				goto out;
		}
	}
 out:
	return rc;
}

static int refresh_part_list(sos_t sos)
{
	int rc;
	ods_lock(sos->part_ods, 0, NULL);
	rc = __refresh_part_list(sos);
	ods_unlock(sos->part_ods, 0);
	return rc;
}

int __sos_open_partitions(sos_t sos, char *tmp_path)
{
	int rc;

	/* Open the partition ODS */
	sprintf(tmp_path, "%s/.__part", sos->path);
	sos->part_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!sos->part_ods)
		goto err_0;
	sos->part_udata = ods_get_user_data(sos->part_ods);
	if (!sos->part_udata)
		goto err_1;
	rc = refresh_part_list(sos);
	return rc;
 err_1:
	ods_close(sos->part_ods, ODS_COMMIT_ASYNC);
	sos->part_ods = NULL;
 err_0:
	return errno;
}

sos_part_t __sos_primary_obj_part(sos_t sos)
{
	sos_part_t part = NULL;

	if ((NULL != sos->primary_part) &&
	    (SOS_PART(sos->primary_part->part_obj)->state == SOS_PART_STATE_PRIMARY))
		return sos->primary_part;

	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ods, 0, NULL);
	if (sos->part_gn != SOS_PART_UDATA(sos->part_udata)->gen) {
		int rc = __refresh_part_list(sos);
		if (rc)
			goto out;
	}
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (SOS_PART(part->part_obj)->state == SOS_PART_STATE_PRIMARY) {
			sos->primary_part = part;
			goto out;
		}
	}
 out:
	ods_unlock(sos->part_ods, 0);
	pthread_mutex_unlock(&sos->lock);
	return part;
}

struct export_obj_iter_args_s {
	sos_t src_sos;
	sos_t dst_sos;
	sos_part_t src_part;
	ods_idx_t exp_idx;
	int reindex;
	int64_t export_count;
};

#pragma pack(4)
union exp_obj_u {
	struct ods_idx_data_s idx_data;
	struct exp_obj_s {
		uint64_t from_ref;
		uint64_t to_ref;
	} exp_data;
};
#pragma pack()

static sos_schema_t __export_schema(sos_t sos, sos_schema_t src, const char *name)
{
	sos_schema_t schema = sos_schema_by_name(sos, name);
	if (!schema) {
		/* Duplicate and add the src_schema */
		schema = sos_schema_dup(src);
		if (!schema)
			return NULL;

		/* Add the schema to the destination container */
		int rc = sos_schema_add(sos, schema);
		if (rc) {
			errno = rc;
			return NULL;
		}
	}
	return schema;
}

static int __shallow_export(sos_t dst_sos, sos_t src_sos,
			    ods_obj_t src_ods_obj, ods_idx_t exp_idx,
			    sos_schema_t *dst_schema, sos_schema_t *src_schema,
			    sos_obj_ref_t *dst_ref, ods_obj_t *dst_ods_obj)
{
	ods_obj_t dobj;
	sos_obj_data_t sos_obj_data = src_ods_obj->as.ptr;
	const char *schema_name;
	sos_schema_t dschema, sschema;
	sos_part_t part;
	int rc;
	ODS_KEY(exp_key);
	union exp_obj_u exp;

	sschema = sos_schema_by_id(src_sos, sos_obj_data->schema);
	if (!sschema) {
		sos_error("ODS object with ref %p has the invalid schema id %ld\n",
			  (void *)ods_obj_ref(src_ods_obj), sos_obj_data->schema);
		return EINVAL;
	}

	schema_name = sos_schema_name(sschema);
	dschema = __export_schema(dst_sos, sschema, schema_name);
	if (!dschema) {
		printf("Error %d adding schema %s to the destination "
		       "container.\n", errno, schema_name);
		return errno;
	}

	part = __sos_primary_obj_part(dst_sos);
	if (!part)
		return ENOSPC;

	/* Check to see if this object already exists in the
	 * destination container. If it does, it was processed by
	 * virtue of being referred to by a previously exported
	 * object
	 */
	exp.exp_data.from_ref = ods_obj_ref(src_ods_obj);
	ods_key_set(&exp_key, &exp.exp_data.from_ref, sizeof(exp.exp_data.from_ref));
	rc = ods_idx_find(exp_idx, &exp_key, &exp.idx_data);
	if (rc == 0) {
		dobj = ods_ref_as_obj(part->obj_ods, exp.exp_data.to_ref);
		if (!dobj)
			return ENOSPC;
	} else {
		dobj = __sos_obj_new(part->obj_ods, src_ods_obj->size, &dst_sos->lock);
		if (!dobj)
			return ENOSPC;

		memcpy(dobj->as.ptr, src_ods_obj->as.ptr, src_ods_obj->size);
		SOS_OBJ(dobj)->schema = dschema->data->id;

		/* Add this object to the index of objects that we've created */
		exp.exp_data.from_ref = ods_obj_ref(src_ods_obj);
		exp.exp_data.to_ref = ods_obj_ref(dobj);
		sos_key_set(&exp_key, &exp.exp_data.from_ref, sizeof(exp.exp_data.from_ref));
		rc = ods_idx_insert(exp_idx, &exp_key, exp.idx_data);
		if (rc) {
			printf("Error indexing exported object.\n");
			return rc;
		}
	}

	if (dst_ref) {
		dst_ref->ref.ods = SOS_PART(part->part_obj)->part_id;
		dst_ref->ref.obj = ods_obj_ref(dobj);
	}
	if (dst_ods_obj)
		*dst_ods_obj = dobj;
	if (dst_schema)
		*dst_schema = dschema;
	if (src_schema)
		*src_schema = sschema;

	return 0;
}

/**
 * For each object in the source container/partition:
 *     1. look up the source object reference to see if it has
 *        been exported it to the new partition.
 *
 *     2. if found, continue
 *
 *     3. create a new object in the dest container
 *
 *     4. Add a tracking object to the obj_idx so we can find
 *        this new object later in 1. above, or 5. below
 *
 *     5. Run through each attribute in the object and if it
 *        is a reference, look it up and/or create it.
 */
static int __export_callback_fn(ods_t ods, ods_obj_t src_ods_obj, void *arg)
{
	struct export_obj_iter_args_s *uarg = arg;
	sos_t src_sos = uarg->src_sos;
	sos_t dst_sos = uarg->dst_sos;
	sos_obj_t dst_sos_obj;
	ods_obj_t dst_ods_obj;
	sos_schema_t src_schema, dst_schema;
	sos_obj_ref_t dst_ref;
	int rc;

	rc = __shallow_export(dst_sos, src_sos, src_ods_obj, uarg->exp_idx,
			      &dst_schema, &src_schema, &dst_ref, &dst_ods_obj);
	if (rc)
		return rc;

	/* If this is an internal schema object, no more processing is required */
	if (dst_schema->flags & SOS_SCHEMA_F_INTERNAL) {
		ods_obj_put(dst_ods_obj);
		return 0;
	}

	/* Instantiate a SOS version of the destination ODS object */
	dst_sos_obj = __sos_init_obj(dst_sos, dst_schema, dst_ods_obj, dst_ref);
	if (!dst_sos_obj)
		return ENOMEM;

	/* Run through the attributes of the object, instantiate any
	 * reference objects, and fix-up the reference values in the
	 * destination object.
	 */
	int attr_id;
	for  (attr_id = 0; attr_id < sos_schema_attr_count(src_schema); attr_id++) {
		sos_value_data_t ref_val;
		sos_attr_t src_attr = sos_schema_attr_by_id(src_schema, attr_id);
		if (!src_attr) {
			printf("Error instantiating attribute id %d in schema %s\n",
			       attr_id, sos_schema_name(src_schema));
			return EINVAL;
		}
		sos_type_t type = sos_attr_type(src_attr);
		if (type < SOS_TYPE_ARRAY && type != SOS_TYPE_OBJ)
			continue;

		ref_val = (sos_value_data_t)&src_ods_obj->as.bytes[src_attr->data->offset];

		/* The reference might point to an object in another partition :-\ */
		ods_t ods = __sos_ods_from_ref(src_sos, ref_val->prim.ref_.ref.ods);
		if (!ods)
			continue;

		ods_obj_t src_attr_obj = ods_ref_as_obj(ods, ref_val->prim.ref_.ref.obj);
		if (!src_attr_obj)
			continue;

		ods_obj_t dst_attr_obj;
		rc = __shallow_export(dst_sos, src_sos, src_attr_obj, uarg->exp_idx,
				      NULL, NULL, &dst_ref, &dst_attr_obj);
		ods_obj_put(src_attr_obj);
		if (0 == rc) {
			ref_val = (sos_value_data_t)&dst_ods_obj->as.bytes[src_attr->data->offset];
			ref_val->prim.ref_ = dst_ref;
			ods_obj_put(dst_attr_obj);
		} else {
			printf("Error exporting internal reference attribute %s\n", sos_attr_name(src_attr));
		}
	}
	if (uarg->reindex)
		sos_obj_index(dst_sos_obj);
	sos_obj_put(dst_sos_obj);
	uarg->export_count ++;
	return 0;
}

/**
 * \brief Export the objects in a partition to another container
 *
 * Export all of the objects in the specified partition to another
 * container. The destination partition in the destination container
 * will be the primary partition in the destination container.
 *
 * The source partition must not be the primary partition in order to
 * ensure that every object will be exported.
 *
 * The source container (the container in which src_part is located)
 * cannot be the same as the destination container.
 *
 * \param src_part	The source partition handle
 * \param dst_cont	The destination container
 * \param reindex	Set to 1 to add exported objects to their schema indices
 * \retval		>= 0 The number of objects exported to the
 *			     destination container
 * \retval		<0   An error occured, see errno for more information
 */
int64_t sos_part_export(sos_part_t src_part, sos_t dst_sos, int reindex)
{
	sos_t src_sos = src_part->sos;
	uint64_t rc = 0;
	sos_part_state_t cur_state;
	struct export_obj_iter_args_s uargs;
	struct ods_obj_iter_pos_s pos;

	/* The source container cannot be the same as the destination
	 * container */
	if (src_sos == dst_sos) {
		return -EINVAL;
		errno = EINVAL;
	}

	/* If the state is PRIMARY or BUSY, return EBUSY */
	pthread_mutex_lock(&src_sos->lock);
	ods_lock(src_sos->part_ods, 0, NULL);
	cur_state = SOS_PART(src_part->part_obj)->state;
	if (cur_state == SOS_PART_STATE_BUSY
	    || cur_state == SOS_PART_STATE_PRIMARY) {
		errno = EBUSY;
		rc = -errno;
		ods_unlock(src_sos->part_ods, 0);
		goto err;
	}

	/* Create an index to track objects that are created so that
	 * we handle objects that are referenced by other objects,
	 * e.g. arrays. The index name is 'partition_name' && '_export'
	 */
	char idx_path[PATH_MAX];
	snprintf(idx_path, sizeof(idx_path), "%s/%s_export",
		 dst_sos->path, sos_part_name(src_part));
	rc = ods_idx_create(idx_path, 0600, "BXTREE", "UINT64", NULL);
	if (rc) {
		errno = rc;
		rc = -rc;
		ods_unlock(src_sos->part_ods, 0);
		goto err;
	}
	uargs.exp_idx = ods_idx_open(idx_path, ODS_PERM_RW);
	if (!uargs.exp_idx) {
		rc = -errno;
		ods_unlock(src_sos->part_ods, 0);
		goto err_1;
	}
	/* Make the source partition busy to prevent changes while the
	 * data is being copied
	 */
	__make_part_busy(src_sos, src_part);
	ods_unlock(src_sos->part_ods, 0);
	pthread_mutex_unlock(&src_sos->lock);

	uargs.src_sos = src_sos;
	uargs.src_part = src_part;
	uargs.dst_sos = dst_sos;
	uargs.export_count = 0;
	uargs.reindex = reindex;

	/* Export all objects in src_part to the destination container */
	ods_obj_iter_pos_init(&pos);
	rc = ods_obj_iter(src_part->obj_ods, &pos, __export_callback_fn, &uargs);

	/* Restore the source partition state */
	pthread_mutex_lock(&src_sos->lock);
	ods_lock(src_sos->part_ods, 0, NULL);
	SOS_PART(src_part->part_obj)->state = cur_state;
	ods_unlock(src_sos->part_ods, 0);
	pthread_mutex_unlock(&src_sos->lock);

	ods_destroy(idx_path);
	if (rc) {
		errno = rc;
		return -rc;
	}
	return uargs.export_count;
 err_1:
	ods_destroy(idx_path);
 err:
	pthread_mutex_unlock(&src_sos->lock);
	return rc;
}

static int __index_callback_fn(ods_t ods, ods_obj_t ods_obj, void *arg)
{
	struct export_obj_iter_args_s *uarg = arg;
	sos_t sos = uarg->src_sos;
	sos_part_t part = uarg->src_part;
	sos_obj_data_t sos_obj_data = ods_obj->as.ptr;
	sos_obj_t sos_obj;
	sos_schema_t schema;
	sos_obj_ref_t ref;

	schema = sos_schema_by_id(sos, sos_obj_data->schema);
	if (!schema) {
		sos_warn("An object with the invalid schema id %d was "
			 "encountered at %p.\n", sos_obj_data->schema,
			 ods_obj_ref(ods_obj));
		return EINVAL;
	}

	/* Internal schema objects are not indexed. */
	if (schema->flags & SOS_SCHEMA_F_INTERNAL)
		return 0;

	/* Instantiate a SOS version of the ODS object */
	ref.ref.ods = SOS_PART(part->part_obj)->part_id;
	ref.ref.obj = ods_obj_ref(ods_obj);
	sos_obj = __sos_init_obj(sos, schema, ods_obj, ref);
	if (!sos_obj)
		return ENOMEM;

	if (sos_obj_index(sos_obj))
		sos_warn("The object at %p could not be indexed: errno %d\n",
			 ref.ref.ods, ref.ref.obj, errno);
	sos_obj_put(sos_obj);
	uarg->export_count ++;
	return 0;
}

int64_t sos_part_index(sos_part_t part)
{
	sos_t sos = part->sos;
	uint64_t rc = 0;
	sos_part_state_t cur_state;
	struct export_obj_iter_args_s uargs;
	struct ods_obj_iter_pos_s pos;

	/* If the state is BUSY, return EBUSY */
	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ods, 0, NULL);
	cur_state = SOS_PART(part->part_obj)->state;
	if (cur_state == SOS_PART_STATE_BUSY
	    || cur_state == SOS_PART_STATE_PRIMARY) {
		sos_info("Cannot index a partition in the BUSY or PRIMARY states.\n");
		errno = EBUSY;
		rc = -errno;
		ods_unlock(sos->part_ods, 0);
		goto err;
	}

	/* Make the source partition busy to prevent changes while the
	 * data is being copied
	 */
	__make_part_busy(sos, part);
	ods_unlock(sos->part_ods, 0);
	pthread_mutex_unlock(&sos->lock);

	memset(&uargs, 0, sizeof(uargs));
	uargs.src_sos = sos;
	uargs.src_part = part;

	/* Index all objects in part */
	ods_obj_iter_pos_init(&pos);
	rc = ods_obj_iter(part->obj_ods, &pos, __index_callback_fn, &uargs);

	/* Restore the source partition state */
	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ods, 0, NULL);
	SOS_PART(part->part_obj)->state = cur_state;
	ods_unlock(sos->part_ods, 0);
	pthread_mutex_unlock(&sos->lock);

	return uargs.export_count;
 err:
	pthread_mutex_unlock(&sos->lock);
	return rc;
}

/**
 * \brief Drop a reference on a partition
 *
 * Partitions are reference counted. When the reference count goes to
 * zero, it is destroyed and all of its storage is released. The
 * sos_part_first(), sos_part_next(), and sos_part_find() functions
 * take a reference on behalf of the application. This reference
 * should be dropped by the application when the application is
 * finished with the partition.
 *
 * \param part The partition handle.
 */
void sos_part_put(sos_part_t part)
{
	if (0 == ods_atomic_dec(&part->ref_count)) {
		__sos_part_obj_put(part->sos, part->part_obj);
		ods_obj_put(part->part_obj);
		ods_close(part->obj_ods, ODS_COMMIT_ASYNC);
		free(part);
	}
}

static void __sos_part_obj_delete(sos_t sos, ods_obj_t part_obj)
{
	ods_ref_t prev_ref, next_ref;

	prev_ref = SOS_PART(part_obj)->prev;
	next_ref = SOS_PART(part_obj)->next;

	if (prev_ref) {
		ods_obj_t prev = ods_ref_as_obj(sos->part_ods, prev_ref);
		SOS_PART(prev)->next = next_ref;
	} else {
		SOS_PART_UDATA(sos->part_udata)->head = next_ref;
	}
	if (next_ref) {
		ods_obj_t next = ods_ref_as_obj(sos->part_ods, next_ref);
		SOS_PART(next)->prev = prev_ref;
	} else {
		SOS_PART_UDATA(sos->part_udata)->tail = prev_ref;
	}
	ods_obj_delete(part_obj);
}

ods_obj_t __sos_part_obj_get(sos_t sos, ods_obj_t part_obj)
{
	ods_atomic_inc(&SOS_PART(part_obj)->ref_count);
	return part_obj;
}

void __sos_part_obj_put(sos_t sos, ods_obj_t part_obj)
{
	if (0 == ods_atomic_dec(&SOS_PART(part_obj)->ref_count)) {
		if (SOS_PART(part_obj)->state == SOS_PART_STATE_OFFLINE) {
			sos_error("Reference count has gone to zero on "
				  "parition %s with state %d\n",
				  SOS_PART(part_obj)->name,
				  SOS_PART(part_obj)->state);
		} else {
			__sos_part_obj_delete(sos, part_obj);
		}
	}
}

ods_obj_t __sos_part_data_first(sos_t sos)
{
	ods_obj_t part_obj;

	/* Take the partition lock */
	part_obj = ods_ref_as_obj(sos->part_ods, SOS_PART_UDATA(sos->part_udata)->head);
	if (part_obj)
		__sos_part_obj_get(sos, part_obj);
	return part_obj;
}

ods_obj_t __sos_part_data_next(sos_t sos, ods_obj_t part_obj)
{
	ods_obj_t next_obj;
	ods_ref_t next_ref = 0;

	if (!part_obj)
		return NULL;

	next_ref = SOS_PART(part_obj)->next;
	if (!next_ref)
		return NULL;

	/* Take the partition lock */
	next_obj = ods_ref_as_obj(sos->part_ods, next_ref);
	if (next_obj)
		__sos_part_obj_get(sos, next_obj);
	return next_obj;
}

/**
 * \brief Free the memory associated with the Iterator
 *
 * \param iter The iterator handle
 */
void sos_part_iter_free(sos_part_iter_t iter)
{
	if (iter->part)
		sos_part_put(iter->part);
	free(iter);
}

/**
 * \brief Create a new partition
 *
 * \param sos The sos_t container handle
 * \param part_name The name of the new partition.
 * \param part_path An optional path to the partition. If null,
 *                  the container path will be used.
 * \retval 0 Success
 * \retval EEXIST The specified partition already exists
 * \retval EBADF Invalid container handle or other storage error
 * \retval ENOMEM Insufficient resources
 */
int sos_part_create(sos_t sos, const char *part_name, const char *part_path)
{
	char tmp_path[PATH_MAX];
	ods_obj_t part_obj;
	sos_part_t part;

	part = sos_part_find(sos, part_name);
	if (part) {
		sos_part_put(part);
		return EEXIST;
	}

	part_obj = __sos_part_create(sos, tmp_path, part_name, part_path);
	if (!part_obj)
		return errno;

	part = __sos_part_new(sos, part_obj);
	pthread_mutex_lock(&sos->lock);
	TAILQ_INSERT_HEAD(&sos->part_list, part, entry);
	pthread_mutex_unlock(&sos->lock);
	return 0;
}

/**
 * \brief Create a partition iterator
 *
 * A partition iterator is used to iterate through all partitions
 * in the container.
 *
 * \param sos The container handle
 * \returns A partition iterator handle
 */
sos_part_iter_t sos_part_iter_new(sos_t sos)
{
	sos_part_iter_t iter = calloc(1, sizeof(struct sos_part_iter_s));
	iter->sos = sos;
	return iter;
}

/**
 * \brief Find a partition
 *
 * Returns the partition handle for the partition with the name
 * specified by the \c name parameter.
 *
 * The application should call sos_part_put() when finished with the
 * partition object. Note that calling sos_part_put() too many times
 * can result in destroying the partition.
 *
 * \param sos The container handle
 * \param name The name of the partition
 * \retval Partition handle
 * \retval NULL if the partition was not found
 */

sos_part_t sos_part_find(sos_t sos, const char *name)
{
	sos_part_t part;
	sos_part_iter_t iter = sos_part_iter_new(sos);
	if (!iter)
		return NULL;

	for (part = sos_part_first(iter); part; part = sos_part_next(iter)) {
		if (0 == strcmp(sos_part_name(part), name))
			goto out;
		sos_part_put(part);
	}
	part = NULL;
 out:
	sos_part_iter_free(iter);
	return part;
}

static sos_part_t __sos_part_first(sos_part_iter_t iter)
{
	sos_part_t part = NULL;
	part = TAILQ_FIRST(&iter->sos->part_list);
	if (iter->part)
		sos_part_put(iter->part); /* drop iterator ref */
	iter->part = part;
	if (part) {
		ods_atomic_inc(&part->ref_count); /* iterator reference */
		ods_atomic_inc(&part->ref_count); /* application reference */
	}
	return part;
}

/**
 * \brief Return the first partition in the container
 *
 * The application should call sos_part_put() when finished with the
 * partition object. Note that calling sos_part_put() too many times
 * can result in destroying the partition.
 *
 * \param iter The partition iterator handle
 * \returns The first partition object or NULL if there are no
 * partitions in the container.
 */
sos_part_t sos_part_first(sos_part_iter_t iter)
{
	sos_part_t part;
	pthread_mutex_lock(&iter->sos->lock);
	part = __sos_part_first(iter);
	pthread_mutex_unlock(&iter->sos->lock);
	return part;
}

static sos_part_t __sos_part_next(sos_part_iter_t iter)
{
	sos_part_t part;
	if (!iter->part)
		return NULL;
	part = TAILQ_NEXT(iter->part, entry);
	sos_part_put(iter->part);	  /* drop iterator reference */
	iter->part = part;
	if (part) {
		ods_atomic_inc(&part->ref_count); /* iterator reference */
		ods_atomic_inc(&part->ref_count); /* application reference */
	}
	return part;
}

/**
 * \brief Return the next partition in the container
 *
 * The application should call sos_part_put() when finished with the
 * partition object. Note that calling sos_part_put() too many times
 * can result in destroying the partition.
 *
 * \param iter The partition iterator handle
 * \returns The next partition object or NULL if there are no more
 * partitions in the container.
 */
sos_part_t sos_part_next(sos_part_iter_t iter)
{
	sos_part_t part;
	pthread_mutex_lock(&iter->sos->lock);
	part = __sos_part_next(iter);
	pthread_mutex_unlock(&iter->sos->lock);
	return part;
}
/**
 * \brief Return the partition's name
 * \param part The partition handle
 * \returns Pointer to a string containing the name
 */
const char *sos_part_name(sos_part_t part)
{
	return SOS_PART(part->part_obj)->name;
}

/**
 * \brief Return the partition's path
 * \param part The partition handle
 * \returns Pointer to a string containing the path
 */
const char *sos_part_path(sos_part_t part)
{
	return SOS_PART(part->part_obj)->path;
}

/**
 * \brief Return the partition's state
 * \param part The partition handle
 * \returns An integer representing the partition's state
 */
sos_part_state_t sos_part_state(sos_part_t part)
{
	return SOS_PART(part->part_obj)->state;
}

/**
 * \brief Return the partition's id
 *
 * Returns the parititions unique 32b id. This id is part of an
 * object's sos_obj_ref_t and identifies the partition in which the
 * object is allocated.
 *
 * \param part The partition handle
 * \returns An integer representing the partition's id
 */
uint32_t sos_part_id(sos_part_t part)
{
	return SOS_PART(part->part_obj)->part_id;
}

/**
 * \brief Return the number of references on the partition
 *
 * \param part The partition handle
 * \retval An 32b integer representing the partition's reference count
 */
uint32_t sos_part_refcount(sos_part_t part)
{
	return SOS_PART(part->part_obj)->ref_count;
}

/**
 * \brief Delete a partition
 *
 * Deletes the paritition specified by the handle. All object storage
 * associated with the parition will be freed. The parition must be in
 * the OFFLINE state to be deleted.
 *
 * \param part The partition handle
 * \retval 0 The parition was deleted
 * \retval EBUSY The partition is not offline
 */
int sos_part_delete(sos_part_t part)
{
	int rc = EBUSY;
	sos_t sos = part->sos;
	sos_part_state_t cur_state;

	pthread_mutex_unlock(&sos->lock);
	ods_lock(sos->part_ods, 0, NULL);
	cur_state = SOS_PART(part->part_obj)->state;
	if (cur_state != SOS_PART_STATE_OFFLINE)
		goto out;

	rc = 0;
	/* Remove the partition from the container */
	TAILQ_REMOVE(&sos->part_list, part, entry);
	/* Put the create reference, we still hold the part reference */
	__sos_part_obj_put(sos, part->part_obj);
	/* Put the container reference */
	sos_part_put(part);
	/* Put the app reference reference */
	sos_part_put(part);
 out:
	ods_unlock(sos->part_ods, 0);
	pthread_mutex_unlock(&sos->lock);
	return rc;
}

/**
 * \brief Move a partition
 *
 * Move a partition from its current storage location to another. Any
 * references to objects in the partition by indices or other objects
 * are preserved by the move.
 *
 * \param part The partition handle
 * \param path The new path for the partition.
 * \retval 0 The parition was deleted
 * \retval EBUSY The partition is not offline
 * \retval EINVAL The destination path is the same as the source path
 * \retval EEXIST The destination path already exists
 * \retval EPERM The user has insufficient privilege to write to the destination path
 */
int sos_part_move(sos_part_t part, const char *new_part_path)
{
	char *old_part_path;
	char tmp_path[PATH_MAX];
	off_t offset;
	ssize_t sz;
	int in_fd, out_fd;
	struct stat sb;
	int rc;
	sos_t sos = part->sos;
	sos_part_state_t cur_state;

	old_part_path = strdup(sos_part_path(part));
	if (!old_part_path)
		return ENOMEM;
	if (0 == strcmp(new_part_path, old_part_path)) {
		free(old_part_path);
		return EINVAL;
	}

	sprintf(tmp_path, "%s/%s", new_part_path, sos_part_name(part));
	pthread_mutex_unlock(&sos->lock);
	ods_lock(sos->part_ods, 0, NULL);

	if (part->obj_ods)
		ods_commit(part->obj_ods, SOS_COMMIT_SYNC);

	/* Check to see if this location is already in use. */
	rc = stat(tmp_path, &sb);
	if (rc == 0 || (rc && errno != ENOENT)) {
		rc = EEXIST;
		goto out;
	}

	cur_state = SOS_PART(part->part_obj)->state;
	if (cur_state == SOS_PART_STATE_PRIMARY) {
		rc = EBUSY;
		goto out;
	}
	/* Create the new partition path */
	rc = __sos_make_all_dir(tmp_path, part->sos->o_mode);
	if (rc)
		goto out;

	/* Copy the PG file */
	sprintf(tmp_path, "%s/%s/objects.PG", old_part_path, sos_part_name(part));
	in_fd = open(tmp_path, O_RDONLY);
	if (in_fd < 0) {
		rc = errno;
		goto out_1;
	}
	rc = fstat(in_fd, &sb);
	if (rc)
		goto out_1;
	sprintf(tmp_path, "%s/%s/objects.PG", new_part_path, sos_part_name(part));
	out_fd = open(tmp_path, O_RDWR | O_CREAT, sos->o_mode);
	if (out_fd < 0) {
		rc = errno;
		goto out_1;
	}
	offset = 0;
	sz = sendfile(out_fd, in_fd, &offset, sb.st_size);
	if (sz < 0) {
		rc = errno;
		goto out_2;
	}
	close(in_fd);
	close(out_fd);
	out_fd = -1;

	/* Copy the OBJ file */
	sprintf(tmp_path, "%s/%s/objects.OBJ", old_part_path, sos_part_name(part));
	in_fd = open(tmp_path, O_RDONLY);
	if (in_fd < 0) {
		rc = errno;
		goto out_2;
	}
	rc = fstat(in_fd, &sb);
	if (rc)
		goto out_2;
	sprintf(tmp_path, "%s/%s/objects.OBJ", new_part_path, sos_part_name(part));
	out_fd = open(tmp_path, O_RDWR | O_CREAT, sos->o_mode);
	if (out_fd < 0) {
		rc = errno;
		goto out_2;
	}
	offset = 0;
	sz = sendfile(out_fd, in_fd, &offset, sb.st_size);
	if (sz < 0) {
		rc = errno;
		goto out_3;
	}
	close(in_fd);
	close(out_fd);
	strcpy(SOS_PART(part->part_obj)->path, new_part_path);
	if (part->obj_ods)
		ods_close(part->obj_ods, SOS_COMMIT_ASYNC);
	rc = __sos_open_partition(part->sos, part);

	goto out;

 out_3:
	close(out_fd);
	out_fd = -1;
	sprintf(tmp_path, "%s/%s/objects.OBJ", new_part_path, sos_part_name(part));
	unlink(tmp_path);
 out_2:
	close(out_fd);
	sprintf(tmp_path, "%s/%s/objects.PG", new_part_path, sos_part_name(part));
	unlink(tmp_path);
 out_1:
	sprintf(tmp_path, "%s/%s", new_part_path, sos_part_name(part));
	rmdir(tmp_path);
	close(in_fd);
 out:
	if (!rc) {
		/* Remove the original files */
		sprintf(tmp_path, "%s/%s/objects.PG", old_part_path, sos_part_name(part));
		rc = remove(tmp_path);
		if (rc)
			perror("Removing objects.PG file");
		sprintf(tmp_path, "%s/%s/objects.OBJ", old_part_path, sos_part_name(part));
		rc = remove(tmp_path);
		if (rc)
			perror("Removing objects.OBJ file");
		sprintf(tmp_path, "%s/%s", old_part_path, sos_part_name(part));
		rc = remove(tmp_path);
		if (rc)
			perror("Removing partition directory");
	}
	free(old_part_path);
	ods_unlock(sos->part_ods, 0);
	pthread_mutex_unlock(&sos->lock);
	sos_part_put(part);
	return rc;
}

/**
 * \brief Return size and access data for the partition
 *
 * \param part The partition handle
 * \param stat The stat buffer
 * \retval 0 Success
 * \retval EINVAL The partition or stat buffer handles were NULL or
 * the partition is not open, for example, OFFLINE.
 */
int sos_part_stat(sos_part_t part, sos_part_stat_t stat)
{
	int rc = 0;
	struct stat sb;

	if (!part || !stat || !part->obj_ods)
		return EINVAL;
	rc = ods_stat(part->obj_ods, &sb);
	if (rc)
		goto out;

	stat->size = sb.st_size;
	stat->modified = sb.st_mtime;
	stat->accessed = sb.st_atime;
	stat->changed = sb.st_ctime;
 out:
	return rc;
}

struct part_obj_iter_args_s {
	sos_part_t part;
	sos_part_obj_iter_fn_t fn;
	void *arg;
};

static int __part_obj_iter_cb(ods_t ods, ods_obj_t obj, void *arg)
{
	struct part_obj_iter_args_s *oi_args = arg;
	sos_obj_ref_t ref;
	sos_obj_t sos_obj;
	sos_part_t part = oi_args->part;
	sos_obj_data_t sos_obj_data = obj->as.ptr;
	sos_schema_t schema = sos_schema_by_id(part->sos, sos_obj_data->schema);
	if (!schema) {
		sos_warn("Object at %p is missing a valid schema id.\n", ods_obj_ref(obj));
		/* This is a garbage object that should not be here */
		return 0;
	}
	ref.ref.ods = SOS_PART(part->part_obj)->part_id;
	ref.ref.obj = ods_obj_ref(obj);
	sos_obj = __sos_init_obj(part->sos, schema, obj, ref);
	return oi_args->fn(oi_args->part, sos_obj, oi_args->arg);
}

void sos_part_obj_iter_pos_init(sos_part_obj_iter_pos_t pos)
{
	ods_obj_iter_pos_init(&pos->pos);
}

/**
 * \brief Iterate over objects in the partition
 *
 * This function iterates over objects allocated in the partition and
 * calls the specified 'iter_fn' for each object. See the
 * sos_part_obj_iter_fn_t() for the function definition.
 *
 * If the the <tt>pos</tt> argument is not NULL, it should be
 * initialized with the sos_obj_iter_pos_init() funuction. The
 * <tt>pos</tt> argument will updated with the location of the next
 * object in the store when sos_part_obj_iter() returns. This facilitates
 * walking through a portion of the objects at a time, continuing
 * later where the function left off.
 *
 * The sos_part_obj_iter_fn_t() function indicates that the iteration
 * should stop by returning !0. Otherwise, the sos_part_obj_iter() function
 * will continue until all objects in the ODS have been seen.
 *
 * \param ods		The ODS handle
 * \param pos		The object iterator position
 * \param iter_fn	Pointer to the function to call
 * \param arg		A void* argument that the user wants passed to
 *			the callback function.
 * \retval 0		All objects were iterated through
 * \retval !0		A callback returned !0
 */
int sos_part_obj_iter(sos_part_t part, sos_part_obj_iter_pos_t pos,
		      sos_part_obj_iter_fn_t fn, void *arg)
{
	if (!part->obj_ods)
		return 0;
	struct part_obj_iter_args_s args;
	ods_obj_iter_pos_t ods_pos;
	if (pos)
		ods_pos = &pos->pos;
	else
		ods_pos = NULL;
	args.part = part;
	args.fn = fn;
	args.arg = arg;
	return ods_obj_iter(part->obj_ods, ods_pos, __part_obj_iter_cb, &args);
}
