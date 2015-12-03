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
 * \page partitions Partitions
 * \section partitions Partitions
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
 * - \b Offline  The contents of the Partition are not accessible.
 * - \b Moving   The contents of the Partition are being moved to
 *               another storage location.
 *
 * In order to store data, a Container must have at least one
 * partition. There are several commands available to manipulate
 * partitions
 *
 * - \ref sos_part_create Create a new partition
 * - \ref sos_part_query Query the partitions in a container
 * - \ref sos_part_modify Modify the state of a partition
 * - \ref sos_part_move Move a partition to another storage location
 * - \ref sos_part_delete Destroy a partition
 *
 * A typical use case for partitions is to keep hot content on one
 * storage element and colder storage on another. Suppose the
 * administrator wishes to keep 5 days worth of hot data. Partitions
 * provide a mechanism to achieve this goal.  First, create 5
 * partitions to contain the data, all of these are initially on 'hot'
 * storage:
 *
 *       sos_part_create -C MyContainer -s primary "2015-03-03"
 *       sos_part_create -C MyContainer "2015-03-04"
 *       sos_part_create -C MyContainer "2015-03-05"
 *       sos_part_create -C MyContainer "2015-03-06"
 *       sos_part_create -C MyContainer "2015-03-07"
 *
 * At the end of every day, we want to move the older data to a slower
 * storage tier. For example:
 *
 *       sos_part_move -C MyContainer -p /slow-boat "2015-03-03"
 *
 * When the command completes, the objects in 2015-03-03 are now
 * located on the /slow-boat storage element. They are, however, still
 * present in the indexes of the container.
 *
 * When a Partition's data is no longer needed, it may be deleted
 * as follows:
 *
 *      sos_part_delete -C MyContainer -s offline "2015-03-03"
 *      sos_part_delete -C MyContainer "2015-03-03"
 *
 * The list of Partitions defined in a Container can be queried as
 * follows:
 *
 *      tom@css:/SOS/import$ sos_part_query /NVME/0/SOS_ROOT/Test
 *      Partition Name       RefCount Status           Size     Modified         Accessed
 *      -------------------- -------- ---------------- -------- ---------------- ----------------
 *      00000000                    3 ONLINE                 1M 2015/08/25 13:49 2015/08/25 13:51
 *      00000001                    3 ONLINE                 2M 2015/08/25 11:54 2015/08/25 13:51
 *      00000002                    3 ONLINE                 2M 2015/08/25 11:39 2015/08/25 13:51
 *      00000003                    3 ONLINE PRIMARY         2M 2015/08/25 11:39 2015/08/25 13:51
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
 * - sos_part_state() Return the state of a parittion
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

/*
 * New partition objects show up in the address space through:
 * __sos_part_create(), __sos_part_data_first(), and
 * __sos_part_data_next(). Each of these functions takes a reference
 * on the object on behalf of the caller.
 */
static ods_obj_t __sos_part_create(sos_t sos, char *tmp_path,
				   const char *part_name, const char *part_path)
{
	int rc;
	struct stat sb;
	ods_ref_t head_ref, tail_ref;
	ods_obj_t part, new_part;

	/* Take the partition lock */
	ods_spin_lock(&sos->part_lock, -1);

	/* See if the specified name already exists in the filesystem */
	if (part_path == NULL)
		part_path = sos->path;
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
	ods_spin_unlock(&sos->part_lock);
	return new_part;
 err_2:
	ods_obj_delete(new_part);
	ods_obj_put(new_part);
 err_1:
	ods_spin_unlock(&sos->part_lock);
	return NULL;
}

static int __sos_open_partition(sos_t sos, sos_part_t part)
{
	char tmp_path[PATH_MAX];
	int rc;
	ods_t ods;

	sprintf(tmp_path, "%s/%s", sos_part_path(part), sos_part_name(part));
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

void __make_part_offline(sos_t sos, sos_part_t part)
{
	uint64_t part_id;
	sos_container_index_iter_t idx_iter;
	sos_index_t idx;
	sos_iter_t obj_iter;
	int rc;

	/*
	 * Remove the partition from the open SOS instance
	 */
	// TAILQ_REMOVE(&sos->part_list, part, entry);
	SOS_PART(part->part_obj)->state = SOS_PART_STATE_OFFLINE;
	ods_atomic_inc(&SOS_PART_UDATA(sos->part_udata)->gen);
	/*
	 * Remove all keys that refer to objects in
	 * this partition from all indices
	 */
	part_id = sos_part_id(part);
	idx_iter = sos_container_index_iter_new(part->sos);
	for (idx = sos_container_index_iter_first(idx_iter); idx;
	     idx = sos_container_index_iter_next(idx_iter)) {

		obj_iter = sos_index_iter_new(idx);
		for (rc = sos_iter_begin(obj_iter); !rc;
		     rc = sos_iter_next(obj_iter)) {

			sos_obj_ref_t ref = sos_iter_ref(obj_iter);
			while (ref.ref.ods == part_id) {
				rc = sos_iter_entry_remove(obj_iter);
				if (rc)
					break;
				ref = sos_iter_ref(obj_iter);
			}
		}
		sos_index_close(idx, SOS_COMMIT_ASYNC);
	}
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
		assert(cur_primary);
		SOS_PART(cur_primary)->state = SOS_PART_STATE_ACTIVE;
		ods_obj_put(cur_primary);
	}
	/* Make part_obj primary */
	SOS_PART(part->part_obj)->state = SOS_PART_STATE_PRIMARY;
	ods_atomic_inc(&SOS_PART_UDATA(sos->part_udata)->gen);
	SOS_PART_UDATA(sos->part_udata)->primary = ods_obj_ref(part->part_obj);
	sos->primary_part = part;
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
 * \param state The desired state of the partition
 * \retval 0 The state was successfully changed
 * \retval EINVAL The specified state is invalid given the current
 * state of the partition.
 */
int sos_part_state_set(sos_part_t part, sos_part_state_t state)
{
	sos_t sos = part->sos;
	int rc = 0;
	sos_part_state_t cur_state;

	pthread_mutex_unlock(&sos->lock);
	ods_spin_lock(&sos->part_lock, -1);
	cur_state = SOS_PART(part->part_obj)->state;

	switch (cur_state) {
	case SOS_PART_STATE_OFFLINE:
		switch (state) {
		case SOS_PART_STATE_OFFLINE:
			break;
		case SOS_PART_STATE_ACTIVE:
			__make_part_active(sos, part);
			break;
		case SOS_PART_STATE_PRIMARY:
			__make_part_primary(sos, part);
			break;
		case SOS_PART_STATE_MOVING:
			rc = EINVAL;
			break;
		default:
			rc = EINVAL;
			break;
		}
		break;
	case SOS_PART_STATE_ACTIVE:
		switch (state) {
		case SOS_PART_STATE_OFFLINE:
			__make_part_offline(sos, part);
			break;
		case SOS_PART_STATE_ACTIVE:
			break;
		case SOS_PART_STATE_PRIMARY:
			__make_part_primary(sos, part);
			break;
		case SOS_PART_STATE_MOVING:
			rc = EINVAL;
			break;
		default:
			rc = EINVAL;
			break;
		}
		break;
	case SOS_PART_STATE_PRIMARY:
		switch (state) {
		case SOS_PART_STATE_OFFLINE:
			rc = EBUSY;
			break;
		case SOS_PART_STATE_ACTIVE:
			rc = EBUSY;
			break;
		case SOS_PART_STATE_PRIMARY:
			break;
		case SOS_PART_STATE_MOVING:
			rc = EBUSY;
			break;
		default:
			rc = EINVAL;
			break;
		}
		break;
	case SOS_PART_STATE_MOVING:
		rc = EBUSY;
		break;
	}
	ods_spin_unlock(&sos->part_lock);
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
	ods_spin_lock(&sos->part_lock, -1);
	rc = __refresh_part_list(sos);
	ods_spin_unlock(&sos->part_lock);
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
	ods_spin_init(&sos->part_lock, &(SOS_PART_UDATA(sos->part_udata)->lock));
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
	ods_spin_lock(&sos->part_lock, -1);
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
	ods_spin_unlock(&sos->part_lock);
	pthread_mutex_unlock(&sos->lock);
	return part;
}

/**
 * \brief Drop a reference on a partition
 *
 * Partitions are reference counted. When the reference count goes to
 * zero, it is destroyed and all of it's storage is released. The
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
		assert(SOS_PART(part_obj)->state == SOS_PART_STATE_OFFLINE);
		__sos_part_obj_delete(sos, part_obj);
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
	int rc;

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
	rc = __sos_open_partition(sos, part);
	pthread_mutex_unlock(&sos->lock);
	return rc;
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
 out:
	sos_part_iter_free(iter);
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
	sos_part_t part = NULL;
	pthread_mutex_lock(&iter->sos->lock);
	part = TAILQ_FIRST(&iter->sos->part_list);
	if (iter->part)
		sos_part_put(iter->part); /* drop iterator ref */
	iter->part = part;
	if (part) {
		ods_atomic_inc(&part->ref_count); /* iterator reference */
		ods_atomic_inc(&part->ref_count); /* application reference */
	}
	pthread_mutex_unlock(&iter->sos->lock);
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
	if (!iter->part)
		return NULL;
	pthread_mutex_lock(&iter->sos->lock);
	part = TAILQ_NEXT(iter->part, entry);
	sos_part_put(iter->part);	  /* drop iterator reference */
	iter->part = part;
	if (part) {
		ods_atomic_inc(&part->ref_count); /* iterator reference */
		ods_atomic_inc(&part->ref_count); /* application reference */
	}
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
	ods_spin_lock(&sos->part_lock, -1);
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
	ods_spin_unlock(&sos->part_lock);
	pthread_mutex_unlock(&sos->lock);
	return rc;
}

/**
 * \brief Move a partition
 *
 * Move a partition from it's current storage location to another. Any
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
int sos_part_move(sos_part_t part, const char *part_path)
{
	char tmp_path[PATH_MAX];
	off_t offset;
	ssize_t sz;
	int in_fd, out_fd;
	struct stat sb;
	int rc;
	sos_t sos = part->sos;
	sos_part_state_t cur_state;

	if (0 == strcmp(part_path, sos_part_path(part)))
		return EINVAL;

	sprintf(tmp_path, "%s/%s", part_path, sos_part_name(part));
	pthread_mutex_unlock(&sos->lock);
	ods_spin_lock(&sos->part_lock, -1);

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
	sprintf(tmp_path, "%s/%s/objects.PG", sos_part_path(part), sos_part_name(part));
	in_fd = open(tmp_path, O_RDONLY);
	if (in_fd < 0) {
		rc = errno;
		goto out;
	}
	rc = fstat(in_fd, &sb);
	sprintf(tmp_path, "%s/%s/objects.PG", part_path, sos_part_name(part));
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
	sprintf(tmp_path, "%s/%s/objects.OBJ", sos_part_path(part), sos_part_name(part));
	in_fd = open(tmp_path, O_RDONLY);
	if (in_fd < 0) {
		rc = errno;
		goto out_2;
	}
	rc = fstat(in_fd, &sb);
	sprintf(tmp_path, "%s/%s/objects.OBJ", part_path, sos_part_name(part));
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
	strcpy(SOS_PART(part->part_obj)->path, part_path);
	ods_close(part->obj_ods, SOS_COMMIT_ASYNC);
	rc = __sos_open_partition(part->sos, part);
	goto out;

 out_3:
	close(out_fd);
	out_fd = -1;
	sprintf(tmp_path, "%s/%s/objects.OBJ", part_path, sos_part_name(part));
	unlink(tmp_path);
 out_2:
	close(out_fd);
	sprintf(tmp_path, "%s/%s/objects.PG", part_path, sos_part_name(part));
	unlink(tmp_path);
 out_1:
	sprintf(tmp_path, "%s/%s", part_path, sos_part_name(part));
	rmdir(tmp_path);
	close(in_fd);
 out:
	ods_spin_unlock(&sos->part_lock);
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
	stat->created = sb.st_ctime;
 out:
	return rc;
}
