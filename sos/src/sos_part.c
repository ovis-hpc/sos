/*
 * Copyright (c) 2021 Open Grid Computing, Inc. All rights reserved.
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
 * - \b Busy     The Partition is being updated and cannot be changed.
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
 * 'primary' state in order for objects to be allocated and stored. A
 * partition is created with the sos_part_create command.
 * For example:
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
 * date and then migrate older partitions to another container on
 * secondary storage.
 *
 * At midnight the administrator starts storing data in tomorrow's partition as follows:readdir
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
 * - sos_part_delete() Delete a partition
 * - sos_part_move() Move a parition to another storage location
 * - sos_part_copy() Copy a partition to another storage location
 * - sos_part_iter_new() Create a partition iterator
 * - sos_part_iter_free() Free a partition iterator
 * - sos_part_first() Return the first partition in the Container
 * - sos_part_next() Return the next partition in the Container
 * - sos_part_by_name() Find a partition by name
 * - sos_part_by_path() Find a partition by path
 * - sos_part_by_uuid() Find a partition by uuid
 * - sos_part_put() Drop a reference on the partition
 * - sos_part_stat() Return size and access information about a partition
 * - sos_part_state_set() Set the state of a parittion
 * - sos_part_name() Return the name of a partition
 * - sos_part_path() Return a partitions storage path
 */
#define _GNU_SOURCE
#include <sys/queue.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/sendfile.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <dirent.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdarg.h>
#include <libgen.h>
#include <limits.h>
#include <errno.h>
#include <assert.h>
#include <uuid/uuid.h>

#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

static sos_part_t __sos_part_by_name(sos_t sos, const char *name);
static sos_part_t __sos_part_by_path(sos_t sos, const char *path);
static sos_part_t __sos_part_first(sos_part_iter_t iter);
static sos_part_t __sos_part_next(sos_part_iter_t iter);
static int __refresh_part_list(sos_t sos);
static int __sos_remove_directory(const char *dir);
static void __sos_part_free(void *free_arg)
{
	sos_part_t part = free_arg;
	ods_obj_put(part->udata_obj);
	ods_obj_put(part->ref_obj);
	if (part->obj_ods)
		ods_close(part->obj_ods, ODS_COMMIT_ASYNC);
	free(part);
}
/**
 * \brief Create a new partition
 *
 * This interface is deprecated. New software should use sos_part_open()
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
	sos_part_t part;

	/* See if there is already a partition by this name */
	part = sos_part_by_name(sos, part_name);
	if (part) {
		sos_part_put(part);
		return EEXIST;
	}

	snprintf(tmp_path, sizeof(tmp_path), "%s/%s", part_path, part_name);
	/* open/create the new partition */
	part = sos_part_open(tmp_path, sos->o_perm | SOS_PERM_CREAT, sos->o_mode, "");
	if (!part)
		return errno;
	sos_part_put(part);

	/* attach the partition to the container */
	return sos_part_attach(sos, part_name, tmp_path);
}

int __sos_part_create(const char *part_path, const char *part_desc,
			sos_perm_t o_perm, int o_mode)
{
	uuid_t uuid;
	char tmp_path[PATH_MAX];
	int rc;
	ods_t ods;
	struct stat sb;

	rc = stat(part_path, &sb);
	if (rc == 0)
		return EEXIST;
	rc = __sos_make_all_dir(part_path, o_mode
				| (o_mode & S_IRUSR ? S_IXUSR : 0)
				| (o_mode & S_IRGRP ? S_IXGRP : 0));
	if (rc)
		return errno;
	sprintf(tmp_path, "%s/objects", part_path);
	ods = ods_open(tmp_path, o_perm | ODS_PERM_RW | ODS_PERM_CREAT, o_mode);
	if (!ods) {
		rc = errno;
		goto err_0;
	}
	ods_obj_t udata = ods_get_user_data(ods);
	uuid_generate(SOS_PART_UDATA(udata)->uuid);
	SOS_PART_UDATA(udata)->signature = SOS_PART_SIGNATURE;
	strncpy(SOS_PART_UDATA(udata)->desc, part_desc, SOS_PART_DESC_LEN);
	uuid_generate(uuid);
	uuid_copy(SOS_PART_UDATA(udata)->uuid, uuid);
	ods_obj_update(udata);
	ods_obj_put(udata);
	ods_close(ods, ODS_COMMIT_SYNC);
	return 0;
 err_0:
	if (errno != EEXIST && errno != EBUSY)
		(void)__sos_remove_directory(part_path);
	return rc;
}

static int __part_open(sos_part_t part, const char *path, sos_perm_t o_perm)
{
	char tmp_path[PATH_MAX];
	ods_t ods;
	int rc;

	assert(0 == (o_perm & SOS_PERM_CREAT));
	snprintf(tmp_path, sizeof(tmp_path), "%s/objects", path);
	ods = ods_open(tmp_path, o_perm);
	if (!ods)
		return errno;

	ods_obj_t part_obj = ods_get_user_data(ods);
	if (!part_obj) {
		rc = errno;
		goto err_0;
	}

	if (SOS_PART_UDATA(part_obj)->signature != SOS_PART_SIGNATURE) {
		rc = EINVAL;
		goto err_1;
	}

	part->udata_obj = part_obj;
	part->obj_ods = ods;
	return 0;
err_1:
	ods_obj_put(part_obj);
	ods_close(ods, ODS_COMMIT_ASYNC);
err_0:
	return rc;
}

static void __part_close(sos_part_t part)
{
	ods_obj_put(part->udata_obj);
	part->udata_obj = NULL;
	ods_close(part->obj_ods, ODS_COMMIT_ASYNC);
	part->obj_ods = NULL;
}

static sos_part_t __sos_part_open(sos_t sos, ods_obj_t ref_obj)
{
	int rc;
	sos_part_t part = calloc(1, sizeof *part);
	if (!part)
		return NULL;
	sos_ref_init(&part->ref_count, "part_list", __sos_part_free, part);
	rc = __part_open(part, SOS_PART_REF(ref_obj)->path,
			sos->o_perm & ~SOS_PERM_CREAT);
	if (rc) {
		errno = rc;
		return NULL;
	}
	part->ref_obj = ods_obj_get(ref_obj);
	part->sos = sos;
	TAILQ_INSERT_TAIL(&sos->part_list, part, entry);
	return part;
}

/**
 * @brief Open a partition
 *
 * Open a partition and return the partition handle. The partition is not
 * associated with a container, see sos_part_attach().
 *
 * @param path
 * @param o_perm The SOS access permissions. If SOS_PERM_CREAT is included,
 *               the parameters o_mode and desc are expected
 * @param o_mode The file creation mode, see open(3)
 * @param desc A pointer to a character string description for the partition.
 * @return sos_part_t
 */
sos_part_t sos_part_open(const char *path, int o_perm, ...)
{
	int rc;
	int o_mode;
	char *desc;
	sos_part_t part;
	va_list ap;
	va_start(ap, o_perm);
	o_mode = 0660;

	if (o_perm & SOS_PERM_CREAT) {
		o_mode = va_arg(ap, int);
		desc = va_arg(ap, char *);
		rc = __sos_part_create(path, desc, o_perm, o_mode);
		if (rc) {
			errno = rc;
			return NULL;
		}
		o_perm &= ~SOS_PERM_CREAT;
	}

	part = calloc(1, sizeof *part);
	if (!part)
		return NULL;
	sos_ref_init(&part->ref_count, "application", __sos_part_free, part);
	rc = __part_open(part, path, o_perm);
	if (rc) {
		errno = rc;
		free(part);
		return NULL;
	}
	if (o_perm & SOS_PERM_CREAT)
		ods_commit(part->obj_ods, ODS_COMMIT_SYNC);
	return part;
}

sos_perm_t sos_part_be_get(sos_part_t part)
{
	ods_perm_t be = ods_backend_type_get(part->obj_ods);
	switch (be) {
	case ODS_BE_MMAP:
		return SOS_BE_MMOS;
	case ODS_BE_LSOS:
		return SOS_BE_LSOS;
	default:
		return SOS_BE_MMOS;
	}
}
struct iter_args {
	double start;
	double timeout;
	sos_part_t part;
	uint64_t count;
};

void __make_part_offline(sos_t sos, sos_part_t part)
{
	SOS_PART_REF(part->ref_obj)->state = SOS_PART_STATE_OFFLINE;
	__sos_schema_reset(part->sos);
}

static void __make_part_active(sos_t sos, sos_part_t part)
{
	SOS_PART_REF(part->ref_obj)->state = SOS_PART_STATE_ACTIVE;
	__sos_schema_reset(part->sos);
}

static int __make_part_primary(sos_t sos, sos_part_t part)
{
	/* Check if the requested partition is primary in any other container */
	if (SOS_PART_UDATA(part->udata_obj)->is_primary)
		return EBUSY;

	/* Fix-up the current primary */
	if (sos->primary_part) {
		SOS_PART_REF(sos->primary_part->ref_obj)->state = SOS_PART_STATE_ACTIVE;
		SOS_PART_UDATA(sos->primary_part->udata_obj)->is_primary = 0;
		sos_ref_put(&sos->primary_part->ref_count, "primary_part");
	}
	/* Make part_obj primary */
	SOS_PART_REF(part->ref_obj)->state = SOS_PART_STATE_PRIMARY;
	SOS_PART_REF_UDATA(sos->part_ref_udata)->primary = ods_obj_ref(part->udata_obj);
	SOS_PART_UDATA(part->udata_obj)->is_primary = 1;
	sos->primary_part = part;
	sos_ref_get(&sos->primary_part->ref_count, "primary_part");
	__sos_schema_reset(part->sos);
	return 0;
}

sos_part_t __sos_part_find_by_ods(sos_t sos, ods_t ods)
{
	sos_part_t part;
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (ods == part->obj_ods)
			goto out;
	}
	part = NULL;
 out:
	return part;
}

sos_part_t __sos_part_find_by_uuid(sos_t sos, uuid_t uuid)
{
	sos_part_t part;
	sos_part_iter_t iter = sos_part_iter_new(sos);
	if (!iter)
		return NULL;

	for (part = __sos_part_first(iter); part; part = __sos_part_next(iter)) {
		if (0 == uuid_compare(uuid, SOS_PART_UDATA(part->udata_obj)->uuid))
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
 * OFFLINE, all data in that partition is no longer visible
 * in the container.
 *
 * \param part The partition handle
 * \param new_state The desired state of the partition
 * \retval 0 The state was successfully changed
 * \retval EBUSY The partition is being modified in another container
 * \retval EEXIST The partition is PRIMARY in another container
 * \retval EINVAL The specified state is invalid given the current
 *    state of the partition.
 */
int sos_part_state_set(sos_part_t part, sos_part_state_t new_state)
{
	sos_t sos = part->sos;
	int rc = 0;
	sos_part_state_t cur_state;

	switch(new_state) {
	case SOS_PART_STATE_PRIMARY:
	case SOS_PART_STATE_OFFLINE:
	case SOS_PART_STATE_ACTIVE:
		break;
	default:
		return EINVAL;
	}
	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ref_ods, 0, NULL);

	/*
	 * Check if another container is currently fiddling
	 * with this partition.
	 */
	if (SOS_PART_UDATA(part->udata_obj)->is_busy) {
		/* TODO: Add retry logic? */
		rc = EBUSY;
		goto out;
	}
	cur_state = SOS_PART_REF(part->ref_obj)->state;
	/*
	 * If the new state is PRIMARY, the partition in question cannot
	 * be primary in any other container
	 */
	if (new_state == SOS_PART_STATE_PRIMARY
		&& SOS_PART_UDATA(part->udata_obj)->is_primary) {
		rc = EEXIST;
		goto out;
	}
	/*
	 * The PRIMARY partition cannot have it's state changed. It is only
	 * changed by making another parition PRIMARY
	 */
	if (cur_state == SOS_PART_STATE_PRIMARY) {
		rc = EINVAL;
		goto out;
	}
	/*
	 * Current state is new state
	 */
	if (cur_state == new_state)
		goto out;

	rc = 0;
	switch (new_state) {
	case SOS_PART_STATE_PRIMARY:
		rc = __make_part_primary(sos, part);
		break;
	case SOS_PART_STATE_ACTIVE:
		__make_part_active(sos, part);
		break;
	case SOS_PART_STATE_OFFLINE:
		__make_part_offline(sos, part);
		break;
	default:
		assert(0);
		break;
	}
 out:
	SOS_PART_UDATA(part->udata_obj)->is_busy = 0;
	ods_obj_update(part->udata_obj);
	ods_obj_update(part->ref_obj);
 	ods_unlock(sos->part_ref_ods, 0);
	ods_commit(sos->part_ref_ods, ODS_COMMIT_SYNC);
	pthread_mutex_unlock(&sos->lock);
	return rc;
}

ods_obj_t __sos_part_ref_first(sos_t sos)
{
	ods_obj_t part_obj =
		ods_ref_as_obj(sos->part_ref_ods,
			SOS_PART_REF_UDATA(sos->part_ref_udata)->head);
	return part_obj;
}

ods_obj_t __sos_part_ref_next(sos_t sos, ods_obj_t part_obj)
{
	ods_obj_t next_obj;
	ods_ref_t next_ref = 0;

	if (!part_obj)
		return NULL;

	next_ref = SOS_PART_REF(part_obj)->next;
	ods_obj_put(part_obj);
	if (!next_ref)
		return NULL;

	next_obj = ods_ref_as_obj(sos->part_ref_ods, next_ref);
	return next_obj;
}

static int __refresh_part_list(sos_t sos)
{
	int rc = 0;
	sos_part_t part;
	ods_obj_t part_obj;
	int new;

	if (sos->primary_part) {
		sos->primary_part = NULL;
		sos_ref_put(&sos->primary_part->ref_count, "primary_part");
	}
	for (part_obj = __sos_part_ref_first(sos);
	     part_obj; part_obj = __sos_part_ref_next(sos, part_obj)) {
		new = 1;
		/* Check if we already have this partition */
		TAILQ_FOREACH(part, &sos->part_list, entry) {
			if (strcmp(ods_path(part->obj_ods), SOS_PART_REF(part_obj)->path))
				continue;
			new = 0;
		}
		if (!new)
			continue;
		/* This is a partition we have not seen before */
		part = __sos_part_open(sos, part_obj);
		if (!part) {
			rc = ENOMEM;
			goto out;
		}
		if (SOS_PART_REF(part_obj)->state == SOS_PART_STATE_PRIMARY) {
			sos->primary_part = part;
			sos_ref_get(&part->ref_count, "primary_part");
		}
	}
 out:
	if (part_obj)
		ods_obj_put(part_obj);
	return rc;
}

static int refresh_part_list(sos_t sos)
{
	int rc;
	ods_lock(sos->part_ref_ods, 0, NULL);
	rc = __refresh_part_list(sos);
	ods_unlock(sos->part_ref_ods, 0);
	return rc;
}

int __sos_open_partitions(sos_t sos, char *tmp_path)
{
	int rc;

	/* Open the partition ODS */
	sprintf(tmp_path, "%s/.__part", sos->path);
	sos->part_ref_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!sos->part_ref_ods)
		goto err_0;
	sos->part_ref_udata = ods_get_user_data(sos->part_ref_ods);
	if (!sos->part_ref_udata)
		goto err_1;
	rc = refresh_part_list(sos);
	return rc;
 err_1:
	ods_close(sos->part_ref_ods, ODS_COMMIT_ASYNC);
	sos->part_ref_ods = NULL;
 err_0:
	return errno;
}

sos_part_t __sos_primary_obj_part(sos_t sos)
{
	sos_part_t part = NULL;

	if ((NULL != sos->primary_part) &&
	    (SOS_PART_REF(sos->primary_part->ref_obj)->state == SOS_PART_STATE_PRIMARY))
		return sos->primary_part;

	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ref_ods, 0, NULL);
	if (sos->part_gn != SOS_PART_REF_UDATA(sos->part_ref_udata)->gen) {
		int rc = __refresh_part_list(sos);
		if (rc)
			goto out;
	}
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (SOS_PART_REF(part->ref_obj)->state == SOS_PART_STATE_PRIMARY) {
			sos->primary_part = part;
			sos_ref_get(&part->ref_count, "primary_part");
			goto out;
		}
	}
 out:
	ods_unlock(sos->part_ref_ods, 0);
	pthread_mutex_unlock(&sos->lock);
	return part;
}

struct export_obj_iter_args_s {
	sos_t src_sos;
	sos_t dst_sos;
	sos_part_t src_part;
	ods_idx_t exp_idx;
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

sos_part_t _sos_part_get(sos_part_t part, const char *func, int line)
{
	_sos_ref_get(&part->ref_count, "application", func, line);
	return part;
}

/**
 * \brief Drop a reference on a partition
 *
 * Partitions are reference counted. When the reference count goes to
 * zero, it is destroyed and all of its storage is released. The
 * sos_part_first(), sos_part_next(), sos_part_by_name(), and
 * sos_part_by_path() functions ake a reference on behalf of the application.
 * This reference should be dropped by the application when the application is
 * finished with the partition.
 *
 * This function ignores a NULL partition handle.
 *
 * \param part The partition handle.
 */
void sos_part_put(sos_part_t part)
{
	if (part)
		sos_ref_put(&part->ref_count, "application");
}


/**
 * \brief Free the memory associated with the Iterator
 *
 * \param iter The iterator handle
 */
void sos_part_iter_free(sos_part_iter_t iter)
{
	if (!iter)
		return;
	if (iter->part)
		sos_ref_put(&iter->part->ref_count, "iterator");
	free(iter);
}

/**
 * @brief Attach a partition to a container
 *
 * @param sos The sos_t container handle
 * @param name A name for this partition in this container
 * @param path The path to the partition.
 * @retval 0 success
 * @retval EEXIST The partition is already attached
 * @retval ENOENT The partition was not found
 */
int sos_part_attach(sos_t sos, const char *name, const char *path)
{
	sos_part_t part;
	ods_ref_t head_ref, tail_ref;
	ods_obj_t new_part_ref;
	int rc;

	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ref_ods, 0, NULL);

	part = __sos_part_by_name(sos, name);
	if (!part)
		part = __sos_part_by_path(sos, path);
	if (part) {
		rc = EEXIST;
		goto err_0;
	}

	head_ref = SOS_PART_REF_UDATA(sos->part_ref_udata)->head;
	tail_ref = SOS_PART_REF_UDATA(sos->part_ref_udata)->tail;

	new_part_ref = ods_obj_alloc(sos->part_ref_ods, sizeof(struct sos_part_ref_data_s));
	if (!new_part_ref)
		goto err_0;

	/* Make sure we can open it */
	strncpy(SOS_PART_REF(new_part_ref)->path, path, SOS_PART_PATH_LEN);
	strncpy(SOS_PART_REF(new_part_ref)->name, name, SOS_PART_NAME_LEN);
	SOS_PART_REF(new_part_ref)->state = SOS_PART_STATE_OFFLINE;
	part = __sos_part_open(sos, new_part_ref);
	if (!part)
		goto err_1;

	/* Add the partition to our list */
	SOS_PART_REF(new_part_ref)->prev = tail_ref;
	SOS_PART_REF(new_part_ref)->next = 0;

	/* Insert it into the partition list */
	if (tail_ref) {
		ods_obj_t prev = ods_ref_as_obj(sos->part_ref_ods, tail_ref);
		if (!prev)
			goto err_1;
		SOS_PART_REF(prev)->next = ods_obj_ref(new_part_ref);
		SOS_PART_REF_UDATA(sos->part_ref_udata)->tail = ods_obj_ref(new_part_ref);
		if (!head_ref)
			SOS_PART_REF_UDATA(sos->part_ref_udata)->head = ods_obj_ref(new_part_ref);
		ods_obj_update(prev);
		ods_obj_put(prev);
	} else {
		SOS_PART_REF_UDATA(sos->part_ref_udata)->head = ods_obj_ref(new_part_ref);
		SOS_PART_REF_UDATA(sos->part_ref_udata)->tail = ods_obj_ref(new_part_ref);
	}
	ods_obj_update(sos->part_ref_udata);
	ods_obj_update(new_part_ref);
	ods_obj_put(new_part_ref);
	ods_atomic_inc(&SOS_PART_REF_UDATA(sos->part_ref_udata)->gen);
	ods_atomic_inc(&SOS_PART_UDATA(part->udata_obj)->ref_count);
	ods_unlock(sos->part_ref_ods, 0);
	ods_commit(sos->part_ref_ods, ODS_COMMIT_SYNC);
	pthread_mutex_unlock(&sos->lock);
	return 0;
err_1:
	ods_obj_delete(new_part_ref);
	ods_obj_put(new_part_ref);
err_0:
	ods_unlock(sos->part_ref_ods, 0);
	pthread_mutex_unlock(&sos->lock);
	return rc;
}

/**
 * @brief Detach a partition from a container
 *
 * Detaches the partition from the container. The application
 * MUST NOT use the partion handle \c part after calling this
 * function.
 *
 * @param sos The container handle
 * @param path The path to the partition
 * @retval 0 The partition was successfully detached
 * @retval ENOENT The partition is not attached
 */
int sos_part_detach(sos_part_t part)
{
	sos_t sos = part->sos;
	ods_ref_t prev_ref, next_ref;
	int rc = 0;

	if (!part->sos || !part->udata_obj || !part->ref_obj)
		return ENOENT;

	pthread_mutex_lock(&part->sos->lock);
	ods_lock(sos->part_ref_ods, 0, NULL);

	if (sos_part_state(part) == SOS_PART_STATE_PRIMARY) {
		rc = EINVAL;
		goto out;
	}

	/* Remove the partition reference */
	prev_ref = SOS_PART_REF(part->ref_obj)->prev;
	next_ref = SOS_PART_REF(part->ref_obj)->next;

	if (prev_ref) {
		ods_obj_t prev = ods_ref_as_obj(sos->part_ref_ods, prev_ref);
		SOS_PART_REF(prev)->next = next_ref;
		ods_obj_update(prev);
		ods_obj_put(prev);
	} else {
		SOS_PART_REF_UDATA(sos->part_ref_udata)->head = next_ref;
	}
	if (next_ref) {
		ods_obj_t next = ods_ref_as_obj(sos->part_ref_ods, next_ref);
		SOS_PART_REF(next)->prev = prev_ref;
		ods_obj_update(next);
		ods_obj_put(next);
	} else {
		SOS_PART_REF_UDATA(sos->part_ref_udata)->tail = prev_ref;
	}

	/* Remove the sos_part_t from the part_list */
	TAILQ_REMOVE(&sos->part_list, part, entry);
	ods_obj_delete(part->ref_obj);
	ods_atomic_inc(&SOS_PART_REF_UDATA(sos->part_ref_udata)->gen);
	ods_atomic_dec(&SOS_PART_UDATA(part->udata_obj)->ref_count);
	ods_obj_update(sos->part_ref_udata);
	__part_close(part);
	sos_ref_put(&part->ref_count, "application");	/* obtained by ...find */
	sos_ref_put(&part->ref_count, "part_list");
out:
	ods_unlock(sos->part_ref_ods, 0);
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

static sos_part_t __sos_part_by_name(sos_t sos, const char *name)
{
	sos_part_t part;
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (0 == strcmp(sos_part_name(part), name))
			goto out;
	}
	part = NULL;
 out:
	return part;
}

/**
 * \brief Find a partition by name
 *
 * Returns the partition handle for the partition with the name
 * specified by the \c name parameter.
 *
 * The application should call sos_part_put() when finished with the
 * partition object.
 *
 * \param sos The container handle
 * \param name The name of the partition
 * \retval Partition handle
 * \retval NULL if the partition was not found
 */
sos_part_t sos_part_by_name(sos_t sos, const char *name)
{
	sos_part_t part;
	pthread_mutex_lock(&sos->lock);
	part = __sos_part_by_name(sos, name);
	if (part)
		sos_part_get(part);
	pthread_mutex_unlock(&sos->lock);
	return part;
}

/**
 * \brief Find a partition by name
 *
 * This interface is deprecated. New software should use
 * sos_part_by_name()
 */
sos_part_t sos_part_find(sos_t sos, const char *name)
{
	return sos_part_by_name(sos, name);
}

static sos_part_t __sos_part_by_path(sos_t sos, const char *path)
{
	sos_part_t part;
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (0 == strcmp(sos_part_path(part), path))
			goto out;
	}
	part = NULL;
 out:
	return part;
}

/**
 * \brief Find a partition by path
 *
 * Returns the partition handle for the partition with the name
 * specified by the \c path parameter.
 *
 * The application should call sos_part_put() when finished with the
 * partition object.
 *
 * \param sos The container handle
 * \param path The name of the partition
 * \retval Partition handle
 * \retval NULL if the partition was not found
 */
sos_part_t sos_part_by_path(sos_t sos, const char *path)
{
	sos_part_t part;
	pthread_mutex_lock(&sos->lock);
	part = __sos_part_by_path(sos, path);
	if (part)
		sos_part_get(part);
	pthread_mutex_unlock(&sos->lock);
	return part;
}

static sos_part_t __sos_part_first(sos_part_iter_t iter)
{
	sos_part_t part = NULL;
	part = TAILQ_FIRST(&iter->sos->part_list);
	if (iter->part) {
		sos_ref_put(&iter->part->ref_count, "iterator");
	}
	iter->part = part;
	if (part) {
		sos_ref_get(&iter->part->ref_count, "iterator");
		sos_ref_get(&iter->part->ref_count, "application");
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
	if (iter->part)
		sos_ref_put(&iter->part->ref_count, "iterator");	  /* drop iterator reference */
	iter->part = part;
	if (part) {
		sos_ref_get(&part->ref_count, "iterator"); /* iterator reference */
		sos_ref_get(&part->ref_count, "application"); /* application reference */
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
	if (part->ref_obj)
		return SOS_PART_REF(part->ref_obj)->name;
	return "";
}
/**
 * @brief Return the number of containers attached to this partition
 *
 * @param part
 * @return The count of containers attached to this partition
 */
uint32_t sos_part_refcount(sos_part_t part)
{
	return SOS_PART_UDATA(part->udata_obj)->ref_count;
}
/**
 * @brief Return the UUID for the partition
 *
 * Returns the universal unique identifer (UUID) in the \c uuid
 * parameter
 *
 * @param part The partition handle
 * @param uuid The uuid_t to receive the value
 */
void sos_part_uuid(sos_part_t part, uuid_t uuid)
{
	uuid_copy(uuid, SOS_PART_UDATA(part->udata_obj)->uuid);
}
/**
 * \brief Return the partition's path
 * \param part The partition handle
 * \returns Pointer to a string containing the path
 */
const char *sos_part_path(sos_part_t part)
{
	if (part->ref_obj)
		return SOS_PART_REF(part->ref_obj)->path;
	return "";
}
/**
 * \brief Return the partition's description data
 * \param part The partition handle
 * \returns Pointer to a string containing the description
 */
const char *sos_part_desc(sos_part_t part)
{
	return SOS_PART_UDATA(part->udata_obj)->desc;
}
/**
 * @brief Return the partition reference count
 *
 * Returns the number of containers that are attached to this
 * partition
 *
 * @param path The path to the partition
 * @retval The partition reference count
 */
int sos_part_ref_count(sos_part_t part)
{
	return SOS_PART_UDATA(part->udata_obj)->ref_count;
}
/**
 * \brief Return the partition's state
 * \param part The partition handle
 * \returns An integer representing the partition's state
 */
sos_part_state_t sos_part_state(sos_part_t part)
{
	return SOS_PART_REF(part->ref_obj)->state;
}

/**
 * @brief Delete a partition
 *
 * Deletes the paritition at the specified path. All object storage
 * associated with the parition will be freed. The parition must be in
 * the OFFLINE state with only a single container reference to be deleted.
 *
 * @param path The path to the partition
 * @retval 0 The parition was deleted
 * @retval EBUSY The partition is not offline or there are outstanding
 *  	container references
 */
int sos_part_destroy(char *path)
{
	int rc;
	sos_part_t part = calloc(1, sizeof *part);
	if (!part)
		return errno;
	rc = __part_open(part, path, SOS_PERM_RW);
	if (rc) {
		goto err_0;
		return rc;
	}
	if (SOS_PART_UDATA(part->udata_obj)->ref_count > 0) {
		rc = EBUSY;
		goto err_1;
	}
	__part_close(part);
	free(part);
	rc =__sos_remove_directory(path);
	return rc;
err_1:
	__part_close(part);
err_0:
	free(part);
	return rc;
}

/* Shallow copy a directory and its contents */
static int __sos_copy_directory(const char *src_dir, const char *dst_dir)
{
	char path[PATH_MAX];
	struct dirent **namelist;
	int n, cnt, rc, in_fd, out_fd;
	struct stat sb;
	loff_t offset;

	rc = stat(src_dir, &sb);
	if (rc) {
		sos_error("Error %d stating the source directory '%s'\n",
				errno, src_dir);
		return errno;
	}
	rc = __sos_make_all_dir(dst_dir, sb.st_mode);
	if (rc) {
		sos_error("Error %d creating the destinatio path '%s'\n",
				errno, dst_dir);
		return errno;
	}
	n = scandir(src_dir, &namelist, NULL, alphasort);
	if (n == 0) {
		sos_info("The source directory '%s' is empty.\n", src_dir);
		return 0;
	}
	for (cnt = 0; cnt < n; ++cnt) {
		/* Ignore . and .. */
		if (0 == strcmp(namelist[cnt]->d_name, "."))
			continue;
		if (0 == strcmp(namelist[cnt]->d_name, ".."))
			continue;
		snprintf(path, sizeof(path), "%s/%s", src_dir, namelist[cnt]->d_name);
		in_fd = open(path, O_RDONLY);
		if (in_fd < 0) {
			rc = errno;
			sos_error("Error %d opening the source file '%s'\n", errno,
					path);
			continue;
		}
		rc = fstat(in_fd, &sb);
		if (rc) {
			sos_error("Error %d stat-int the source file '%s'\n", path);
			close(in_fd);
			continue;
		}
		snprintf(path, sizeof(path), "%s/%s", dst_dir, namelist[cnt]->d_name);
		out_fd = open(path, O_RDWR | O_CREAT, sb.st_mode);
		if (out_fd < 0) {
			sos_error("Error %d creating the destination file '%s'\n",
					errno, path);
			close(in_fd);
			continue;
		}
		offset = 0;
		size_t sz = sendfile(out_fd, in_fd, &offset, sb.st_size);
		if (sz < 0) {
			sos_error("Error %d copying the file '%s'\n",
					errno, namelist[cnt]->d_name);
		}
		close(in_fd);
		close(out_fd);
	}
	return 0;
}

/* Shallow delete a directory and its contents */
static int __sos_remove_directory(const char *dir)
{
	struct dirent **namelist;
	char path[PATH_MAX];
	int n, rc;

	n = scandir (dir, &namelist, NULL, alphasort);
	if (n >= 0) {
		int cnt;
		for (cnt = 0; cnt < n; ++cnt) {
			/* Ignore . and .. */
			if (0 == strcmp(namelist[cnt]->d_name, "."))
				continue;
			if (0 == strcmp(namelist[cnt]->d_name, ".."))
				continue;
			snprintf(path, sizeof(path), "%s/%s", dir, namelist[cnt]->d_name);
			rc = unlink(path);
			if (rc) {
				sos_error("Error %d removing the file '%s'\n",
						errno, namelist[cnt]->d_name);
			}
		}
	} else {
		return errno;
	}
	rc = rmdir(dir);
	if (rc) {
		sos_error("Error %d removing the directory '%s'\n",
				errno, dir);
		return rc;
	}
	return 0;
}

/**
 * \brief Move a partition
 *
 * Move an *offline* partition from its current storage location to
 * another location.
 *
 * \param part The partition handle
 * \param path The new path for the partition
 * \retval 0 The partition was successfully moved
 * \retval EBUSY The partition is not offline or another container is modifying it
 * \retval EINVAL The destination path is the same as the source path
 * \retval EEXIST The destination path already exists
 * \retval EPERM The user has insufficient privilege to write to the destination path
 */
int sos_part_move(sos_part_t part, const char *dest_path)
{
	sos_t sos = part->sos;
	struct stat sb;
	int rc;

	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ref_ods, 0, NULL);

	if (sos_part_state(part) != SOS_PART_STATE_OFFLINE) {
		rc = EBUSY;
		goto out;
	}

	if (SOS_PART_UDATA(part->udata_obj)->is_busy) {
		rc = EBUSY;
		goto out;
	}

	rc = stat(dest_path, &sb);
	if (rc == 0 || (rc && errno != ENOENT)) {
		sos_error("The destination directory '%s' already exists.\n",
			dest_path);
		goto out;
	}
	rc = __sos_copy_directory(sos_part_path(part), dest_path);
	if (rc)
		goto out;

	rc = __sos_remove_directory(sos_part_path(part));
	if (rc) {
		sos_error("Error %d removing source directory '%s'\n",
			rc, sos_part_path(part));
	}
	__part_close(part);
	rc = __part_open(part, dest_path, sos->o_perm & ~SOS_PERM_CREAT);
	if (!rc)
		strcpy(SOS_PART_REF(part->ref_obj)->path, dest_path);
	ods_obj_update(part->ref_obj);
out:
	ods_unlock(sos->part_ref_ods, 0);
	pthread_mutex_unlock(&sos->lock);
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

static int __part_obj_iter_cb(ods_t ods, ods_ref_t obj_ref, void *arg)
{
	ods_obj_t ods_obj = ods_ref_as_obj(ods, obj_ref);
	if (!ods_obj) {
		sos_error("Object reference %p could not be instantiated as a partition\n", (void *)obj_ref);
		return 0;
	}
	struct part_obj_iter_args_s *oi_args = arg;
	sos_obj_ref_t ref;
	sos_obj_t sos_obj;
	sos_part_t part = oi_args->part;
	sos_obj_data_t sos_obj_data = ods_obj->as.ptr;
	sos_schema_t schema = sos_schema_by_uuid(part->sos, sos_obj_data->schema_uuid);
	if (!schema) {
		sos_warn("Object at %p is missing a valid schema id.\n", (void *)obj_ref);
		/* This is a garbage object that should not be here */
		return 0;
	}
	uuid_copy(ref.ref.part_uuid, SOS_PART_UDATA(part->udata_obj)->uuid);
	ref.ref.obj = obj_ref;
	sos_obj = __sos_init_obj(part->sos, schema, ods_obj, ref);
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
