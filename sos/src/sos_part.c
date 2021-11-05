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
 * The most common use cases for partitions are: partitioning object
 * data by date, partitioning object data by user, or both:
 *
 * First let's create a container:
 *
 *      $ sos-db --path theContainer --create
 *      $ sos-db --path theContainer --query
 *
 *      $ sos-db --path myContainer --query
 *      Name               State      Accessed           Modified           Size       Path
 *      ------------------ ---------- ------------------ ------------------ ----------- --------------------                                                                                  *      default            PRIMARY    11/04/21 10:34:25  11/04/21 10:34:03       0.1MB /primaryStorage/myContainer/default
 *
 * We can also use the sos-part command to query the partition and get more detail:
 *
 *      $ sos-part --cont myContainer --query
 *      Name               State      uid   gid   Permissions  Size       Description                          Path
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *      default            PRIMARY    12345 12345 rw-rw-r--        65.5K  default container partition          /primaryStorage/myContainer/default
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *
 * A default partition is created to accept new data, but let's add two new
 * new partitions for storing data by date:
 *
 *      $ sos-part --create --path /primaryStorage/partitions/today --mode 0o666
 *      $ sos-part --create --path /primaryStorage/partitions/tomorrow --mode 0o666
 *
 * These partitions are not associated with a container yet, we have to attach them
 * to myContainer:
 *
 *      $ sos-part --query --cont myContainer
 *      Name               State      uid   gid   Permissions  Size       Description                          Path
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *      default            PRIMARY    12345 12345 rw-rw-r--        65.5K  default container partition          /primaryStorage/myContainer/default
 *      today              OFFLINE        0     0 rw-rw-rw-         245B  today's data                         /primaryStorage/partitions/today
 *      tomorrow           OFFLINE        0     0 rw-rw-rw-         254B  tomorrow's data                      /primaryStorage/partitions/tomorrow
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *
 * To cause new data to flow into 'today', we need to set it's state to primary:
 *
 *      $ sos-part --state primary --name today --cont myContainer
 *      $ sos-part --query --cont myContainer
 *      Name               State      uid   gid   Permissions  Size       Description                          Path
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *      default            ACTIVE     62538 62538 rw-rw-r--        65.5K  default container partition          /primaryStorage/myContainer/default
 *      today              PRIMARY        0     0 rw-rw-rw-         331B  today's data                         /primaryStorage/partitions/today
 *      tomorrow           OFFLINE        0     0 rw-rw-rw-         254B  tomorrow's data                      /primaryStorage/partitions/tomorrow
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *
 * A typical use case for partitions is to group objects together by
 * date and then migrate older partitions to another container on
 * secondary storage.
 *
 * At midnight the administrator starts storing data in tomorrow's partition as follows:
 *
 *      $ sos-part --state primary --name tomorrow --cont myContainer
 *      $ sos-part --query --cont myContainer
 *      Name               State      uid   gid   Permissions  Size       Description                          Path
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *      default            ACTIVE     12345 12345 rw-rw-r--        65.5K  default container partition          /primaryStorage/myContainer/default
 *      today              ACTIVE         0     0 rw-rw-rw-         331B  today's data                         /primaryStorage/partitions/today
 *      tomorrow           PRIMARY        0     0 rw-rw-rw-         254B  tomorrow's data                      /primaryStorage/partitions/tomorrow
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *
 * New object allocations will immediately start flowing into the new
 * primary partition called 'tomorrow'. Objects in the original partition
 * are still accessible.
 *
 * The administrator then wants to migrate the data from the today
 * partition to another container in secondary storage. This can be
 * accomplished as follows:
 *
 * Detach the 'today' partition from the container and move it to
 * secondary storage:
 *
 *      $ sos-part --detach --cont myContainer --name today
 *      $ sos-part --query --cont myContainer
 *      Name               State      uid   gid   Permissions  Size       Description                          Path
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *      default            ACTIVE     12345 12345 rw-rw-r--        65.5K  default container partition          /primaryStorage/myContainer/default
 *      tomorrow           PRIMARY        0     0 rw-rw-rw-         254B  tomorrow's data                      /primaryStorage/partitions/tomorrow
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *
 * The partition is no longer attached to myContainer; move it to secondary storage:
 *
 *      $ export NAME=$(date +%s)
 *      $ mv /primaryStorage/partitions/today /secondaryStorage/partitions/$NAME
 *
 * Create a new container to attacgh the partition to:
 *
 *      $ sos-db --path /secondarStorage/backupContainer --create
 *      $ sos-part --attach --cont /secondaryStorage/backupContainer --name $NAME
 *      $ sos-part --state active --cont /secondaryStorage/backupContainer --name $NAME
 *      $ sos-part --query --cont /secondaryStorage/backupContainer
 *      Name               State      uid   gid   Permissions  Size       Description                          Path
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *      default            PRIMARY    12345 12345 rw-rw-r--        65.5K  default container partition          /secodaryStorage/backupContainer/default
 *      1636055168         ACTIVE         0     0 rw-rw-rw-         254B  today's data                         /secondaryStorage/partitions/1636055168
 *      ------------------ ---------- ----- ----- ------------ ---------- ------------------------------------ --------------------
 *
 * All objects in the today partition are now available to be queried
 * from the backupContainer.
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
#include <pwd.h>
#include <grp.h>
#include <pthread.h>

#include <sos/sos.h>
#include <ods/ods.h>
#include <ods/ods_idx.h>
#include "sos_priv.h"

int64_t part_comparator(void *a, const void *b, void *arg)
{
	return strcmp(a, b);
}

struct ods_rbt part_tree = ODS_RBT_INITIALIZER(part_comparator);
pthread_mutex_t part_tree_lock = PTHREAD_MUTEX_INITIALIZER;

static sos_part_t __sos_part_by_name(sos_t sos, const char *name);
static sos_part_t __sos_part_by_path(sos_t sos, const char *path);
static sos_part_t __sos_part_first(sos_part_iter_t iter);
static sos_part_t __sos_part_next(sos_part_iter_t iter);
static int __refresh_part_list(sos_t sos, uid_t uid, gid_t gid, int acc);
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

/* Return 1 if the gid is in the group list for uid */
static int __uid_gid_check(uid_t uid, gid_t gid)
{
	struct passwd pwd;
	struct passwd *result;
	int rc;
	long bufsize;
	char *buf;
	int i;
	int access = 0;
	int n_gids = 0;
	gid_t *gid_list = NULL;

        bufsize = sysconf(_SC_GETPW_R_SIZE_MAX);
        if (bufsize == -1) {
		/* Can't get it ... guess */
		bufsize = 16384;
	}

	buf = malloc(bufsize);
	if (buf == NULL)
		goto out_0;

	/* Get the user name */
	rc = getpwuid_r(uid, &pwd, buf, bufsize, &result);
	if (result == NULL)
		goto out_1;

	/*
	 * Get the number of groups needed, when n_gids == 0,
	 * output is gid count
	 */
	rc = getgrouplist(result->pw_name, result->pw_gid, gid_list, &n_gids);

	/* Allocate the group list */
retry:
	gid_list = calloc(n_gids, sizeof(*gid_list));
	if (!gid_list)
		goto out_1;

	rc = getgrouplist(result->pw_name, result->pw_gid, gid_list, &n_gids);
	if (rc < 0) {
		/* Another process added a group asynchronously */
		free(gid_list);
		goto retry;
	}

	for (i = 0; i < n_gids; i++) {
		if (gid == gid_list[i]) {
			access = 1;
			goto out_1;
		}
	}

out_1:
	free(buf);
out_0:
	return access;
}

/* Returns 1 if the specified user/group has access to the partition */
static int __access_check(ods_obj_t part_obj, uid_t euid, gid_t egid, int access)
{
	int permissions = SOS_PART_UDATA(part_obj)->perm;
	if (euid == 0)
		return 1;
	/* Other */
	if (07 & permissions) {
		if ((access & permissions) == access)
			return 1;
	}
	/* Owner */
	if (0700 & permissions) {
		if (SOS_PART_UDATA(part_obj)->user_id == euid) {
			if (((access << 6) & permissions) == (access << 6))
				return 1;
		}
	}
	/* Group */
	if (070 & permissions) {
		if (SOS_PART_UDATA(part_obj)->group_id == egid) {
			if (((access << 3) & permissions) == (access << 3))
				return 1;
		}
		if (__uid_gid_check(euid, SOS_PART_UDATA(part_obj)->group_id)) {
			if (((access << 3) & permissions) == (access << 3))
				return 1;
		}
	}
	return 0;
}

static int __part_open(sos_part_t part, const char *path,
		       sos_perm_t o_perm, uid_t euid, gid_t egid, int acc)
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

	if (!__access_check(part_obj, euid, egid, acc)) {
		rc = EPERM;
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

void __part_close(sos_part_t part)
{
	ods_obj_put(part->udata_obj);
	part->udata_obj = NULL;
	ods_close(part->obj_ods, ODS_COMMIT_ASYNC);
	part->obj_ods = NULL;
}

static sos_part_t __sos_part_open(sos_t sos, ods_obj_t ref_obj,
				  uid_t euid, gid_t egid, int acc)
{
	int rc;
	sos_part_t part;

	/* See if this partition is already open */
	pthread_mutex_lock(&part_tree_lock);
	struct ods_rbn *rbn = ods_rbt_find(&part_tree, SOS_PART_REF(ref_obj)->path);
	if (rbn) {
		part = container_of(rbn, struct sos_part_s, rbn);
		if (!__access_check(part->udata_obj, euid, egid, acc)) {
			rc = EPERM;
			part = NULL;
			goto out;
		}
		sos_ref_get(&part->ref_count, __func__);
		goto out;
	}
	part  = calloc(1, sizeof *part);
	if (!part)
		goto out;
	sos_ref_init(&part->ref_count, "part_list", __sos_part_free, part);
	rc = __part_open(part, SOS_PART_REF(ref_obj)->path,
			sos->o_perm & ~SOS_PERM_CREAT, euid, egid, acc);
	if (rc) {
		free(part);
		part = NULL;
		errno = rc;
		goto out;
	}
	part->ref_obj = ods_obj_get(ref_obj);
	part->sos = sos;
	TAILQ_INSERT_TAIL(&sos->part_list, part, entry);
	ods_rbn_init(&part->rbn, SOS_PART_REF(ref_obj)->path);
	sos_ref_get(&part->ref_count, __func__);
	ods_rbt_ins(&part_tree, &part->rbn);
out:
	pthread_mutex_unlock(&part_tree_lock);
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
 * @param desc   A pointer to a character string description for the partition.
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
	rc = __part_open(part, path, o_perm, 0, 0, 06);
	if (rc) {
		errno = rc;
		free(part);
		return NULL;
	}
	if (o_perm & SOS_PERM_CREAT)
		ods_commit(part->obj_ods, ODS_COMMIT_SYNC);
	return part;
}

void sos_part_close(sos_part_t part)
{
	if (!part)
		return;
	if (part->sos)
		return;		/* It will get closed when container closes */
	ods_obj_put(part->ref_obj);
	ods_obj_put(part->udata_obj);
	ods_close(part->obj_ods, ODS_COMMIT_SYNC);
}

int sos_part_chown(sos_part_t part, uid_t uid, gid_t gid)
{
	if (uid >= 0)
		SOS_PART_UDATA(part->udata_obj)->user_id = uid;
	if (gid >= 0)
		SOS_PART_UDATA(part->udata_obj)->group_id = gid;
	ods_obj_update(part->udata_obj);
	return 0;
}

int sos_part_chmod(sos_part_t part, int mode)
{
	SOS_PART_UDATA(part->udata_obj)->perm = mode;
	ods_obj_update(part->udata_obj);
	return 0;
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
 * \brief Set the state of an attached partition
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
 * \retval ENOENT The partition is not attached to a container
 */
int sos_part_state_set(sos_part_t part, sos_part_state_t new_state)
{
	sos_t sos = part->sos;
	if (!sos)
		/* A partition must be attached to set it's state */
		return EINVAL;
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

static int __refresh_part_list(sos_t sos, uid_t euid, gid_t egid, int acc)
{
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
		part = __sos_part_open(sos, part_obj, euid, egid, acc);
		if (!part) {
			if (errno == EPERM) {
				continue;
			} else {
				sos_error("Error %d opening the partition '%s' "
					  "at path '%s'.\n",
					  errno,
					  SOS_PART_REF(part_obj)->name,
					  SOS_PART_REF(part_obj)->path);
				continue;
			}
		}
		if (SOS_PART_REF(part_obj)->state == SOS_PART_STATE_PRIMARY) {
			sos->primary_part = part;
			sos_ref_get(&part->ref_count, "primary_part");
		}
	}
	if (part_obj)
		ods_obj_put(part_obj);
	return 0;
}

static int refresh_part_list(sos_t sos, uid_t euid, gid_t egid, int acc)
{
	int rc;
	ods_lock(sos->part_ref_ods, 0, NULL);
	rc = __refresh_part_list(sos, euid, egid, acc);
	ods_unlock(sos->part_ref_ods, 0);
	return rc;
}

int __sos_open_partitions(sos_t sos, char *tmp_path, uid_t euid, gid_t egid, int acc)
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
	rc = refresh_part_list(sos, euid, egid, acc);
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
		int rc = __refresh_part_list(sos, 0, 0, 06);
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
	char path_[PATH_MAX];
	sos_part_t part = NULL;
	ods_ref_t head_ref, tail_ref;
	ods_obj_t new_part_ref;
	int rc = 0;

	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ref_ods, 0, NULL);

	part = __sos_part_by_name(sos, name);
	if (!part)
		part = __sos_part_by_path(sos, path);
	if (part) {
		rc = EEXIST;
		goto err_0;
	}

	/*
	 * Make the path absolute, relative paths are not allowed
	 * because it would make the partition inaccessible from
	 * other directories
	 */
	if (NULL == realpath(path, path_)) {
		rc = ENAMETOOLONG;
		goto err_0;
	}
	path = path_;
	if (strlen(path) >= SOS_PART_PATH_LEN || strlen(name) >= SOS_PART_NAME_LEN) {
		rc = ENAMETOOLONG;
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
	part = __sos_part_open(sos, new_part_ref, 0, 0, 06);
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
	ods_obj_update(part->udata_obj);
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
	sos_part_put(part);
	return rc;
}

/**
 * @brief Detach a partition from a container
 *
 * Detaches the partition from the container.
 *
 * @param sos The container handle
 * @param name The name of the partition
 * @retval 0 The partition was successfully detached
 * @retval ENOENT The partition is not attached
 */
int sos_part_detach(sos_t sos, const char *name)
{
	sos_part_t part;
	ods_ref_t prev_ref, next_ref;
	ods_obj_t part_ref;
	int rc = 0;

	pthread_mutex_lock(&sos->lock);
	ods_lock(sos->part_ref_ods, 0, NULL);

	/* Find the partition reference  */
	for (part_ref = __sos_part_ref_first(sos);
	     part_ref; part_ref = __sos_part_ref_next(sos, part_ref)) {
		if (0 == strcmp(name, SOS_PART_REF(part_ref)->name)) {
			prev_ref = SOS_PART_REF(part_ref)->prev;
			next_ref = SOS_PART_REF(part_ref)->next;
			break;
		}
	}
	if (part_ref == NULL) {
		rc = ENOENT;
		goto out;
	}

	if (SOS_PART_REF(part_ref)->state == SOS_PART_STATE_PRIMARY) {
		rc = EINVAL;
		goto out;
	}

	/* Remove the partition reference */
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
	ods_atomic_inc(&SOS_PART_REF_UDATA(sos->part_ref_udata)->gen);
	ods_obj_update(sos->part_ref_udata);
	/* Remove the sos_part_t from the part_list, if present */
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (part->ref_obj == part_ref)
			break;
	}
	if (part) {
		/* The partition may have failed to be opened due to
		 * permission or path errors */
		TAILQ_REMOVE(&sos->part_list, part, entry);
		ods_atomic_dec(&SOS_PART_UDATA(part->udata_obj)->ref_count);
	}
	ods_obj_delete(part_ref);
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
 * @brief Return the user-id for the partition
 *
 * Returns the user id that owns the partition.
 *
 * @param part The partition handle
 * @return The uid_t that owns the container
 */
uid_t sos_part_uid(sos_part_t part)
{
	return SOS_PART_UDATA(part->udata_obj)->user_id;
}
/**
 * @brief Return the group-id for the partition
 *
 * Returns the group id that owns the partition.
 *
 * @param part The partition handle
 * @return The gid_t that owns the container
 */
uid_t sos_part_gid(sos_part_t part)
{
	return SOS_PART_UDATA(part->udata_obj)->group_id;
}
/**
 * @brief Return the access rights bit mask
 *
 * Returns the permission bits for access to the partition.
 *
 * @param part The partition handle
 */
int sos_part_perm(sos_part_t part)
{
	return SOS_PART_UDATA(part->udata_obj)->perm;
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
 * @brief Set a partition's description
 *
 * @param part The partition handle
 * @param desc The description string
 */
void sos_part_desc_set(sos_part_t part, const char *desc)
{
	strncpy(SOS_PART_UDATA(part->udata_obj)->desc,
		desc, sizeof(SOS_PART_UDATA(part->udata_obj)->desc));
	ods_obj_update(part->udata_obj);
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
	if (!part->sos || !part->udata_obj || !part->ref_obj)
		return SOS_PART_STATE_DETACHED;
	return SOS_PART_REF(part->ref_obj)->state;
}

/* Shallow copy a directory and its contents */
int __sos_copy_directory(const char *src_dir, const char *dst_dir)
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
			sos_error("Error %d opening the source file '%s'\n", errno, path);
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
	sos_obj_t sos_obj = NULL;
	ods_obj_t ods_obj = ods_ref_as_obj(ods, obj_ref);
	if (!ods_obj) {
		sos_error("Object reference %p could not be instantiated as a partition\n", (void *)obj_ref);
		return 0;
	}
	struct part_obj_iter_args_s *oi_args = arg;
	sos_obj_ref_t ref;
	sos_part_t part = oi_args->part;
	sos_obj_data_t sos_obj_data = ods_obj->as.ptr;
	if (part->sos) {
		/* Partition is attached, instantiate the SOS object */
		sos_schema_t schema = sos_schema_by_uuid(part->sos, sos_obj_data->schema_uuid);
		if (!schema) {
			sos_warn("Object at %p is missing a valid schema id.\n", (void *)obj_ref);
			/* This is a garbage object that should not be here */
			return 0;
		}
		uuid_copy(ref.ref.part_uuid, SOS_PART_UDATA(part->udata_obj)->uuid);
		ref.ref.obj = obj_ref;
		sos_obj = __sos_init_obj(part->sos, schema, ods_obj, ref);
		if (!sos_obj)
			sos_warn("SOS object could not be instantiated in attached partition.");
	}
	return oi_args->fn(oi_args->part, ods_obj, sos_obj, oi_args->arg);
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
