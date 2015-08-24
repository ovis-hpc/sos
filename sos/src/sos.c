/*
 * Copyright (c) 2012-2015 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2012-2015 Sandia Corporation. All rights reserved.
 *
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
/**
 * \mainpage Scalable Object Store Documentation
 *
 * \section intro Introduction
 *
 * The Scalable Object Storage (SOS) Service is a high performance
 * storage engine designed to efficiently store structured data to
 * persistent media.
 *
 * \subsection coll Collection
 *
 * A SOS Object Store is called a Collection. A Collection is
 * identified by a name that looks like a POSIX Filesytem path. This
 * allows Collections to be organized into a hierarchical name
 * space. This is a convenience and does not mean that a Container is
 * necessarily stored in a Filesytem.
 *
 * \subsection schema Schema
 *
 * Inside a Container are Schemas, Objects, and Indices. A Schema
 * defines the format of an Object. There can be any number of Schema
 * in the Container such that a single Container may contain Objects
 * of many different types. The Container has a directory of
 * Schemas. When Objects are created, the Schema handle is specified
 * to inform the object store of the size and format of the object and
 * whether or not one or more of it's attributes has an Index.
 *
 * \subsection object Object
 *
 * An Object is an instance of a Schema. An Object is a collection of
 * Attributes. An Attribute has a Name and a Type. There are built-in
 * types for an Attribute and user-defined types. The built in types
 * include the familiar <tt>int</tt>, <tt>long</tt>, <tt>double</tt>
 * types as well as arrays of these types. A special Attribute type is
 * <tt>SOS_TYPE_OBJ</tt>, which is a <tt>Reference</tt> to another Object. This
 * allows complex data structures like linked lists to be implemented
 * in the Container.
 *
 * The user-defined types are Objects. An Object's type is essentially
 * it's Schema ID and Schema Name.
 *
 * An Index is a strategy for quickly finding an Object in a container
 * based on the value of one of it's Attributes. Whether or not an
 * Attribute has an Index is specified by the Object's Schema.
 */

#include <sys/queue.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
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

LIST_HEAD(cont_list_head, sos_container_s) cont_list;
pthread_mutex_t cont_list_lock;

/**
 * \page container_overview Container Overview
 *
 * SOS Container group Schmema, Objects, and Indices together into a
 * single namespace. The root of the namespace is the container's
 * name. Containers are created with the sos_container_new(). SOS
 * implements the POSIX security model. When a Container is created,
 * it inherits the owner and group of the process that created the
 * container. The sos_container_new() function takes an o_mode
 * parameter that identifies the standard POSIX umask to specify RO/RW
 * access for owner/group/other.
 *
 * The sos_container_open() function opens a previously created
 * container. The user/group of the process opening the container must
 * have adequate permission to provide the requested R/W access. The
 * sos_container_open() function returns a sos_t container handle that
 * is used in subsequent SOS API.
 *
 * Changes to a container are opportunistically commited to stable
 * storage. An application can initiate a commit to storage with the
 * sos_container_commit() function.
 *
 * - sos_container_new() Create a new Container
 * - sos_container_open() Open a previously created Container
 * - sos_container_close() Close a container
 * - sos_container_commit() Commit a Container's contents to stable storage
 * - sos_container_delete() Destroy a Container and all of it's contents
 * - sos_container_info() - Print Container information to a FILE pointer
 */

/* This function effectively implements 'mkdir -p' */
static int make_all_dir(const char *inp_path, mode_t omode)
{
	struct stat sb;
	mode_t numask, oumask;
	int first, last, retval;
	char *p, *path;

	p = path = strdup(inp_path);
	if (!p) {
		errno = ENOMEM;
		return 1;
	}

	oumask = 0;
	retval = 0;
	if (p[0] == '/')
		++p;

	for (first = 1, last = 0; !last ; ++p) {
		if (p[0] == '\0')
			last = 1;
		else if (p[0] != '/')
			continue;
		*p = '\0';
		if (!last && p[1] == '\0')
			last = 1;
		if (first) {
			oumask = umask(0);
			numask = oumask & ~(S_IWUSR | S_IXUSR);
			(void)umask(numask);
			first = 0;
		}
		if (last)
			(void)umask(oumask);
		if (mkdir(path, last ? omode : S_IRWXU | S_IRWXG | S_IRWXO) < 0) {
			if (errno == EEXIST || errno == EISDIR) {
				if (stat(path, &sb) < 0) {
					retval = 1;
					break;
				} else if (!S_ISDIR(sb.st_mode)) {
					if (last)
						errno = EEXIST;
					else
						errno = ENOTDIR;
					retval = 1;
					break;
				}
			} else {
				retval = 1;
				break;
			}
		}
		if (!last)
			*p = '/';
	}
	if (!first && !last)
		(void)umask(oumask);
	free(path);
	return retval;
}

/** \defgroup container SOS Storage Containers
 * @{
 */

/**
 * \brief Create a Container
 *
 * Creates a SOS container. The o_flags and o_mode parameters accept
 * the same values and have the same meaning as the corresponding
 * parameters to the open() system call.
 *
 * Containers are logically maintained in a Unix filesystem
 * namespace. The specified path must be unique for the Container and
 * all sub-directories in the path up to, but not including the
 * basename() must exist.
 *
 * \param path		Pathname for the Container.
 * \param o_mode	The file mode for the Container.
 * \retval 0		The container was successfully created.
 * \retval EINVAL	A parameter was invalid
 * \retval EPERM	The user has insufficient permission
 * \retval EEXIST	A container already exists at the specified path
 */
int sos_container_new(const char *path, int o_mode)
{
	char tmp_path[PATH_MAX];
	int rc;
	int x_mode;
	struct stat sb;

	/* Check to see if the container already exists */
	rc = stat(path, &sb);
	if (rc != 0 && errno != ENOENT)
		return rc;

	x_mode = o_mode;
	if (x_mode & (S_IWGRP | S_IRGRP))
		x_mode |= S_IXGRP;
	if (x_mode & (S_IWUSR | S_IRUSR))
		x_mode |= S_IXUSR;
	if (x_mode & (S_IWOTH | S_IROTH))
		x_mode |= S_IXOTH;

	rc = make_all_dir(path, x_mode);
	if (rc) {
		rc = errno;
		goto err_0;
	}

	/* Create the ODS to contain configuration objects */
	sprintf(tmp_path, "%s/.__config", path);
	rc = ods_create(tmp_path, o_mode);
	if (rc)
		goto err_1;

	/* Create the configuration object index */
	sprintf(tmp_path, "%s/.__config_idx", path);
	rc = ods_idx_create(tmp_path, o_mode, "BXTREE", "STRING", NULL);
	if (rc)
		goto err_2;

	/* Create the ODS to contain partition objects */
	sprintf(tmp_path, "%s/.__part", path);
	rc = ods_create(tmp_path, o_mode);
	if (rc)
		goto err_3;

	ods_t part_ods = ods_open(tmp_path, O_RDWR);
	if (!part_ods)
		goto err_3;

	ods_obj_t udata = ods_get_user_data(part_ods);
	if (!udata) {
		rc = errno;
		ods_close(part_ods, ODS_COMMIT_ASYNC);
		goto err_4;
	}
	SOS_PART_UDATA(udata)->signature = SOS_PART_SIGNATURE;
	SOS_PART_UDATA(udata)->primary = 0;
	SOS_PART_UDATA(udata)->head = 0;
	SOS_PART_UDATA(udata)->tail = 0;
	SOS_PART_UDATA(udata)->lock = 0;
	SOS_PART_UDATA(udata)->next_part_id = 1;

	ods_obj_put(udata);
	ods_close(part_ods, ODS_COMMIT_SYNC);

	/* Default configuration is to disable partition rotation */
	rc = sos_container_config_set(path, "PARTITION_ENABLE", "false");
	if (rc)
		goto err_4;

 	/* Create the ODS to contain the schema objects */
	sprintf(tmp_path, "%s/.__schemas", path);
 	rc = ods_create(tmp_path, o_mode);
 	if (rc)
 		goto err_4;

	ods_t schema_ods = ods_open(tmp_path, O_RDWR);
	if (!schema_ods)
		goto err_5;
	/* Initialize the schema dictionary */
	udata = ods_get_user_data(schema_ods);
	if (!udata) {
		rc = errno;
		ods_close(schema_ods, ODS_COMMIT_ASYNC);
		goto err_5;
	}
	SOS_SCHEMA_UDATA(udata)->signature = SOS_SCHEMA_SIGNATURE;
	SOS_SCHEMA_UDATA(udata)->version = SOS_LATEST_VERSION;
	SOS_SCHEMA_UDATA(udata)->last_schema_id = SOS_SCHEMA_FIRST_USER;
	ods_obj_put(udata);
	ods_close(schema_ods, ODS_COMMIT_ASYNC);

	/* Create the index to look up the schema names */
	sprintf(tmp_path, "%s/.__schema_idx", path);
 	rc = ods_idx_create(tmp_path, o_mode, "BXTREE", "STRING", NULL);
 	if (rc)
 		goto err_5;

 	/* Create the ODS to contain the index objects */
	sprintf(tmp_path, "%s/.__index", path);
 	rc = ods_create(tmp_path, o_mode);
 	if (rc)
 		goto err_6;

	ods_t idx_ods = ods_open(tmp_path, O_RDWR);
	if (!idx_ods)
		goto err_7;

	/* Initialize the index dictionary */
	udata = ods_get_user_data(schema_ods);
	if (!udata) {
		rc = errno;
		ods_close(idx_ods, ODS_COMMIT_ASYNC);
		goto err_7;
	}
	SOS_IDXDIR_UDATA(udata)->signature = SOS_IDXDIR_SIGNATURE;
	SOS_IDXDIR_UDATA(udata)->lock = 0;
	ods_obj_put(udata);
	ods_close(idx_ods, ODS_COMMIT_ASYNC);

	/* Create the index to look up the schema names */
	sprintf(tmp_path, "%s/.__index_idx", path);
 	rc = ods_idx_create(tmp_path, o_mode, "BXTREE", "STRING", NULL);
 	if (rc)
 		goto err_7;
	return 0;
 err_7:
	sprintf(tmp_path, "%s/.__index", path);
	ods_destroy(tmp_path);
 err_6:
	sprintf(tmp_path, "%s/.__schemas_idx", path);
	ods_destroy(tmp_path);
 err_5:
	sprintf(tmp_path, "%s/.__schemas", path);
	ods_destroy(tmp_path);
 err_4:
	sprintf(tmp_path, "%s/.__part", path);
	ods_destroy(tmp_path);
 err_3:
	sprintf(tmp_path, "%s/.__config_idx", path);
	ods_destroy(tmp_path);
 err_2:
	sprintf(tmp_path, "%s/.__config", path);
	ods_destroy(tmp_path);
 err_1:
	rmdir(path);
 err_0:
	return rc;
}

sos_part_t __sos_part_new(sos_t sos, ods_obj_t part_obj)
{
	sos_part_t part = calloc(1, sizeof(*part));
	if (!part)
		return NULL;
	part->sos = sos;
	part->part_obj = part_obj;
	return part;
}

void sos_part_put(sos_part_t part)
{
	__sos_part_obj_put(part->sos, part->part_obj);
	free(part);
}

static int __sos_open_partition(sos_t sos, sos_part_t part)
{
	char tmp_path[PATH_MAX];
	int rc;
	ods_t ods;

	sprintf(tmp_path, "%s/%s", sos->path, sos_part_name_get(part));
	rc = make_all_dir(tmp_path, sos->o_mode);
	if (rc) {
		rc = errno;
		goto err_0;
	}
	sprintf(tmp_path, "%s/%s/objects", sos->path, sos_part_name_get(part));
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

static int __sos_part_create(sos_t sos, char *tmp_path, const char *part_name)
{
	int rc;
	struct stat sb;
	ods_ref_t head_ref, tail_ref;
	ods_obj_t part, new_part;

	/* Take the partition lock */
	ods_spin_lock(&sos->part_lock, -1);

	/* See if the specified name already exists in the filesystem */
	sprintf(tmp_path, "%s/%s", sos->path, part_name);
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
	SOS_PART(new_part)->part_id = ods_atomic_inc(&SOS_PART_UDATA(sos->part_udata)->next_part_id);
	SOS_PART(new_part)->prev = tail_ref;
	SOS_PART(new_part)->next = 0;
	SOS_PART(new_part)->ref_count = 1;
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
	ods_spin_unlock(&sos->part_lock);
	ods_obj_put(new_part);
	return 0;
 err_2:
	ods_obj_delete(new_part);
	ods_obj_put(new_part);
 err_1:
	ods_spin_unlock(&sos->part_lock);
	return errno;
}

static int __sos_open_partitions(sos_t sos, char *tmp_path)
{
	int rc;
	sos_part_t part;

	/* Open the partition ODS */
	sprintf(tmp_path, "%s/.__part", sos->path);
	sos->part_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!sos->part_ods)
		goto err_0;
	sos->part_udata = ods_get_user_data(sos->part_ods);
	if (!sos->part_udata)
		goto err_1;
	ods_spin_init(&sos->part_lock, &(SOS_PART_UDATA(sos->part_udata)->lock));

	sos_part_iter_t iter = sos_part_iter_new(sos);
	if (!iter)
		goto err_1;

	part = sos_part_first(iter);
	if (!part) {
		rc = __sos_part_create(sos, tmp_path, SOS_PART_NAME_DEFAULT);
		if (rc)
			goto err_2;
		part = sos_part_first(iter);
		assert(part);
		sos_part_primary_set(part);
	}
	for (;part; part = sos_part_next(iter)) {

		if (0 == (sos_part_state_get(part) & SOS_PART_STATE_ACTIVE)) {
			/* Skip offline partitions so that we don't hold a reference on them */
			sos_part_put(part);
			continue;
		}

		rc = __sos_open_partition(sos, part);
		if (rc)
			goto err_3;
		TAILQ_INSERT_TAIL(&sos->part_list, part, entry);
	}
	sos_part_iter_free(iter);
	return 0;
 err_3:
	sos_part_put(part);
 err_2:
	sos_part_iter_free(iter);
 err_1:
	ods_close(sos->part_ods, ODS_COMMIT_ASYNC);
 err_0:
	return errno;
}

/**
 * \brief Delete storage associated with a Container
 *
 * Removes all resources associated with the Container. The sos_t
 * handle must be provided (requiring an open) because it is necessary
 * to know the associated indexes in order to be able to know the
 * names of the associated files. sos_destroy will also close \c sos, as the
 * files should be closed before removed.
 *
 * \param c	The container handle
 * \retval 0	The container was deleted
 * \retval EPERM The user has insufficient privilege
 * \retval EINUSE The container is in-use by other clients
 */
int sos_container_delete(sos_t c)
{
	return ENOSYS;
}

int sos_index_commit(sos_index_t index, sos_commit_t flags)
{
	ods_idx_commit(index->idx, flags);
	return 0;
}

/**
 * \brief Flush outstanding changes to persistent storage
 *
 * This function commits the index changes to stable storage. If
 * SOS_COMMIT_SYNC is specified in the flags parameter, the function
 * will wait until the changes are commited to stable stroage before
 * returning.
 *
 * \param sos	The SOS container handle
 * \param flags	The commit flags
 */
int sos_container_commit(sos_t sos, sos_commit_t flags)
{
	sos_schema_t schema;
	sos_attr_t attr;
	int commit;

	if (flags == SOS_COMMIT_SYNC)
		commit = ODS_COMMIT_SYNC;
	else
		commit = ODS_COMMIT_ASYNC;

	/* Commit the schema idx and ods */
	ods_commit(sos->schema_ods, commit);
	ods_idx_commit(sos->schema_idx, commit);

	/* Commit the object ods */
	sos_part_t part;
	TAILQ_FOREACH(part, &sos->part_list, entry)
		if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_ACTIVE)
			ods_commit(part->obj_ods, commit);

	/* Commit all the attribute indices */
	LIST_FOREACH(schema, &sos->schema_list, entry) {
		TAILQ_FOREACH(attr, &schema->attr_list, entry) {
			if (attr->index)
				sos_index_commit(attr->index, commit);
		}
	}
	return 0;
}

void sos_index_info(sos_index_t index, FILE *fp)
{
	ods_idx_info(index->idx, fp);
	ods_info(ods_idx_ods(index->idx), fp);
}

int print_schema(struct rbn *n, void *fp_, int level)
{
	FILE *fp = fp_;
	sos_attr_t attr;

	sos_schema_t schema = container_of(n, struct sos_schema_s, name_rbn);
	sos_schema_print(schema, fp);

	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		if (attr->index)
			sos_index_info(attr->index, fp);
	}
	return 0;
}

/**
 * \brief Print container information
 *
 * Prints information about the container to the specified FILE pointer.
 *
 * \param sos	The container handle
 * \param fp	The FILE pointer
 */
void sos_container_info(sos_t sos, FILE *fp)
{
	rbt_traverse(&sos->schema_name_rbt, print_schema, fp);
	ods_idx_info(sos->schema_idx, fp);
	ods_info(ods_idx_ods(sos->schema_idx), fp);
	ods_info(sos->schema_ods, fp);
	sos_part_t part;
	TAILQ_FOREACH(part, &sos->part_list, entry)
		ods_info(part->obj_ods, fp);
}

static void free_sos(sos_t sos, sos_commit_t flags)
{
	struct rbn *rbn;
	sos_obj_t obj;

	/* There should be no objects on the active list */
	assert(LIST_EMPTY(&sos->obj_list));

	/* Iterate through the object free list and free all the objects */
	while (!LIST_EMPTY(&sos->obj_free_list)) {
		obj = LIST_FIRST(&sos->obj_free_list);
		LIST_REMOVE(obj, entry);
		free(obj);
	}

	/* Iterate through all the schema and free each one */
	while (NULL != (rbn = rbt_min(&sos->schema_name_rbt))) {
		rbt_del(&sos->schema_name_rbt, rbn);
		__sos_schema_free(container_of(rbn, struct sos_schema_s, name_rbn));
	}
	if (sos->path)
		free(sos->path);
	if (sos->schema_idx)
		ods_idx_close(sos->schema_idx, flags);
	if (sos->schema_ods)
		ods_close(sos->schema_ods, flags);
	if (sos->idx_idx)
		ods_idx_close(sos->idx_idx, flags);
	if (sos->idx_ods)
		ods_close(sos->idx_ods, flags);

	sos_part_t part;
	while (!TAILQ_EMPTY(&sos->part_list)) {
		part = TAILQ_FIRST(&sos->part_list);
		TAILQ_REMOVE(&sos->part_list, part, entry);
		ods_close(part->obj_ods, flags);
		sos_part_put(part);
	}

	if (sos->part_ods)
		ods_close(sos->part_ods, flags);

	free(sos);
}

int __sos_schema_name_cmp(void *a, void *b)
{
	return strcmp((char *)a, (char *)b);
}

static int schema_id_cmp(void *a, void *b)
{
	return *(uint32_t *)a - *(uint32_t *)b;
}

/**
 * \brief Open a Container
 *
 * Open a SOS container. If successfull, the <tt>c</tt> parameter will
 * contain a valid sos_t handle on exit.
 *
 * \param path		Pathname for the Container. See sos_container_new()
 * \param o_perm	The requested read/write permissions
 * \retval !NULL	The sos_t handle for the container.
 * \retval NULL		An error occured, consult errno for the reason.
 * \retval EPERM	The user has insufficient privilege to open the container.
 * \retval ENOENT	The container does not exist
 */
sos_t sos_container_open(const char *path, sos_perm_t o_perm)
{
	char tmp_path[PATH_MAX];
	sos_t sos;
	struct stat sb;
	int rc;

	sos = calloc(1, sizeof(*sos));
	if (!sos) {
		errno = ENOMEM;
		return NULL;
	}
	pthread_mutex_init(&sos->lock, NULL);
	LIST_INIT(&sos->obj_list);
	LIST_INIT(&sos->obj_free_list);
	TAILQ_INIT(&sos->part_list);

	sos->path = strdup(path);
	if (!sos->path)
		goto err;
	/* Stat the container path to get the file mode bits */
	rc = stat(sos->path, &sb);
	if (rc)
		goto err;
	sos->o_mode = sb.st_mode;
	sos->o_perm = (ods_perm_t)o_perm;
	rbt_init(&sos->schema_name_rbt, __sos_schema_name_cmp);
	rbt_init(&sos->schema_id_rbt, schema_id_cmp);
	sos->schema_count = 0;

	rc = __sos_config_init(sos);
	if (rc)
		goto err;

	/* Open the ODS containing the schema objects */
	sprintf(tmp_path, "%s/.__schemas", path);
	sos->schema_ods = ods_open(tmp_path, sos->o_perm);
	if (!sos->schema_ods)
		goto err;
	ods_obj_t udata = ods_get_user_data(sos->schema_ods);
	if ((SOS_SCHEMA_UDATA(udata)->signature != SOS_SCHEMA_SIGNATURE)
	    || (SOS_SCHEMA_UDATA(udata)->version != SOS_LATEST_VERSION)) {
		errno = EINVAL;
		ods_obj_put(udata);
		goto err;
	}
	ods_obj_put(udata);

	/* Open the index on the schema objects */
	sprintf(tmp_path, "%s/.__schema_idx", path);
	sos->schema_idx = ods_idx_open(tmp_path, sos->o_perm);
	if (!sos->schema_idx)
		goto err;

	/* Open the ODS containing the Index objects */
	sprintf(tmp_path, "%s/.__index", path);
	sos->idx_ods = ods_open(tmp_path, sos->o_perm);
	if (!sos->idx_ods)
		goto err;

	sos->idx_udata = ods_get_user_data(sos->idx_ods);
	if (SOS_IDXDIR_UDATA(sos->idx_udata)->signature != SOS_IDXDIR_SIGNATURE) {
		errno = EINVAL;
		ods_obj_put(sos->idx_udata);
		goto err;
	}
	ods_spin_init(&sos->idx_lock, &SOS_IDXDIR_UDATA(sos->idx_udata)->lock);

	/* Open the index on the schema objects */
	sprintf(tmp_path, "%s/.__index_idx", path);
	sos->idx_idx = ods_idx_open(tmp_path, sos->o_perm);
	if (!sos->idx_idx) {
		ods_obj_put(sos->idx_udata);
		goto err;
	}

	/*
	 * Build the schema dictionary
	 */
	ods_iter_t iter = ods_iter_new(sos->schema_idx);
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_next(iter)) {
		sos_obj_ref_t obj_ref;
		obj_ref.idx_data = ods_iter_data(iter);
		ods_obj_t schema_obj = ods_ref_as_obj(sos->schema_ods, obj_ref.ref.obj);
		sos_schema_t schema = __sos_schema_init(sos, schema_obj);
		if (!schema)
			goto err;
		sos->schema_count++;
		LIST_INSERT_HEAD(&sos->schema_list, schema, entry);
		rc = __sos_schema_open(sos, schema);
		if (rc)
			goto err;
	}
	ods_iter_delete(iter);

	/*
	 * Open the partitions
	 */
	rc = __sos_open_partitions(sos, tmp_path);
	if (rc)
		goto err;
	pthread_mutex_lock(&cont_list_lock);
	LIST_INSERT_HEAD(&cont_list, sos, entry);
	pthread_mutex_unlock(&cont_list_lock);
	return sos;
 err:
	free_sos(sos, SOS_COMMIT_ASYNC);
	return NULL;
}

int sos_container_stat(sos_t sos, struct stat *sb)
{
	sos_part_t part = TAILQ_FIRST(&sos->part_list);
	return ods_stat(part->obj_ods, sb);
}

/**
 * \brief Extend the size of a Container
 *
 * Expand the size of  Container's object store. This function cannot
 * be used to make the container smaller. See the
 * sos_container_truncate() function.
 *
 * \param sos	The container handle
 * \param new_size The desired size of the container
 * \retval 0 The container was successfully extended.
 * \retval ENOSPC There is insufficient storage to extend the container
 * \retval EINVAL The container is currently larger than the requested size
 */
int sos_container_extend(sos_t sos, size_t new_size)
{
	return ods_extend(sos->primary_part->obj_ods, new_size);
}

/**
 * \brief Close a Container
 *
 * This function commits the index changes to stable storage and
 * releases all in-memory resources associated with the container.
 *
 * If SOS_COMMIT_SYNC is specified in the flags parameter, the function
 * will wait until the changes are commited to stable stroage before
 * returning.
 *
 * \param sos	The SOS container handle
 * \param flags	The commit flags
 */
void sos_container_close(sos_t sos, sos_commit_t flags)
{
	pthread_mutex_lock(&cont_list_lock);
	LIST_REMOVE(sos, entry);
	pthread_mutex_unlock(&cont_list_lock);

	free_sos(sos, SOS_COMMIT_ASYNC);
}

sos_obj_t __sos_init_obj(sos_t sos, sos_schema_t schema, ods_obj_t ods_obj,
			 sos_obj_ref_t obj_ref)
{
	sos_obj_t sos_obj;
	pthread_mutex_lock(&sos->lock);
	if (!LIST_EMPTY(&sos->obj_free_list)) {
		sos_obj = LIST_FIRST(&sos->obj_free_list);
		LIST_REMOVE(sos_obj, entry);
	} else
		sos_obj = malloc(sizeof *sos_obj);
	pthread_mutex_unlock(&sos->lock);
	if (!sos_obj)
		return NULL;
	SOS_OBJ(ods_obj)->schema = schema->data->id;
	sos_obj->sos = sos;
	sos_obj->obj = ods_obj;
	sos_obj->obj_ref = obj_ref;
	ods_atomic_inc(&schema->data->ref_count);
	sos_obj->schema = sos_schema_get(schema);
	sos_obj->ref_count = 1;

	return sos_obj;
}

sos_part_t __sos_primary_obj_part(sos_t sos)
{
	sos_part_t part = NULL;

	if ((NULL != sos->primary_part) &&
	    (SOS_PART(sos->primary_part->part_obj)->state & SOS_PART_STATE_PRIMARY))
		return sos->primary_part;

	pthread_mutex_lock(&sos->lock);
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_PRIMARY) {
			sos->primary_part = part;
			goto out;
		}
	}
 out:
	pthread_mutex_unlock(&sos->lock);
	return part;;
}

void __sos_part_primary_set(sos_t sos, ods_obj_t part_obj)
{
	ods_ref_t cur_ref;
	ods_obj_t cur_primary;

	if (SOS_PART(part_obj)->state == (SOS_PART_STATE_PRIMARY | SOS_PART_STATE_ACTIVE))
		return;

	ods_spin_lock(&sos->part_lock, -1);

	/* Fix-up the current primary */
	cur_ref = SOS_PART_UDATA(sos->part_udata)->primary;
	if (cur_ref) {
		cur_primary = ods_ref_as_obj(sos->part_ods, cur_ref);
		assert(cur_primary);
		SOS_PART(cur_primary)->state &= ~SOS_PART_STATE_PRIMARY;
		ods_obj_put(cur_primary);
	}

	/* Make part_obj primary */
	SOS_PART(part_obj)->state = SOS_PART_STATE_PRIMARY | SOS_PART_STATE_ACTIVE;
	SOS_PART_UDATA(sos->part_udata)->primary = ods_obj_ref(part_obj);

	ods_spin_unlock(&sos->part_lock);
}

void sos_part_primary_set(sos_part_t part)
{
	__sos_part_primary_set(part->sos, part->part_obj);
	part->sos->primary_part = part;
}

const char *sos_part_name_get(sos_part_t part)
{
	return SOS_PART(part->part_obj)->name;
}

uint32_t sos_part_state_get(sos_part_t part)
{
	return SOS_PART(part->part_obj)->state;
}

uint32_t sos_part_id_get(sos_part_t part)
{
	return SOS_PART(part->part_obj)->part_id;
}

uint32_t sos_part_refcount_get(sos_part_t part)
{
	return SOS_PART(part->part_obj)->ref_count;
}

/**
 * \brief Create a new partition
 *
 * \param sos The sos_t container handle
 * \param part_name The name of the new partition.
 * \retval 0 Success
 * \retval EEXIST The specified partition already exists
 * \retval EBADF Invalid container handle or other storage error
 * \retval ENOMEM Insufficient resources
 */
int sos_part_create(sos_t sos, const char *part_name)
{
	char tmp_path[PATH_MAX];
	sos_part_t p = sos_part_find(sos, part_name);
	if (p)
		return EEXIST;

	int rc = __sos_part_create(sos, tmp_path, part_name);
	if (rc)
		return rc;

	p = sos_part_find(sos, part_name);
	if (!p)
		return ENOENT;

	rc = __sos_open_partition(sos, p);
	if (rc)
		return rc;

	pthread_mutex_lock(&sos->lock);
	TAILQ_INSERT_TAIL(&sos->part_list, p, entry);
	pthread_mutex_unlock(&sos->lock);

	return 0;
}

sos_part_iter_t sos_part_iter_new(sos_t sos)
{
	sos_part_iter_t iter = calloc(1, sizeof(struct sos_part_iter_s));
	iter->sos = sos;
	return iter;
}

sos_part_t sos_part_find(sos_t sos, const char *name)
{
	sos_part_t part;
	sos_part_iter_t iter = sos_part_iter_new(sos);
	if (!iter)
		return NULL;

	for (part = sos_part_first(iter); part; part = sos_part_next(iter)) {
		if (0 == strcmp(sos_part_name_get(part), name))
			goto out;
		sos_part_put(part);
	}
 out:
	sos_part_iter_free(iter);
	return part;
}

sos_part_t sos_part_first(sos_part_iter_t iter)
{
	sos_part_t part = NULL;
	ods_obj_t part_obj = __sos_part_data_first(iter->sos);
	if (part_obj) {
		part = __sos_part_new(iter->sos, part_obj);
		if (!part) {
			__sos_part_obj_put(iter->sos, part_obj);
			goto out;
		}
		iter->part_obj = part_obj;
	}
 out:
	return part;
}

sos_part_t sos_part_next(sos_part_iter_t iter)
{
	sos_part_t part = NULL;
	ods_obj_t part_obj = __sos_part_data_next(iter->sos, iter->part_obj);
#if 0
	if (iter->part_obj)
		__sos_part_obj_put(iter->sos, iter->part_obj);
#endif
	if (part_obj) {
		part = __sos_part_new(iter->sos, part_obj);
		if (!part) {
			__sos_part_obj_put(iter->sos, part_obj);
			goto out;
		}
		iter->part_obj = part_obj;
	}
 out:
	return part;
}

static void __sos_part_obj_delete(sos_t sos, ods_obj_t part_obj)
{
	ods_ref_t prev_ref, next_ref;

	/* Take the partition lock */
	ods_spin_lock(&sos->part_lock, -1);

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
	ods_spin_unlock(&sos->part_lock);
	ods_obj_delete(part_obj);
}

ods_obj_t __sos_part_obj_get(sos_t sos, ods_obj_t part_obj)
{
	ods_atomic_inc(&SOS_PART(part_obj)->ref_count);
	return ods_obj_get(part_obj);
}

void __sos_part_obj_put(sos_t sos, ods_obj_t part_obj)
{
	if (0 == ods_atomic_dec(&SOS_PART(part_obj)->ref_count))
		__sos_part_obj_delete(sos, part_obj);
	ods_obj_put(part_obj);
}

ods_obj_t __sos_part_data_first(sos_t sos)
{
	ods_obj_t part_obj;

	/* Take the partition lock */
	ods_spin_lock(&sos->part_lock, -1);
	part_obj = ods_ref_as_obj(sos->part_ods, SOS_PART_UDATA(sos->part_udata)->head);
	if (part_obj) {
		__sos_part_obj_get(sos, part_obj);
		/* __sos_part_obj_get also takes a reference on the ODS object.
		 * Drop the ods_ref_as_obj reference.
		 */
		ods_obj_put(part_obj);
	}
	ods_spin_unlock(&sos->part_lock);
	return part_obj;
}

ods_obj_t __sos_part_data_next(sos_t sos, ods_obj_t part_obj)
{
	ods_obj_t next_obj;
	ods_ref_t next_ref = NULL;

	if (!part_obj)
		return NULL;

	next_ref = SOS_PART(part_obj)->next;
	if (!next_ref)
		return NULL;

	/* Take the partition lock */
	ods_spin_lock(&sos->part_lock, -1);
	next_obj = ods_ref_as_obj(sos->part_ods, next_ref);
	if (next_obj) {
		__sos_part_obj_get(sos, next_obj);
		ods_obj_put(next_obj);
	}
	ods_spin_unlock(&sos->part_lock);
	return next_obj;
}

void sos_part_iter_free(sos_part_iter_t iter)
{
	free(iter);
}

/* Must be called with sos->lock held */
static sos_part_t __sos_container_part_find(sos_t sos, const char *name)
{
	sos_part_t part = NULL;
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (0 == strcmp(name, SOS_PART(part->part_obj)->name))
			goto out;
	}
 out:
	return part;
}

int sos_part_active_set(sos_part_t part, int online)
{
	int rc = 0;
	int active;

	if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_PRIMARY)
		return EBUSY;

	pthread_mutex_lock(&part->sos->lock);
	active = SOS_PART(part->part_obj)->state & SOS_PART_STATE_ACTIVE;
	if (online) {
		if (active)
			/* Partition is already active */
			goto out;
		SOS_PART(part->part_obj)->state |= SOS_PART_STATE_ACTIVE;

		/* It should not already be in the container */
		sos_part_t p = __sos_container_part_find(part->sos, sos_part_name_get(part));
		assert(p == NULL);

		/* Open the partition and add it to the container */
		rc = __sos_open_partition(part->sos, part);
		if (rc)
			goto out;
		TAILQ_INSERT_TAIL(&part->sos->part_list, part, entry);
	} else {
		/* TODO: Remove all keys in indexes that refer to objects in this partition */
		sos_part_t p = __sos_container_part_find(part->sos, sos_part_name_get(part));
		assert(p);
		uint64_t part_id = sos_part_id_get(p);
		sos_container_index_iter_t idx_iter =
			sos_container_index_iter_new(part->sos);
		sos_index_t idx;
		for (idx = sos_container_index_iter_first(idx_iter); idx;
		     idx = sos_container_index_iter_next(idx_iter)) {

			sos_iter_t obj_iter = sos_index_iter_new(idx);
			int rc;

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
		if (active)
			SOS_PART(part->part_obj)->state &= ~SOS_PART_STATE_ACTIVE;
	}
out:
	pthread_mutex_unlock(&part->sos->lock);
	return rc;
}

const char *pretty_file_size(off_t size)
{
	int i;
	static char buf[32];
	const char sz_strs[] = { ' ', 'K', 'M', 'G', 'T' };
	size_t str_cnt = sizeof(sz_strs) / sizeof(sz_strs[0]);

	for (i = 0; i < str_cnt; i++) {
		if (size < 1000 || str_cnt == 4) {
			sprintf(buf, "%ld%c", size, sz_strs[i]);
			return buf;
		}
		size = size / 1000;
	}
	return NULL;		/* NB: can't happen */
}

void sos_container_part_list(sos_t sos, FILE *fp)
{
	struct stat sb;
	sos_part_t part;
	sos_part_iter_t iter;

	if (!fp)
		fp = stdout;

	iter = sos_part_iter_new(sos);
	if (!iter)
		return;

	fprintf(fp, "%-20s %-8s %-16s %-8s %-16s %-16s\n", "Partition Name", "RefCount",
		"Status", "Size", "Modified", "Accessed");
	fprintf(fp, "-------------------- -------- ---------------- "
		"-------- ---------------- ----------------\n");
	for (part = sos_part_first(iter); part; part = sos_part_next(iter)) {

		char *statestr;
		fprintf(fp, "%-20s %8d ", sos_part_name_get(part), sos_part_refcount_get(part));
		if (0 == (sos_part_state_get(part) & SOS_PART_STATE_ACTIVE))
			statestr = "OFFLINE";
		else if (sos_part_state_get(part) & SOS_PART_STATE_PRIMARY)
			statestr = "ONLINE PRIMARY";
		else
			statestr = "ONLINE";
		fprintf(fp, "%-16s ", statestr);
		pthread_mutex_lock(&sos->lock);
		sos_part_t p = __sos_container_part_find(sos, sos_part_name_get(part));
		if (p && p->obj_ods && (0 == ods_stat(p->obj_ods, &sb))) {
			char datestr[80];
			struct tm *tm;

			fprintf(fp, "%8s ", pretty_file_size(sb.st_size));

			tm = localtime(&sb.st_mtime);
			strftime(datestr, sizeof(datestr), "%Y/%m/%d %H:%M", tm);
			fprintf(fp, "%-16s ", datestr);

			tm = localtime(&sb.st_atime);
			strftime(datestr, sizeof(datestr), "%Y/%m/%d %H:%M", tm);
			fprintf(fp, "%-16s ", datestr);
		}
		pthread_mutex_unlock(&sos->lock);
		fprintf(fp, "\n");
		sos_part_put(part);
	}
	sos_part_iter_free(iter);
}

/**
 * \brief Allocate an object from the SOS object store.
 *
 * This call will automatically extend the size of the backing store
 * to accomodate the new object. This call will fail if there is
 * insufficient disk space. Use the sos_obj_index() to add the object
 * to all indices defined by it's object class.
 *
 * See the sos_schema_by_name() function call for information on how to
 * obtain a schema handle.
 *
 * \param schema	The schema handle
 * \returns Pointer to the new object
 * \returns NULL if there is an error
 */
sos_obj_t sos_obj_new(sos_schema_t schema)
{
	ods_obj_t ods_obj;
	sos_obj_t sos_obj;
	sos_part_t part;
	sos_obj_ref_t obj_ref;
	if (!schema->sos)
		return NULL;
	part = __sos_primary_obj_part(schema->sos);
	if (!part)
		return NULL;
	ods_obj = __sos_obj_new(part->obj_ods, schema->data->obj_sz,
				&schema->sos->lock);
	if (!ods_obj)
		goto err_0;
	memset(ods_obj->as.ptr, 0, schema->data->obj_sz);
	obj_ref.ref.ods = SOS_PART(part->part_obj)->part_id;
	obj_ref.ref.obj = ods_obj_ref(ods_obj);
	sos_obj = __sos_init_obj(schema->sos, schema, ods_obj, obj_ref);
	if (!sos_obj)
		goto err_1;
	return sos_obj;
 err_1:
	ods_obj_delete(ods_obj);
	ods_obj_put(ods_obj);
 err_0:
	return NULL;
}

static sos_obj_ref_t NULL_REF = {
	.ref = { 0, 0 }
};

sos_obj_ref_t sos_obj_ref(sos_obj_t obj)
{
	if (!obj)
		return NULL_REF;
	return obj->obj_ref;
}

ods_t __sos_ods_from_ref(sos_t sos, ods_ref_t ref)
{
	sos_part_t part;
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (ref == SOS_PART(part->part_obj)->part_id)
			goto found;
	}
	return NULL;
 found:
	return part->obj_ods;
}

static int ref_is_null(sos_obj_ref_t ref)
{
	return ((ref.ref.ods == 0) && (ref.ref.obj == 0));
}

/**
 * \brief  Return the object associated with the reference
 *
 * This function will return a sos_obj_t for the object that is referred
 * to by 'ref'. Use the function sos_obj_from_value() to obtain an
 * object from a sos_value_t value.
 *
 * \param sos The container handle
 * \param ref The object reference
 * \retval The object to which the reference refers.
 * \retval NULL The reference did not point to a well formed object, or the schema
 *              in the object header was not part of the container.
 */
sos_obj_t sos_ref_as_obj(sos_t sos, sos_obj_ref_t ref)
{
	ods_obj_t ods_obj;
	if (ref_is_null(ref))
		return NULL;

	ods_t ods = __sos_ods_from_ref(sos, ref.ref.ods);
	if (!ods)
		return NULL;

	ods_obj = ods_ref_as_obj(ods, ref.ref.obj);
	if (!ods_obj)
		return NULL;

	/* Get the schema id from the SOS object */
	sos_obj_data_t sos_obj = ods_obj->as.ptr;
	sos_schema_t schema = sos_schema_by_id(sos, sos_obj->schema);
	if (!schema)
		return NULL;

	return __sos_init_obj(sos, schema, ods_obj, ref);
}

/**
 * \brief  Return the object associated with the value
 *
 * This function will return a sos_obj_t for the object that is referred
 * to by ref_val. Use the function sos_ref_as_obj() to obtain an
 * object from a raw <tt>sos_obj_ref_t</tt> value.
 *
 * \param sos The container handle
 * \param ref_val A value handle to an attribute of type SOS_TYPE_OBJ
 * \retval The object to which the reference refers.
 * \retval NULL The reference did not point to a well formed object, or the schema
 *              in the object header was not part of the container.
 */
sos_obj_t sos_obj_from_value(sos_t sos, sos_value_t ref_val)
{
	if (sos_attr_type(ref_val->attr) != SOS_TYPE_OBJ)
		return NULL;

	if (ref_is_null(ref_val->data->prim.ref_))
		return NULL;

	return sos_ref_as_obj(sos, ref_val->data->prim.ref_);
}

/**
 * \brief Returns an Object's schema
 * \param obj The object handle
 * \retval The schema used to create the object.
 */
sos_schema_t sos_obj_schema(sos_obj_t obj)
{
	return sos_schema_get(obj->schema);
}

/**
 * \brief Release the storage consumed by the object in the SOS object store.
 *
 * Deletes the object and any arrays to which the object refers. It
 * does not delete an object that is referred to by this object,
 * i.e. SOS_TYPE_OBJ_REF attribute values.
 *
 * This function does not drop any references on the memory resources
 * for this object. Object references must still be dropped with the
 * sos_obj_put() function.
 *
 * \param obj	Pointer to the object
 */
void sos_obj_delete(sos_obj_t obj)
{
	sos_attr_t attr;
	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		struct sos_value_s v_;
		sos_value_t value;
		if (!sos_attr_is_array(attr))
			continue;
		value = sos_value_init(&v_, obj, attr);
		if (!value)
			continue;
		ods_t ods = __sos_ods_from_ref(obj->sos, value->data->prim.ref_.ref.ods);
		ods_ref_delete(ods, value->data->prim.ref_.ref.obj);
		sos_value_put(value);
	}
	ods_obj_delete(obj->obj);
	obj->obj = NULL;
}

/**
 * \brief Take a reference on an object
 *
 * SOS objects are reference counted. This function takes a reference
 * on a SOS object and returns a pointer to the object. The typical
 * calling sequence is:
 *
 *     sos_obj_t my_obj_ptr = sos_obj_get(obj);
 *
 * This allows for the object to be safely pointed to from multiple
 * places. The sos_obj_put() function is used to drop a reference on
 * a SOS object. For example:
 *
 *     sos_obj_put(my_obj_ptr);
 *     my_obj_ptr = NULL;
 *
 * \param obj	The SOS object handle
 * \retval The object handle
 */
sos_obj_t sos_obj_get(sos_obj_t obj)
{
	ods_atomic_inc(&obj->ref_count);
	return obj;
}

/**
 * \brief Drop a reference on an object
 *
 * SOS objects are reference counted. The memory consumed by the
 * object is not released until all references have been dropped. This
 * refers only to references in main memory. The object will continue
 * to exist in persistent storage. See the sos_obj_delete() function
 * for information on removing an object from persistent storage.
 *
 * \param obj	The object handle
 */
void sos_obj_put(sos_obj_t obj)
{
	if (obj && !ods_atomic_dec(&obj->ref_count)) {
		sos_t sos = obj->sos;
		sos_schema_t schema = obj->schema;
		ods_obj_put(obj->obj);
		pthread_mutex_lock(&sos->lock);
		LIST_INSERT_HEAD(&sos->obj_free_list, obj, entry);
		pthread_mutex_unlock(&sos->lock);
		sos_schema_put(schema);
	}
}

/**
 * \brief Remove an object from the SOS
 *
 * This removes an object from all indexes of which it is a
 * member. The object itself is not destroyed. Use the
 * sos_obj_delete() function to release the storage consumed by the
 * object itself.
 *
 * \param obj	Handle for the object to remove
 *
 * \returns 0 on success.
 * \returns Error code on error.
 */
int sos_obj_remove(sos_obj_t obj)
{
	sos_attr_t attr;
	size_t key_sz;
	int rc;
	ods_idx_data_t data;
	sos_key_t key;

	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		struct sos_value_s v_;
		sos_value_t value;
		if (!attr->data->indexed)
			continue;
		value = sos_value_init(&v_, obj, attr);
		key_sz = sos_value_size(value);
		key = sos_key_new(key_sz);
		if (!key) {
			sos_value_put(value);
			return ENOMEM;
		}
		ods_key_set(key, sos_value_as_key(value), key_sz);
		rc = ods_idx_delete(attr->index->idx, key, &data);
		sos_key_put(key);
		sos_value_put(value);
		if (rc)
			return rc;
	}

	return 0;
}

/**
 * \brief Add an object to it's indexes
 *
 * Add an object to all the indices defined in it's schema. This
 * function should only be called after all attributes that have
 * indexes have had their values set.
 *
  * \param obj	Handle for the object to add
 *
 * \retval 0	Success
 * \retval -1	An error occurred. Refer to errno for detail.
 */
int sos_obj_index(sos_obj_t obj)
{
	struct sos_value_s v_;
	sos_value_t value;
	sos_attr_t attr;
	size_t key_sz;
	SOS_KEY(key);
	int rc;

	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		if (!attr->data->indexed)
			continue;
		value = sos_value_init(&v_, obj, attr);
		key_sz = sos_value_size(value);
		sos_key_set(key, sos_value_as_key(value), key_sz);
		rc = sos_index_insert(attr->index, key, obj);
		sos_value_put(value);
		if (rc)
			goto err;
	}
	return 0;
 err:
	return rc;
}

/**
 * \brief Set an object attribute's value from a string
 *
 * This convenience function set's an object's attribute value specified as a
 * string. The attribute to set is specified by name.
 *
 * For example:
 *
 *     int rc = sos_obj_attr_by_name_from_str(an_obj, "my_int_attr", "1234");
 *     if (!rc)
 *        printf("Success!!\n");
 *
 * See the sos_obj_attr_from_str() function to set the value with a string if
 * the attribute handle is known.
 *
 * \param sos_obj	The object handle
 * \param attr_name	The attribute name
 * \param attr_value	The attribute value as a string
 * \param endptr Receives the point in the str argumeent where parsing stopped.
 *               This parameter may be NULL.
 * \retval 0 Success
 * \retval EINVAL The string format was invalid for the attribute type
 * \retval ENOSYS There is no string formatter for this attribute type
 */
int sos_obj_attr_by_name_from_str(sos_obj_t sos_obj,
				  const char *attr_name, const char *attr_value,
				  char **endptr)
{
	sos_attr_t attr;
	attr = sos_schema_attr_by_name(sos_obj->schema, attr_name);
	if (!attr)
		return ENOENT;

	return sos_obj_attr_from_str(sos_obj, attr, attr_value, endptr);
}

char *sos_obj_attr_by_name_to_str(sos_obj_t sos_obj,
				  const char *attr_name, char *str, size_t len)
{
	sos_attr_t attr;
	attr = sos_schema_attr_by_name(sos_obj->schema, attr_name);
	if (!attr)
		return NULL;

	return sos_obj_attr_to_str(sos_obj, attr, str, len);
}

static void __attribute__ ((constructor)) sos_lib_init(void)
{
	LIST_INIT(&cont_list);
	pthread_mutex_init(&cont_list_lock, NULL);
}

static void __attribute__ ((destructor)) sos_lib_term(void)
{
	while (!LIST_EMPTY(&cont_list)) {
		sos_t sos = LIST_FIRST(&cont_list);
		LIST_REMOVE(sos, entry);
		sos_container_close(sos, SOS_COMMIT_SYNC);
	}
}

/** @} */

