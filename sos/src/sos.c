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
 * NB: These may be deprecated
 * - sos_container_get() - Take a counted reference on the Container
 * - sos_container_put() - Put a counted reference on the Container
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
static time_t midnight_timestamp(void)
{
	time_t now = time(NULL);
	struct tm ctm, *ptm;
	ptm = localtime_r(&now, &ctm);
	ptm->tm_sec = ptm->tm_min = ptm->tm_hour = 0;
	now = mktime(ptm);
	return now;
}

static time_t partition_timestamp(sos_t sos)
{
	time_t ts;
	uint32_t period = sos->config.partition_period;
	if ((sos->config.options &= SOS_OPTIONS_PARTITION_ENABLE)
	    && (0 != sos->config.partition_period)) {
		ts = time(NULL);
		ts = ts - (ts % period);
	} else
		ts = 0;
	return ts;
}

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

	/*
	 * The container contains one or more partitions. Each
	 * partition is named with a timestamp. At create time, there
	 * is a single partition in the container named at midnight on
	 * the day it was created.
	 */
	rc = make_all_dir(path, x_mode);
	if (rc) {
		rc = errno;
		goto err_0;
	}

	/* Create the ODS to contain configuration objects */
	sprintf(tmp_path, "%s/config", path);
	rc = ods_create(tmp_path, o_mode);
	if (rc)
		goto err_1;

	/* Create the configuration object index */
	sprintf(tmp_path, "%s/config_idx", path);
	rc = ods_idx_create(tmp_path, o_mode, "BXTREE", "STRING", 5);
	if (rc)
		goto err_2;

	/* Create the ODS to contain partition objects */
	sprintf(tmp_path, "%s/part", path);
	rc = ods_create(tmp_path, o_mode);
	if (rc)
		goto err_1;

	/* Create the partition object index */
	sprintf(tmp_path, "%s/part_idx", path);
	rc = ods_idx_create(tmp_path, o_mode, "BXTREE", "STRING", 5);
	if (rc)
		goto err_2;

	/* Default configuration is to disable partition rotation */
	rc = sos_container_config(path, "PARTITION_ENABLE", "false");
	if (rc)
		goto err_2;
	rc = sos_part_new(path, SOS_PART_NAME_DEFAULT,
			  SOS_PART_STATE_ACTIVE | SOS_PART_STATE_PRIMARY);

 	/* Create the ODS to contain the schema objects */
	sprintf(tmp_path, "%s/schemas", path);
 	rc = ods_create(tmp_path, o_mode);
 	if (rc)
 		goto err_3;

	ods_t schema_ods = ods_open(tmp_path, O_RDWR);
	if (!schema_ods)
		goto err_4;
	/* Initialize the schema dictionary */
	ods_obj_t udata = ods_get_user_data(schema_ods);
	if (!udata) {
		rc = errno;
		goto err_5;
	}
	SOS_UDATA(udata)->signature = SOS_SCHEMA_SIGNATURE;
	SOS_UDATA(udata)->version = SOS_LATEST_VERSION;
	SOS_UDATA(udata)->last_schema_id = SOS_SCHEMA_FIRST_USER;
	ods_obj_put(udata);

	/* Create the index to look up the schema names */
	sprintf(tmp_path, "%s/schema_idx", path);
 	rc = ods_idx_create(tmp_path, o_mode, "BXTREE", "STRING", 5);
 	if (rc)
 		goto err_5;

	return 0;

 err_5:
	ods_close(schema_ods, ODS_COMMIT_SYNC);
 err_4:
	sprintf(tmp_path, "%s/schemas", path);
	ods_destroy(tmp_path);
 err_3:
	sprintf(tmp_path, "%s/config_idx", path);
	ods_destroy(tmp_path);
 err_2:
	sprintf(tmp_path, "%s/config", path);
	ods_destroy(tmp_path);
 err_1:
	rmdir(path);
 err_0:
	return rc;
}

static int __sos_partition_open(sos_t sos)
{
	char tmp_path[PATH_MAX];
	int rc;
	int o_mode;
	struct stat sb;
	time_t timestamp;
	ods_obj_t part_obj = NULL;
	sos_part_t part;
	ods_t ods;
	ods_iter_t part_iter;

	/* Stat the container path to get the file mode bits */
	rc = stat(sos->path, &sb);
	if (rc)
		goto err_0;

	/* Open the partition ODS */
	sprintf(tmp_path, "%s/part", sos->path);
	sos->part_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!sos->part_ods)
		goto err_0;

	/* Open the partition object index */
	sprintf(tmp_path, "%s/part_idx", sos->path);
	sos->part_idx = ods_idx_open(tmp_path, ODS_PERM_RW);
	if (!sos->part_idx)
		goto err_1;

	part_iter = ods_iter_new(sos->part_idx);
	if (!part_iter)
		goto err_2;

	for (rc = ods_iter_begin(part_iter); !rc; rc = ods_iter_next(part_iter)) {
		ods_ref_t obj_ref = ods_iter_ref(part_iter);
		part_obj = ods_ref_as_obj(sos->part_ods, obj_ref);
		part = SOS_PART(part_obj);
		if (0 == (part->state & SOS_PART_STATE_ACTIVE))
			goto skip;

		sprintf(tmp_path, "%s/%s", sos->path, part->name);
		rc = make_all_dir(tmp_path, sb.st_mode);
		if (rc) {
			rc = errno;
			goto err_3;
		}
		o_mode = sb.st_mode;
		o_mode &= ~S_IXGRP;
		o_mode &= ~S_IXUSR;
		o_mode &= ~S_IXOTH;
		sos->o_mode = o_mode;

		sprintf(tmp_path, "%s/%s/objects", sos->path, part->name);
	retry:
		ods = ods_open(tmp_path, sos->o_perm);
		if (!ods) {
			/* Create the ODS to contain the objects */
			rc = ods_create(tmp_path, o_mode);
			if (rc)
				goto err_3;
			goto retry;
		}
		sos_obj_part_t obj_part = calloc(1, sizeof *obj_part);
		obj_part->part_obj = ods_obj_get(part_obj);
		obj_part->obj_ods = ods;
		TAILQ_INSERT_TAIL(&sos->ods_list, obj_part, entry);
		/*
		 * Iterate through all the schemas and open/create the indices.
		 */
		sos_schema_t schema;
		LIST_FOREACH(schema, &sos->schema_list, entry) {
			rc = __sos_schema_open(sos, schema, part_obj);
			if (rc)
				goto err_3;
		}
	skip:
		ods_obj_put(part_obj);
	}
	return 0;
 err_3:
	ods_obj_put(part_obj);
 err_2:
	ods_idx_close(sos->part_idx, ODS_COMMIT_ASYNC);
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
	sos_idx_part_t part;
	TAILQ_FOREACH(part, &index->idx_list, entry)
		if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_ACTIVE)
			ods_idx_commit(part->index, flags);
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
	sos_obj_part_t part;
	TAILQ_FOREACH(part, &sos->ods_list, entry)
		if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_ACTIVE)
			ods_commit(part->obj_ods, commit);

	/* Commit all the attribute indices */
	LIST_FOREACH(schema, &sos->schema_list, entry) {
		TAILQ_FOREACH(attr, &schema->attr_list, entry) {
			sos_idx_part_t part;
			if (attr->index)
				sos_index_commit(attr->index, commit);
		}
	}
	return 0;
}

void sos_index_info(sos_index_t index, FILE *fp)
{
	sos_idx_part_t part;
	TAILQ_FOREACH(part, &index->idx_list, entry) {
		if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_PRIMARY) {
			ods_idx_info(part->index, fp);
			ods_info(ods_idx_ods(part->index), fp);
			break;
		}
	}
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
	sos_obj_part_t part;
	TAILQ_FOREACH(part, &sos->ods_list, entry)
		ods_info(part->obj_ods, fp);
}

static void free_sos(sos_t sos, sos_commit_t flags)
{
	struct rbn *rbn;
	sos_obj_t obj;

	assert(sos->ref_count == 0);

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
	sos_obj_part_t part;
	TAILQ_FOREACH(part, &sos->ods_list, entry)
		ods_close(part->obj_ods, flags);
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
 * \brief Take a reference on a container
 *
 * SOS container are reference counted. This function takes a reference
 * on a SOS container and returns a pointer to the same. The typical
 * calling sequence is:
 *
 *     sos_t my_sos_ptr = sos_container_get(sos);
 *
 * This allows for the container to be safely pointed to from multiple
 * places. The sos_container_put() function is used to drop a reference on
 * the container. For example:
 *
 *     sos_container_put(my_sos_ptr);
 *     my_sos_ptr = NULL;
 *
 * \param sos	The SOS container handle
 * \retval The container handle
 */
static sos_t sos_container_get(sos_t sos)
{
	ods_atomic_inc(&sos->ref_count);
	return sos;
}

/**
 * \brief Drop a reference on a container
 *
 * The memory consumed by the container is not released until all
 * references have been dropped. This refers only to references in
 * main memory.
 *
 * \param sos	The container handle
 */
static void sos_container_put(sos_t sos)
{
	if (sos && !ods_atomic_dec(&sos->ref_count)) {
		free_sos(sos, SOS_COMMIT_ASYNC);
	}
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
	int rc;

	sos = calloc(1, sizeof(*sos));
	if (!sos) {
		errno = ENOMEM;
		return NULL;
	}
	pthread_mutex_init(&sos->lock, NULL);
	sos->ref_count = 1;
	LIST_INIT(&sos->obj_list);
	LIST_INIT(&sos->obj_free_list);
	TAILQ_INIT(&sos->ods_list);

	sos->path = strdup(path);
	sos->o_perm = (ods_perm_t)o_perm;
	rbt_init(&sos->schema_name_rbt, __sos_schema_name_cmp);
	rbt_init(&sos->schema_id_rbt, schema_id_cmp);
	sos->schema_count = 0;

	rc = __sos_config_init(sos);
	if (rc)
		goto err;

	/* Open the ODS containing the schema objects */
	sprintf(tmp_path, "%s/schemas", path);
	sos->schema_ods = ods_open(tmp_path, sos->o_perm);
	if (!sos->schema_ods)
		goto err;
	ods_obj_t udata = ods_get_user_data(sos->schema_ods);
	if ((SOS_UDATA(udata)->signature != SOS_SCHEMA_SIGNATURE)
	    || (SOS_UDATA(udata)->version != SOS_LATEST_VERSION)) {
		errno = EINVAL;
		ods_obj_put(udata);
		goto err;
	}
	ods_obj_put(udata);

	/* Open the index on the schema objects */
	sprintf(tmp_path, "%s/schema_idx", path);
	sos->schema_idx = ods_idx_open(tmp_path, sos->o_perm);
	if (!sos->schema_idx)
		goto err;

	/*
	 * Build the schema dictionary
	 */
	ods_iter_t iter = ods_iter_new(sos->schema_idx);
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_next(iter)) {
		ods_obj_t obj_ref = ods_iter_obj(iter);
		ods_obj_t schema_obj =
			ods_ref_as_obj(sos->schema_ods,
				       SOS_OBJ_REF(obj_ref)->obj_ref);
		ods_obj_put(obj_ref);
		sos_schema_t schema = __sos_schema_init(sos, schema_obj);
		if (!schema)
			goto err;
		sos->schema_count++;
		LIST_INSERT_HEAD(&sos->schema_list, schema, entry);
	}
	ods_iter_delete(iter);

	/*
	 * Open the partitions
	 */
	rc = __sos_partition_open(sos);
	if (rc)
		goto err;
	return sos;
 err:
	sos_container_put(sos);
	return NULL;
}

int sos_container_stat(sos_t sos, struct stat *sb)
{
	sos_obj_part_t part = TAILQ_FIRST(&sos->ods_list);
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
	sos_obj_part_t part;
	TAILQ_FOREACH(part, &sos->ods_list, entry) {
		if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_PRIMARY)
			return ods_extend(part->obj_ods, new_size);
	}
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
	sos_container_put(sos);
}

sos_obj_t __sos_init_obj(sos_t sos, sos_schema_t schema, ods_obj_t ods_obj, sos_obj_part_t part)
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
	sos_obj->part = part;
	ods_atomic_inc(&schema->data->ref_count);
	sos_obj->schema = sos_schema_get(schema);
	sos_obj->ref_count = 1;

	return sos_obj;
}

sos_obj_part_t __sos_active_obj_part(sos_t sos)
{
	sos_obj_part_t part;

	TAILQ_FOREACH(part, &sos->ods_list, entry) {
		if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_ACTIVE)
			return part;
	}
	return NULL;
}

sos_obj_part_t __sos_primary_obj_part(sos_t sos)
{
	sos_obj_part_t part;

	if ((NULL != sos->primary_part) &&
	    (SOS_PART(sos->primary_part->part_obj)->state & SOS_PART_STATE_PRIMARY))
		return sos->primary_part;

	TAILQ_FOREACH(part, &sos->ods_list, entry) {
		if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_PRIMARY) {
			sos->primary_part = part;
			return part;
		}
	}
	return NULL;
}

sos_idx_part_t __sos_active_idx_part(sos_index_t index)
{
	sos_idx_part_t part;

	if ((NULL != index->last_part) &&
	    (SOS_PART(index->last_part->part_obj)->state & SOS_PART_STATE_ACTIVE))
		return index->last_part;

	TAILQ_FOREACH(part, &index->idx_list, entry) {
		if (SOS_PART(part->part_obj)->state & SOS_PART_STATE_ACTIVE) {
			index->last_part = part;
			return part;
		}
	}
	return NULL;
}

sos_idx_part_t __sos_matching_idx_part(sos_index_t index, sos_obj_part_t obj_part)
{
	sos_idx_part_t part;

	if ((NULL != index->last_part) &&
	    (SOS_PART(index->last_part->part_obj) == SOS_PART(obj_part->part_obj)))
		return index->last_part;

	TAILQ_FOREACH(part, &index->idx_list, entry) {
		if (SOS_PART(part->part_obj) == SOS_PART(obj_part->part_obj)) {
			index->last_part = part;
			return index->last_part;
		}
	}
	return NULL;
}

int sos_part_new(const char *path, const char *part_name, int state_flags)
{
	int rc;
	char tmp_path[PATH_MAX];
	ods_idx_t idx;
	ods_t ods;
	ods_ref_t ref;
	ods_obj_t obj;
	sos_part_t part;
	ods_key_t part_key;
	size_t key_sz;
	SOS_KEY(key);

	/* Open the partition ODS */
	sprintf(tmp_path, "%s/part", path);
	ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!ods)
		goto err_0;

	/* Open the partition object index */
	sprintf(tmp_path, "%s/part_idx", path);
	idx = ods_idx_open(tmp_path, ODS_PERM_RW);
	if (!idx)
		goto err_1;
	key_sz = strlen(part_name)+1;
	ods_key_set(key, part_name, key_sz);
	rc = ods_idx_find(idx, key, &ref);
	if (rc) {
		/* Create the object */
		obj = ods_obj_alloc(ods, sizeof(*part));
		if (!obj)
			goto err_2;
		part_key = ods_key_alloc(idx, key_sz);
		if (!part_key)
			goto err_3;
		ods_key_set(part_key, part_name, key_sz);
		rc = ods_idx_insert(idx, part_key, ods_obj_ref(obj));
		if (rc)
			goto err_4;
	}
	part = SOS_PART(obj);
	strcpy(part->name, part_name);
	part->state = state_flags;
	ods_obj_put(obj);
	ods_obj_put(part_key);
	ods_close(ods, ODS_COMMIT_SYNC);
	ods_idx_close(idx, ODS_COMMIT_SYNC);
	return 0;
 err_4:
	ods_obj_delete(part_key);
	ods_obj_put(part_key);
 err_3:
	ods_obj_delete(obj);
	ods_obj_put(obj);
 err_2:
	ods_idx_close(idx, ODS_COMMIT_ASYNC);
 err_1:
	ods_close(ods, ODS_COMMIT_ASYNC);
 err_0:
	return errno;
}

sos_part_iter_t __sos_part_iter_new(sos_t sos)
{
	int rc;
	char tmp_path[PATH_MAX];
	sos_part_iter_t iter = calloc(1, sizeof *iter);
	if (!iter)
		return NULL;

	/* Open the partition ODS */
	sprintf(tmp_path, "%s/part", sos->path);
	iter->part_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!iter->part_ods)
		goto err_0;

	/* Open the partition object index */
	sprintf(tmp_path, "%s/part_idx", sos->path);
	iter->part_idx = ods_idx_open(tmp_path, ODS_PERM_RW);
	if (!iter->part_idx)
		goto err_1;

	iter->iter = ods_iter_new(iter->part_idx);
	if (!iter->iter)
		goto err_2;
	return iter;
 err_2:
	ods_idx_close(iter->part_idx, ODS_COMMIT_ASYNC);
 err_1:
	ods_close(iter->part_ods, ODS_COMMIT_ASYNC);
 err_0:
	return NULL;
}

void __sos_part_iter_free(sos_part_iter_t iter)
{
	if (iter->obj)
		ods_obj_put(iter->obj);
	ods_iter_delete(iter->iter);
	ods_idx_close(iter->part_idx, ODS_COMMIT_ASYNC);
	ods_close(iter->part_ods, ODS_COMMIT_ASYNC);
	free(iter);
}

ods_obj_t __sos_part_first(sos_part_iter_t iter)
{
	int rc;
	if (iter->obj) {
		ods_obj_put(iter->obj);
		iter->obj = NULL;
	}
	rc = ods_iter_begin(iter->iter);
	if (rc)
		return NULL;
	ods_ref_t obj_ref = ods_iter_ref(iter->iter);
	if (!obj_ref)
		return NULL;
	iter->obj = ods_ref_as_obj(iter->part_ods, obj_ref);
	return iter->obj;
}

ods_obj_t __sos_part_next(sos_part_iter_t iter)
{
	int rc;
	if (iter->obj) {
		ods_obj_put(iter->obj);
		iter->obj = NULL;
	}
	rc = ods_iter_next(iter->iter);
	if (rc)
		return NULL;
	ods_ref_t obj_ref = ods_iter_ref(iter->iter);
	if (!obj_ref)
		return NULL;
	iter->obj = ods_ref_as_obj(iter->part_ods, obj_ref);
	return iter->obj;
}

void sos_part_print(sos_t sos, FILE *fp)
{
	ods_obj_t part_obj;
	sos_part_t part;
	sos_part_iter_t iter = __sos_part_iter_new(sos);
	if (!iter)
		return;
	printf("%-32s %s\n", "Partition Name", "Status");
	printf("-------------------------------- --------------------------------\n");
	for (part_obj = __sos_part_first(iter); part_obj; part_obj = __sos_part_next(iter)) {
		fprintf(fp, "%-32s ", SOS_PART(part_obj)->name);
		if (!SOS_PART(part_obj)->state) {
			printf("OFFLINE\n");
			continue;
		}
		if (SOS_PART(part_obj)->state & SOS_PART_STATE_ACTIVE)
			printf("ACTIVE ");
		if (SOS_PART(part_obj)->state & SOS_PART_STATE_PRIMARY)
			printf("PRIMARY ");
		printf("\n");
	}
	__sos_part_iter_free(iter);
}

sos_obj_t sos_obj_new(sos_schema_t schema)
{
	ods_obj_t ods_obj;
	sos_obj_t sos_obj;
	sos_obj_part_t part;
	if (!schema->sos)
		return NULL;
	part = __sos_primary_obj_part(schema->sos);
	if (!part)
		return NULL;
	ods_obj = __sos_obj_new(part->obj_ods, schema->data->obj_sz,
				&schema->sos->lock);
	if (!ods_obj)
		goto err_0;
	sos_obj = __sos_init_obj(schema->sos, schema, ods_obj, part);
	if (!sos_obj)
		goto err_1;
	return sos_obj;
 err_1:
	ods_obj_delete(ods_obj);
	ods_obj_put(ods_obj);
 err_0:
	return NULL;
}

sos_ref_t sos_obj_ref(sos_obj_t obj)
{
	sos_ref_t ref = { NULL, 0 };
	if (obj->obj) {
		ref.part = obj->part;
		ref.ref = ods_obj_ref(obj->obj);
	}
	return ref;
}

sos_obj_t sos_obj_from_ref(sos_t sos, sos_ref_t ref)
{
	ods_obj_t ods_obj;
	if (SOS_NULL_REF(ref))
		return NULL;

	ods_obj = ods_ref_as_obj(((sos_obj_part_t)ref.part)->obj_ods, ref.ref);
	if (!ods_obj)
		return NULL;

	/* Get the schema id from the SOS object */
	sos_obj_data_t sos_obj = ods_obj->as.ptr;
	sos_schema_t schema = sos_schema_by_id(sos, sos_obj->schema);
	if (!schema)
		return NULL;

	return __sos_init_obj(sos, schema, ods_obj, ref.part);
}

sos_obj_t sos_obj_from_value(sos_t sos, sos_value_t ref_val)
{
	sos_ref_t ref;
	if (sos_attr_type(ref_val->attr) != SOS_TYPE_OBJ)
		return NULL;

	if (0 == ref_val->data->prim.ref_)
		return NULL;
	ref.part = ref_val->obj->part;
	ref.ref = ref_val->data->prim.ref_;
	return sos_obj_from_ref(sos, ref);
}

sos_schema_t sos_obj_schema(sos_obj_t obj)
{
	return sos_schema_get(obj->schema);
}

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
		ods_ref_delete(obj->obj->ods, value->data->prim.ref_);
		sos_value_put(value);
	}
	ods_obj_delete(obj->obj);
	obj->obj = NULL;
}

/** @} */



sos_obj_t sos_obj_get(sos_obj_t obj)
{
	ods_atomic_inc(&obj->ref_count);
	return obj;
}

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

int sos_obj_remove(sos_obj_t obj)
{
	sos_value_t value;
	sos_attr_t attr;
	size_t key_sz;
	int rc;
	ods_ref_t ref;
	sos_key_t key;

	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		sos_key_t mkey;
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
		sos_idx_part_t part = __sos_matching_idx_part(attr->index, obj->part);
		assert(part);
		rc = ods_idx_delete(part->index, key, &ref);
		sos_key_put(key);
		sos_value_put(value);
		if (rc)
			return rc;
	}

	return 0;
}

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
		sos_idx_part_t part = __sos_matching_idx_part(attr->index, obj->part);
		assert(part);
		value = sos_value_init(&v_, obj, attr);
		key_sz = sos_value_size(value);
		ods_key_set(key, sos_value_as_key(value), key_sz);
		rc = ods_idx_insert(part->index, key, ods_obj_ref(obj->obj));
		sos_value_put(value);
		if (rc)
			goto err;
	}
	return 0;
 err:
	ods_obj_put(key);
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
	sos_value_t value;
	sos_attr_t attr;
	attr = sos_schema_attr_by_name(sos_obj->schema, attr_name);
	if (!attr)
		return ENOENT;

	return sos_obj_attr_from_str(sos_obj, attr, attr_value, endptr);
}

char *sos_obj_attr_by_name_to_str(sos_obj_t sos_obj,
				  const char *attr_name, char *str, size_t len)
{
	sos_value_t value;
	sos_attr_t attr;
	attr = sos_schema_attr_by_name(sos_obj->schema, attr_name);
	if (!attr)
		return NULL;

	return sos_obj_attr_to_str(sos_obj, attr, str, len);
}

