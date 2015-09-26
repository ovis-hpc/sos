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
 * \mainpage Introduction
 *
 * The Scalable Object Storage (SOS) is a high performance storage
 * engine designed to efficiently manage structured data in persistent
 * media.
 *
 * \section cont Container
 *
 * An instance of a SOS Object Store is called a Container. A
 * Container is identified by a name that looks like a POSIX Filesytem
 * path. This allows Containers to be organized into a hierarchical
 * name space. This is a convenience and does not mean that a
 * Container is necessarily stored in a Filesytem.
 *
 * See \ref container_overview for more information.

 * \section partition Partition
 *
 * In order to facilitate management of the storage consumed by a
 * Container, a Container is divided up into one or more
 * Partitions. Paritions contain the objects that are created in the
 * Container. The purpose of a Partition is to allow subsets of a
 * Container's objects to be migrated from primary storage to secondary
 * storage.
 *
 * See \ref partition_overview for more information
 *
 * \section schema Schema
 *
 * Schemas define the format of Objects and are logically an Object's
 * "type." There can be any number of Schema in the Container such
 * that a single Container may contain Objects of many different
 * types. The Container has a directory of Schemas. When Objects are
 * created, the Schema handle is specified to inform the object store
 * of the size and format of the object and whether or not one or more
 * of it's attributes has an Index.
 *
 * See \ref schema_overview for more information.
 *
 * \section object Object
 *
 * An Object is a collection of Attributes. An Attribute has a Name
 * and a Type. There are built-in types for an Attribute and
 * user-defined types. The built in types include the familiar
 * <tt>int</tt>, <tt>long</tt>, <tt>double</tt> types as well as
 * arrays of these types. A special Attribute type is
 * <tt>SOS_TYPE_OBJ</tt>, which is a <tt>Reference</tt> to another
 * Object. This allows complex data structures like linked lists to be
 * implemented in the Container.
 *
 * The user-defined types are Objects.
 *
 * \section sindex Index
 *
 * An Index is an ordered collection for for quickly finding an Object
 * in a container based on a key. Indexes can be associated with a
 * Schema or be independent of a particular Schema, for example,
 * allowing a single Index to refer to objects of different types. If
 * an Index is associated with a Schema Attribute, all management and
 * insertion is handled automatically by the sos_obj_index() function.
 *
 * Indexes that are not directly related to a Schema are managed
 * directly by the application; including the creation of keys, and
 * insertion of Objects into the index.
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

#ifndef DOXYGEN
LIST_HEAD(cont_list_head, sos_container_s) cont_list;
#endif
pthread_mutex_t cont_list_lock;

/**
 * \page container_overview Containers
 *
 * A SOS Container groups Partitions, Schema, Objects, and Indices
 * together into a single namespace. The root of the namespace is the
 * Container's name. Containers are created with the
 * sos_container_new() function. SOS implements the POSIX security
 * model. When a Container is created, it inherits the owner and group
 * of the process that created the container. The sos_container_new()
 * function takes an <tt>o_mode</tt> parameter that identifies the
 * standard POSIX umask to specify RO/RW access for owner/group/other.
 *
 * The sos_container_open() function opens a previously created
 * container. The user/group of the process opening the container must
 * have adequate permission to provide the requested R/W access. The
 * sos_container_open() function returns a <tt>sos_t</tt> container handle that
 * is used in subsequent SOS API.
 *
 * Changes to a container are opportunistically commited to stable
 * storage. An application can initiate a commit to storage with the
 * sos_container_commit() function.
 *
 * The SOS Container API include the following:
 *
 * - sos_container_new() Create a new Container
 * - sos_container_open() Open a previously created Container
 * - sos_container_close() Close a container
 * - sos_container_commit() Commit a Container's contents to stable storage
 * - sos_container_delete() Destroy a Container and all of it's contents
 * - sos_container_info() - Print Container information to a FILE pointer
 */

/* This function effectively implements 'mkdir -p' */
int __sos_make_all_dir(const char *inp_path, mode_t omode)
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

	rc = __sos_make_all_dir(path, x_mode);
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
	SOS_PART_UDATA(udata)->gen = 1;

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

	/* Create the index to look up the indexes */
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

/**
 * \brief Delete storage associated with a Container
 *
 * Removes all resources associated with the Container. The sos_t
 * handle must be provided (requiring an open) because it is necessary
 * to know the associated Indices in order to be able to know the
 * names of the associated files. sos_destroy() will also close \c sos, as the
 * files should be closed before begin removed.
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
	if (!fp)
		fp = stdout;
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

	if (strlen(path) >= SOS_PART_PATH_LEN) {
		errno = E2BIG;
		return NULL;
	}
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
 * \brief Close a Container
 *
 * This function commits changes to stable storage and releases all
 * in-memory resources associated with the container.
 *
 * If SOS_COMMIT_SYNC is specified in the flags parameter, the
 * function will wait until the changes are commited before returning.
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
	sos_obj->schema = schema;
	sos_obj->ref_count = 1;

	return sos_obj;
}

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

/* Must be called with sos->lock held */
sos_part_t __sos_container_part_find(sos_t sos, const char *name)
{
	sos_part_t part = NULL;
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (0 == strcmp(name, SOS_PART(part->part_obj)->name))
			goto out;
	}
 out:
	return part;
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
	return obj->schema;
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
		ods_obj_put(obj->obj);
		pthread_mutex_lock(&sos->lock);
		LIST_INSERT_HEAD(&sos->obj_free_list, obj, entry);
		pthread_mutex_unlock(&sos->lock);
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

