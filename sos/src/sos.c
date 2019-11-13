/*
 * Copyright (c) 2012-2017 Open Grid Computing, Inc. All rights reserved.
 * Copyright (c) 2012-2017 Sandia Corporation. All rights reserved.
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
 * engine designed to efficiently manage structured data on persistent
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

 * \section part Partition
 *
 * In order to facilitate management of the storage consumed by a
 * Container, a Container is divided up into one or more
 * Partitions. Paritions contain the objects that are created in the
 * Container. The purpose of a Partition is to allow subsets of a
 * Container's objects to be migrated from primary storage to secondary
 * storage.
 *
 * See the \ref partitions section for more information
 *
 * \section schema Schema
 *
 * Schemas define the format of Objects and are logically an Object's
 * "type." There can be any number of Schema in the Container such
 * that a single Container may contain Objects of many different
 * types. The Container has a directory of Schemas. When Objects are
 * created, the Schema handle is specified to inform the object store
 * of the size and format of the object and whether or not one or more
 * of its attributes has an Index.
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
 * \section index Index
 *
 * An Index is a named, ordered collection of Keyy/Value references to
 * Objects. Their purpose is to quickly find an Object in a container
 * based on a Key. Indexes can be associated with a Schema or be
 * independent of a Schema, for example, allowing a single Index to
 * refer to objects of different types. If an Index is associated with
 * a Schema Attribute, all management and insertion is handled
 * automatically by the sos_obj_index() function.
 *
 * Indexes that are not part of a Schema, i.e. not associated with an
 * indexed attribute, are managed by the application; including the
 * creation of keys, and insertion of Objects into the index.
 *
 * See the \ref indices section for more information.
 */

#define _GNU_SOURCE
#include <sys/queue.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/file.h>
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
#include <ftw.h>

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
 * - sos_container_delete() Destroy a Container and all of its contents
 * - sos_container_move() Reset a Container's path after it has been copied to a new location
 * - sos_container_info() - Print Container information to a FILE pointer
 * - sos_container_lock_info() - Print Container lock information to a FILE pointer
 * - sos_container_lock_cleanup() - Release locks held by no process
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

static int __create_pos_info(char *tmp_path, const char *path, int o_mode)
{
	int rc;
	ods_obj_t udata;
	ods_t pos_ods;

	/* Create the ODS to contain the position objects */
	sprintf(tmp_path, "%s/.__pos", path);
 	rc = ods_create(tmp_path, o_mode);
 	if (rc)
		goto err_0;

	pos_ods = ods_open(tmp_path, O_RDWR);
	if (!pos_ods)
		goto err_1;

	/* Initialize the index dictionary */
	udata = ods_get_user_data(pos_ods);
	if (!udata) {
		rc = errno;
		goto err_2;
	}

	SOS_POS_UDATA(udata)->signature = SOS_POS_SIGNATURE;
	SOS_POS_UDATA(udata)->lock = 0;
	ods_obj_put(udata);
	ods_close(pos_ods, ODS_COMMIT_ASYNC);

	/* Create the index to look up positions */
	sprintf(tmp_path, "%s/.__pos_idx", path);
 	rc = ods_idx_create(tmp_path, o_mode, "BXTREE", "UINT32", NULL);
 	if (rc)
 		goto err_2;
	return 0;

 err_2:
	ods_close(pos_ods, ODS_COMMIT_ASYNC);
 err_1:
	sprintf(tmp_path, "%s/.__pos", path);
	ods_destroy(tmp_path);
 err_0:
	return rc;
}

struct pos_ent_s {
	char name[SOS_INDEX_NAME_LEN];
	sos_pos_t pos;
	LIST_ENTRY(pos_ent_s) entry;
};

struct idx_ent_s {
	char name[SOS_INDEX_NAME_LEN];
	sos_index_t idx;
	sos_iter_t iter;
	struct rbn rbn;
};

static int __open_pos_info(sos_t sos, char *tmp_path, char *path)
{
	int rc = 0;
	sprintf(tmp_path, "%s/.__pos", path);
	sos->pos_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!sos->pos_ods) {
		int rc = __create_pos_info(tmp_path, path, sos->o_mode);
		if (rc)
			goto err;
		sprintf(tmp_path, "%s/.__pos", path);
		sos->pos_ods = ods_open(tmp_path, sos->o_perm);
		if (!sos->pos_ods) {
			rc = errno;
			goto err;
		}
	}

	/* Get the position object user data */
	sos->pos_udata = ods_get_user_data(sos->pos_ods);
	if (SOS_POS_UDATA(sos->pos_udata)->signature != SOS_POS_SIGNATURE) {
		rc = EINVAL;
		goto err;
	}

	/* Open the index on the position objects */
	sprintf(tmp_path, "%s/.__pos_idx", path);
	sos->pos_idx = ods_idx_open(tmp_path, ODS_PERM_RW);
	if (!sos->pos_idx) {
		rc = errno;
		goto err;
	}
	rc = 0;
 err:
	return rc;
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
	char real_path[PATH_MAX];
	int rc;
	int x_mode;
	struct stat sb;
	int dfd = -1;

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
	path = realpath(path, real_path);
	if (!path) {
		rc = errno;
		goto err_1;
	}

	dfd = open(path, O_DIRECTORY | O_RDONLY);
	if (dfd < 0) {
		rc = errno;
		goto err_1;
	}
	rc = flock(dfd, LOCK_EX);
	if (rc)
		goto err_1;

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
	rc = sos_container_config_set(path, "POS_KEEP_TIME",
				      stringify(SOS_POS_KEEP_TIME_DEFAULT));
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
	udata = ods_get_user_data(idx_ods);
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

	/* Create the ODS to contain the position objects */
	rc = __create_pos_info(tmp_path, path, o_mode);
	if (rc)
		goto err_8;
	close(dfd);
	return 0;
 err_8:
	sprintf(tmp_path, "%s/.__index_idx", path);
	ods_destroy(tmp_path);
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
 	if (dfd >= 0)
 		close(dfd); /* The file descriptor will be unlocked automatically */
	rmdir(path);
	errno = rc;		/* rmdir will stomp errno */
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
		if (schema->state != SOS_SCHEMA_OPEN)
			continue;
		TAILQ_FOREACH(attr, &schema->attr_list, entry) {
			if (sos_attr_index(attr))
				sos_index_commit(sos_attr_index(attr), commit);
		}
	}
	return 0;
}

void sos_index_info(sos_index_t index, FILE *fp)
{
	ods_idx_info(index->idx, fp);
	ods_info(ods_idx_ods(index->idx), fp, ODS_ALL_INFO);
}

int sos_index_verify(sos_index_t index, FILE *fp)
{
	return ods_idx_verify(index->idx, fp);
}

int print_schema(struct rbn *n, void *fp_, int level)
{
	FILE *fp = fp_;
	sos_attr_t attr;

	sos_schema_t schema = container_of(n, struct sos_schema_s, name_rbn);
	sos_schema_print(schema, fp);

	TAILQ_FOREACH(attr, &schema->attr_list, entry) {
		if (sos_attr_index(attr))
			sos_index_info(sos_attr_index(attr), fp);
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
	ods_info(ods_idx_ods(sos->schema_idx), fp, ODS_ALL_INFO);
	ods_info(sos->schema_ods, fp, ODS_ALL_INFO);
	sos_part_t part;
	TAILQ_FOREACH(part, &sos->part_list, entry) {
		if (part->obj_ods)
			ods_info(part->obj_ods, fp, ODS_ALL_INFO);
	}
}

static int show_locks(const char *path, const struct stat *sb,
		      int typeflags, struct FTW *ftw)
{
	size_t len;
	char tmp_path[PATH_MAX];

	strncpy(tmp_path, path, PATH_MAX);
	len = strlen(tmp_path);
	if (strcmp(&tmp_path[len-3], ".PG"))
		return 0;
	/* strip the .PG, ods_lock_info will append it */
	tmp_path[len-3] = '\0';
	ods_lock_info(tmp_path, stdout);
	return 0;
}

static int release_locks(const char *path, const struct stat *sb,
			 int typeflags, struct FTW *ftw)
{
	size_t len;
	char tmp_path[PATH_MAX];

	strncpy(tmp_path, path, PATH_MAX);
	len = strlen(tmp_path);
	if (strcmp(&tmp_path[len-3], ".PG"))
		return 0;
	/* strip the .PG, ods_lock_info will append it */
	tmp_path[len-3] = '\0';
	ods_lock_cleanup(tmp_path);
	return 0;
}

/**
 * \brief Print container lock information
 *
 * Prints information about the locks held on the container to a FILE pointer.
 *
 * \param sos	The container handle
 * \param fp	The FILE pointer
 */
int sos_container_lock_info(const char *path, FILE *fp)
{
	int rc;
	rc = nftw(path, show_locks, 1024, FTW_DEPTH);
	return rc;
}

/**
 * \brief Release locks held by dead processes
 *
 * It is possible for a process to die while holding container
 * locks. This renders the container inaccessible. This function will
 * release locks held by dead processes.
 *
 * \param sos	The container handle
 */
int sos_container_lock_cleanup(const char *path)
{
	int rc;

	rc = nftw(path, release_locks, 1024, FTW_DEPTH|FTW_PHYS);
	return rc;
}

void sos_inuse_obj_info(sos_t sos, FILE *outp)
{
	sos_obj_t obj;
	if (!outp)
		outp = stdout;
	fprintf(outp, "Active Objects\n");
	fprintf(outp, "%-10s %-12s %s\n", "Ref Count", "Obj Ref", "Schema");
	fprintf(outp, "---------- ------------ ------------\n");
	LIST_FOREACH(obj, &sos->obj_list, entry) {
		char ref_str[32];
		sprintf(ref_str, "%p@%p", (void*)obj->obj_ref.ref.ods, (void*)obj->obj_ref.ref.obj);
		fprintf(outp, "%10d %-12s %s\n",
			obj->ref_count, ref_str, obj->schema->data->name);
	}
	fprintf(outp, "---------- ------------ ------------\n");
}

void sos_free_obj_info(sos_t sos, FILE *outp)
{
	sos_obj_t obj;
	if (!outp)
		outp = stdout;
	fprintf(outp, "Free Objects\n");
	fprintf(outp, "%-10s %-12s %s\n", "Ref Count", "Obj Ref", "Schema");
	fprintf(outp, "---------- ------------ ------------\n");
	LIST_FOREACH(obj, &sos->obj_free_list, entry) {
		char ref_str[32];
		sprintf(ref_str, "%p@%p", (void*)obj->obj_ref.ref.ods, (void*)obj->obj_ref.ref.obj);
		fprintf(outp, "%10d %-12s %s\n",
			obj->ref_count, ref_str, obj->schema->data->name);
	}
	fprintf(outp, "---------- ------------ ------------\n");
}

static void free_sos(sos_t sos, sos_commit_t flags)
{
	struct rbn *rbn;
	sos_obj_t obj;

	/* There should be no objects on the active list */
	if (!LIST_EMPTY(&sos->obj_list)) {
		sos_inuse_obj_info(sos, __ods_log_fp);
		LIST_FOREACH(obj, &sos->obj_list, entry) {
			obj->sos = NULL;
			obj->obj->ods = NULL;
		}
	}

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
	if (sos->idx_udata)
		ods_obj_put(sos->idx_udata);
	if (sos->idx_ods)
		ods_close(sos->idx_ods, flags);
	if (sos->pos_udata)
		ods_obj_put(sos->pos_udata);
	if (sos->pos_idx)
		ods_idx_close(sos->pos_idx, flags);
	if (sos->pos_ods)
		ods_close(sos->pos_ods, flags);
	sos_part_t part;
	while (!TAILQ_EMPTY(&sos->part_list)) {
		part = TAILQ_FIRST(&sos->part_list);
		TAILQ_REMOVE(&sos->part_list, part, entry);
		sos_part_put(part); /* the list reference */
	}
	if (sos->part_udata)
		ods_obj_put(sos->part_udata);
	if (sos->part_ods)
		ods_close(sos->part_ods, flags);
	pthread_mutex_destroy(&sos->lock);
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

/* Iterates over pos objects in the container and deletes objects that
 * are older than the lifetime threshold
 */
static void __pos_cleanup(sos_t sos)
{
	int rc;
	struct rbt index_rbt;
	struct rbn *rbn;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	struct pos_ent_s *pos_ent;
	struct idx_ent_s *idx_ent;

	/* Best effort clean-up any position objects older than an hour */
	ods_iter_t it = ods_iter_new(sos->pos_idx);
	if (!it)
		return;

	LIST_HEAD(pos_ent_list, pos_ent_s) pos_list =
		LIST_HEAD_INITIALIZER(pos_list);

	ods_lock(sos->pos_ods, 0, NULL);
	for (rc = ods_iter_begin(it); rc == 0; rc = ods_iter_next(it)) {
		ods_idx_data_t data;
		ods_obj_t pos_obj;
		data = ods_iter_data(it);
		pos_obj = ods_ref_as_obj(sos->pos_ods, data.uint64_[0]);
		if (pos_obj) {
			sos_debug("key %08x secs %d usecs %d\n",
				  SOS_POS(pos_obj)->key,
				  SOS_POS(pos_obj)->create_secs,
				  SOS_POS(pos_obj)->create_usecs
				  );
			if ((tv.tv_sec - SOS_POS(pos_obj)->create_secs)
			    > sos->config.pos_keep_time) {
				sos_debug("key %08x will be deleted\n", SOS_POS(pos_obj)->key);
				pos_ent = malloc(sizeof *pos_ent);
				if (!pos_ent) {
					sos_error("OOM @ %s:%d\n", __func__, __LINE__);
				} else {
					pos_ent->pos = SOS_POS(pos_obj)->key;
					strncpy(pos_ent->name, SOS_POS(pos_obj)->name,
						SOS_INDEX_NAME_LEN);
					LIST_INSERT_HEAD(&pos_list, pos_ent, entry);
				}
			}
			ods_obj_put(pos_obj);
		}
	}
	rbt_init(&index_rbt, __sos_schema_name_cmp);
	while (!LIST_EMPTY(&pos_list)) {
		pos_ent = LIST_FIRST(&pos_list);
		LIST_REMOVE(pos_ent, entry);
		/* See if we have the index open already */
		rbn = rbt_find(&index_rbt, pos_ent->name);
		if (!rbn) {
			idx_ent = malloc(sizeof *idx_ent);
			if (!idx_ent) {
				sos_error("OOM @ %s:%d\n", __func__, __LINE__);
				continue;
			}
			/* Get the index this pos entry is for */
			sos_index_t idx = sos_index_open(sos, pos_ent->name);
			if (!idx) {
				sos_error("The index %s for pos %08x could not be opened.\n",
					  pos_ent->name, pos_ent->pos);
				free(pos_ent);
				continue;
			}
			sos_iter_t iter = sos_index_iter_new(idx);
			if (!iter) {
				sos_error("An iterator for index %s could not be created, errno %d.\n",
					  pos_ent->name, errno);
				sos_index_close(idx, SOS_COMMIT_ASYNC);
				free(idx_ent);
				free(pos_ent);
				continue;
			}
			strcpy(idx_ent->name, pos_ent->name);
			idx_ent->idx = idx;
			idx_ent->iter = iter;
			rbn_init(&idx_ent->rbn, idx_ent->name);
			rbt_ins(&index_rbt, &idx_ent->rbn);
			rbn = &idx_ent->rbn;
		}
		idx_ent = container_of(rbn, struct idx_ent_s, rbn);
		/* Put the SOS position */
		sos_iter_pos_put_no_lock(idx_ent->iter, pos_ent->pos);
		free(pos_ent);
	}
	rbn = rbt_min(&index_rbt);
	while (rbn) {
		idx_ent = container_of(rbn, struct idx_ent_s, rbn);
		sos_iter_free(idx_ent->iter);
		sos_index_close(idx_ent->idx, SOS_COMMIT_ASYNC);
		rbt_del(&index_rbt, rbn);
		free(idx_ent);
		rbn = rbt_min(&index_rbt);
	}
	ods_iter_delete(it);
	ods_unlock(sos->pos_ods, 0);
}

struct sos_version_s sos_container_version(sos_t sos)
{
	struct sos_version_s vers;
	struct ods_version_s overs = ods_version(sos->schema_ods);
	vers.major = overs.major;
	vers.minor = overs.minor;
	vers.fix = overs.fix;
	vers.git_commit_id = overs.git_commit_id;
	return vers;
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
 */
sos_t sos_container_open(const char *path_arg, sos_perm_t o_perm)
{
	char tmp_path[PATH_MAX];
	char *path = NULL;
	sos_t sos;
	struct stat sb;
	ods_iter_t iter = NULL;
	int rc;
	int dfd = -1;

	if (strlen(path_arg) >= SOS_PART_PATH_LEN) {
		errno = E2BIG;
		return NULL;
	}
	if (path_arg[0] != '/') {
		if (!getcwd(tmp_path, sizeof(tmp_path)))
			return NULL;
		if (strlen(tmp_path) + strlen(path_arg) > SOS_PART_PATH_LEN) {
			errno = E2BIG;
			return NULL;
		}
		strcat(tmp_path, "/");
		strcat(tmp_path, path_arg);
		path = strdup(tmp_path);
	} else
		path = strdup(path_arg);
	if (!path) {
		errno = ENOMEM;
		return NULL;
	}
	sos = calloc(1, sizeof(*sos));
	if (!sos) {
		errno = ENOMEM;
		free(path);
		return NULL;
	}
	pthread_mutex_init(&sos->lock, NULL);
	LIST_INIT(&sos->obj_list);
	LIST_INIT(&sos->obj_free_list);
	TAILQ_INIT(&sos->part_list);

	/* Stat the container path to get the file mode bits */
	sos->path = path;
	rc = stat(sos->path, &sb);
	if (rc)
		goto err;

	sos->o_mode = sb.st_mode;
	sos->o_perm = (ods_perm_t)o_perm;
	rbt_init(&sos->schema_name_rbt, __sos_schema_name_cmp);
	rbt_init(&sos->schema_id_rbt, schema_id_cmp);
	sos->schema_count = 0;

	dfd = open(sos->path, O_DIRECTORY | O_RDONLY);
	if (dfd < 0) {
		rc = errno;
		goto err;
	}
	rc = flock(dfd, LOCK_SH);
	if (rc)
		goto err;

	rc = __sos_config_init(sos);
	if (rc) {
		sos_error("Error %d initializing configuration on open of %s\n", errno, path_arg);
		goto err;
	}

	/* Open the ODS containing the Index objects */
	sprintf(tmp_path, "%s/.__index", path);
	sos->idx_ods = ods_open(tmp_path, sos->o_perm);
	if (!sos->idx_ods)
		goto err;

	/* Get the index object user data */
	sos->idx_udata = ods_get_user_data(sos->idx_ods);
	if (SOS_IDXDIR_UDATA(sos->idx_udata)->signature != SOS_IDXDIR_SIGNATURE) {
		errno = EINVAL;
		sos_error("Database %s index directory is not valid. "
			  "Expected a signature of %llx, got %llx\n",
			  path_arg,
			  SOS_IDXDIR_SIGNATURE,
			  SOS_IDXDIR_UDATA(sos->idx_udata)->signature);
		ods_obj_put(sos->idx_udata);
		goto err;
	}

	/* Open the index on the index objects */
	sprintf(tmp_path, "%s/.__index_idx", path);
	sos->idx_idx = ods_idx_open(tmp_path, sos->o_perm);
	if (!sos->idx_idx) {
		sos_error("Error %d opening index index in open of %s\n", errno, path_arg);
		ods_obj_put(sos->idx_udata);
		goto err;
	}

	/* Open the ODS containing the Position objects */
	rc = __open_pos_info(sos, tmp_path, path);
	if (rc) {
		sos_error("Error %d opening the pos container %s\n", rc, path);
		goto err;
	}

	/* Open the ODS containing the schema objects */
	sprintf(tmp_path, "%s/.__schemas", path);
	sos->schema_ods = ods_open(tmp_path, sos->o_perm);
	if (!sos->schema_ods) {
		sos_error("Error %d opening schema ODS in open of %s\n", errno, path_arg);
		goto err;
	}
	ods_obj_t udata = ods_get_user_data(sos->schema_ods);
	if ((SOS_SCHEMA_UDATA(udata)->signature != SOS_SCHEMA_SIGNATURE)) {
		errno = EINVAL;
		sos_error("Schema ODS in %s is corrupted expected %llX, got %llx\n", path_arg,
			  SOS_SCHEMA_SIGNATURE,
			  SOS_SCHEMA_UDATA(udata)->signature);
		ods_obj_put(udata);
		goto err;
	}

	if (SOS_SCHEMA_UDATA(udata)->version != SOS_LATEST_VERSION) {
		errno = EPROTO;
		sos_error("Schema ODS in %s is an unsupported version expected %llX, got %llx\n",
			  path_arg,
			  SOS_LATEST_VERSION,
			  SOS_SCHEMA_UDATA(udata)->version);
		ods_obj_put(udata);
		goto err;
	}
	ods_obj_put(udata);

	/* Open the index on the schema objects */
	sprintf(tmp_path, "%s/.__schema_idx", path);
	sos->schema_idx = ods_idx_open(tmp_path, sos->o_perm);
	if (!sos->schema_idx)
		goto err;

	/*
	 * Build the schema dictionary
	 */
	iter = ods_iter_new(sos->schema_idx);
	ods_lock(sos->schema_ods, 0, NULL);
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_next(iter)) {
		sos_obj_ref_t obj_ref;
		obj_ref.idx_data = ods_iter_data(iter);
		ods_obj_t schema_obj = ods_ref_as_obj(sos->schema_ods, obj_ref.ref.obj);
		sos_schema_t schema = __sos_schema_init(sos, schema_obj);
		if (!schema) {
			ods_unlock(sos->schema_ods, 0);
			goto err;
		}
	}
	ods_unlock(sos->schema_ods, 0);

	/*
	 * Open the partitions
	 */
	rc = __sos_open_partitions(sos, tmp_path);
	if (rc) {
		errno = rc;
		goto err;
	}

	ods_iter_delete(iter);
	__pos_cleanup(sos);

	pthread_mutex_lock(&cont_list_lock);
	LIST_INSERT_HEAD(&cont_list, sos, entry);
	pthread_mutex_unlock(&cont_list_lock);
	close(dfd);
	return sos;
 err:
 	if (dfd >= 0)
 		close(dfd); /* The file descriptor will be unlocked automatically */
	if (iter)
		ods_iter_delete(iter);
	free_sos(sos, SOS_COMMIT_ASYNC);
	return NULL;
}

/**
 * \brief Verify a container
 *
 * Perform internal consistency checks on the container. Use this to
 * verify that the container is not corrrupted.
 *
 * \param sos The container handle
 * \returns 0 if the database is healthy
 */
int sos_container_verify(sos_t sos)
{
	char path[PATH_MAX];
	sos_obj_ref_t idx_ref;
	ods_idx_t idx;
	int rc, res = 0;
	ods_iter_t iter = ods_iter_new(sos->idx_idx);
	for (rc = ods_iter_begin(iter); !rc; rc = ods_iter_next(iter)) {
		idx_ref.idx_data = ods_iter_data(iter);
		ods_obj_t idx_obj = ods_ref_as_obj(sos->idx_ods, idx_ref.ref.obj);
		fprintf(stdout, "Verifying %s\n", SOS_IDX(idx_obj)->name);
		snprintf(path, PATH_MAX, "%s/%s_idx", sos->path, SOS_IDX(idx_obj)->name);
		idx = ods_idx_open(path, sos->o_perm);
		ods_obj_put(idx_obj);
		if (!idx) {
			fprintf(stdout, "Error %d opening %s\n",
				errno, path);
			goto out;
		}
		res = ods_idx_verify(idx, stdout);
	}
 out:
	ods_iter_delete(iter);
	return res;
}

/**
 * \brief Move a container
 *
 * Changes the path data that the container keeps internally for partitions.
 *
 * \param path		Pathname to the Container. See sos_container_new()
 * \param new_path	Pathname where the containter was copied.
 */
int sos_container_move(const char *path_arg, const char *new_path)
{
	char tmp_path[PATH_MAX];
	char *path = NULL;
	ods_obj_t part_obj;
	ods_t part_ods;
	ods_obj_t part_udata;
	ods_ref_t next_ref;
	int rc;

	if (strlen(new_path) >= SOS_PART_PATH_LEN)
		return E2BIG;

	if (strlen(path_arg) >= SOS_PART_PATH_LEN)
		return E2BIG;

	if (path_arg[0] != '/') {
		if (!getcwd(tmp_path, sizeof(tmp_path)))
			return ENOENT;
		if (strlen(tmp_path) + strlen(path_arg) > SOS_PART_PATH_LEN)
			return E2BIG;
		strcat(tmp_path, "/");
		strcat(tmp_path, path_arg);
		path = strdup(tmp_path);
	} else
		path = strdup(path_arg);
	if (!path)
		return ENOMEM;

	/* Open the partition ODS */
	sprintf(tmp_path, "%s/.__part", path);
	part_ods = ods_open(tmp_path, ODS_PERM_RW);
	if (!part_ods)
		goto out_0;

	part_udata = ods_get_user_data(part_ods);
	if (!part_udata)
		goto out_1;

	rc = 0;
	part_obj = ods_ref_as_obj(part_ods, SOS_PART_UDATA(part_udata)->head);
	if (!part_obj)
		goto out_2;

	while (part_obj) {
		strcpy(SOS_PART(part_obj)->path, new_path);
		ods_obj_put(part_obj);

		next_ref = SOS_PART(part_obj)->next;
		part_obj = ods_ref_as_obj(part_ods, next_ref);
	}

 out_2:
	ods_obj_put(part_udata);
 out_1:
	ods_close(part_ods, ODS_COMMIT_SYNC);
 out_0:
	free(path);
	return rc;
}

/**
 * \brief Return information about a container
 *
 * Fills a Unix struct stat buffer with information about a container's meta data.
 *
 * \param sos The container handle
 * \param sb The struct stat buffer
 * \retval 0 Success
 * \retval !0 A Unix error code
 */
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
	__pos_cleanup(sos);

	pthread_mutex_lock(&cont_list_lock);
	LIST_REMOVE(sos, entry);
	pthread_mutex_unlock(&cont_list_lock);

	free_sos(sos, SOS_COMMIT_ASYNC);
}

/*
 * This function steals the reference count of the input ods_obj
 */
sos_obj_t __sos_init_obj_no_lock(sos_t sos, sos_schema_t schema, ods_obj_t ods_obj,
				 sos_obj_ref_t obj_ref)
{
	sos_obj_t sos_obj;

	/* Verify the reference provided */
	if (!ods_ref_valid(ods_obj->ods, obj_ref.ref.obj)) {
		sos_error("Invalid object reference %p:%p",
			  obj_ref.ref.ods, obj_ref.ref.obj);
		return NULL;
	}
	if (!LIST_EMPTY(&sos->obj_free_list)) {
		sos_obj = LIST_FIRST(&sos->obj_free_list);
		LIST_REMOVE(sos_obj, entry);
	} else
		sos_obj = malloc(sizeof *sos_obj);
	if (!sos_obj)
		return NULL;
	LIST_INSERT_HEAD(&sos->obj_list, sos_obj, entry);
	SOS_OBJ(ods_obj)->schema = schema->data->id;
	sos_obj->sos = sos;
	sos_obj->obj = ods_obj;
	sos_obj->obj_ref = obj_ref;
	ods_atomic_inc(&schema->data->ref_count);
	sos_obj->schema = schema;
	sos_obj->ref_count = 1;

	return sos_obj;
}

sos_obj_t __sos_init_obj(sos_t sos, sos_schema_t schema, ods_obj_t ods_obj,
			 sos_obj_ref_t obj_ref)
{
	sos_obj_t sos_obj;
	pthread_mutex_lock(&sos->lock);
	sos_obj = __sos_init_obj_no_lock(sos, schema, ods_obj, obj_ref);
	pthread_mutex_unlock(&sos->lock);
	return sos_obj;
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
 * to all indices defined by its object class.
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

	if (!schema || !schema->sos)
		return NULL;
	part = __sos_primary_obj_part(schema->sos);
	if (!part) {
		errno = ENOSPC;
		return NULL;
	}
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

/**
 * \brief Copy the data in one object to another
 *
 * Copy all of the data from the object specified by \c src_obj
 * to the object specified by \c dst_obj. The objects do not
 * need to be in the same container. If \c src_obj contains
 * array attributes, an array of the required size is allocated
 * in the destination object and the data copied.
 *
 * The objects do not need to have the same schema, however, the
 * type of the attributes in \c src_obj must match the types of
 * the attributes in the same order in \c dst_obj.
 *
 * \param dst_obj	The object to copy to
 * \param src_obj	The object to copy from
 * \retval 0		The object data was sucessfully copied
 * \retval EINVAL	The source and destination object attributes object attributes don't match
 * \retval ENOMEM	There were insufficient resources to assign the object
 */
int sos_obj_copy(sos_obj_t dst_obj, sos_obj_t src_obj)
{
	sos_attr_t src_attr, dst_attr;
	sos_value_data_t src_data, dst_data;

	dst_attr = TAILQ_FIRST(&dst_obj->schema->attr_list);
	TAILQ_FOREACH(src_attr, &src_obj->schema->attr_list, entry) {
		if (sos_attr_type(src_attr) != sos_attr_type(dst_attr))
			return EINVAL;
		if (sos_attr_type(src_attr) == SOS_TYPE_JOIN) {
			/* JOIN attrs do not occupy space in the object data */
			dst_attr = TAILQ_NEXT(dst_attr, entry);
			continue;
		}
		if (!sos_attr_is_array(src_attr)) {
			src_data = sos_obj_attr_data(src_obj, src_attr, NULL);
			dst_data = sos_obj_attr_data(dst_obj, dst_attr, NULL);
			if (!src_data || !dst_data)
				return EINVAL;
			memcpy(dst_data->prim.struc_, src_data->prim.struc_,
			       sos_attr_size(src_attr));
		} else {
			struct sos_value_s src_v_, dst_v_;
			sos_value_t src_value;
			sos_value_t dst_value;
			src_value = sos_value_init(&src_v_, src_obj, src_attr);
			if (!src_value)
				return EINVAL;
			dst_value = sos_array_new(&dst_v_, dst_attr, dst_obj,
						  src_value->data->array.count);
			if (!dst_value)
				return ENOMEM;
			memcpy(dst_value->data->array.data.byte_,
			       src_value->data->array.data.byte_,
			       sos_value_size(src_value));
			sos_value_put(src_value);
			sos_value_put(dst_value);
		}
		dst_attr = TAILQ_NEXT(dst_attr, entry);
	}
	return 0;
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
 * i.e. SOS_TYPE_OBJ_REF attribute values. This function does does not
 * remove the object from indices that may refer to the object.
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
		ods_obj_delete(value->obj->obj);
		sos_value_put(value);
	}
	ods_obj_delete(obj->obj);
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
 * \param obj	The object handle. If NULL, this function is a no-op.
 */
void sos_obj_put(sos_obj_t obj)
{
	if (obj && !ods_atomic_dec(&obj->ref_count)) {
		sos_t sos = obj->sos;
		if (!sos)
			return;
		ods_obj_put(obj->obj);
		pthread_mutex_lock(&sos->lock);
		LIST_REMOVE(obj, entry);
		LIST_INSERT_HEAD(&sos->obj_free_list, obj, entry);
		pthread_mutex_unlock(&sos->lock);
	}
}
void __sos_obj_put_no_lock(sos_obj_t obj)
{
	if (obj && !ods_atomic_dec(&obj->ref_count)) {
		sos_t sos = obj->sos;
		if (!sos)
			return;
		ods_obj_put(obj->obj);
		LIST_INSERT_HEAD(&sos->obj_free_list, obj, entry);
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
	sos_key_t key;
	TAILQ_FOREACH(attr, &obj->schema->attr_list, entry) {
		sos_index_t index;
		struct sos_value_s v_;
		sos_value_t value;
		index = sos_attr_index(attr);
		if (!index)
			continue;
		value = sos_value_init(&v_, obj, attr);
		key_sz = sos_value_size(value);
		key = sos_key_new(key_sz);
		if (!key) {
			sos_value_put(value);
			return ENOMEM;
		}
		ods_key_set(key, sos_value_as_key(value), key_sz);
		rc = ods_idx_delete(index->idx, key, &obj->obj_ref.idx_data);
		sos_key_put(key);
		sos_value_put(value);
		if (rc)
			return rc;
	}

	return 0;
}

/**
 * \brief Add an object to its indexes
 *
 * Add an object to all the indices defined in its schema. This
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
	sos_key_t the_key = NULL;
	SOS_KEY(key);
	int rc;

	TAILQ_FOREACH(attr, &obj->schema->idx_attr_list, idx_entry) {
		sos_index_t index = sos_attr_index(attr);
		if (!index)
			return errno;
		value = sos_value_init(&v_, obj, attr);
		if (!value) {
			/* Array value not set, skip */
			continue;
		}
		key_sz = sos_value_size(value);
		if (key_sz < 254) {
			the_key = key;
		} else {
			the_key = sos_key_new(key_sz);
		}
		sos_key_set(the_key, sos_value_as_key(value), key_sz);
		rc = sos_index_insert(index, the_key, obj);
		if (the_key != key)
			sos_key_put(the_key);
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

