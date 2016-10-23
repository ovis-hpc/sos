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

#ifndef __ODS_H
#define __ODS_H
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/queue.h>
#include <sys/stat.h>
#include <ods/ods_atomic.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ods_s *ods_t;
typedef uint64_t ods_ref_t;
typedef struct ods_map_s *ods_map_t;
typedef struct ods_obj_s *ods_obj_t;
extern int ods_debug;
extern int ods_obj_track;

/**
 * \brief Create an object store
 *
 * \param path The path to the ODS to be opened.
 * \param o_mode The file mode. See the creat() system call.
 * \retval 0		The object store was created
 * \retval EINVAL	A incorrect parameter was specified
 * \retval EPERM	Permission denied.
 */
extern int ods_create(const char *path, int o_mode);

/**
 * \brief Destroy an object store
 *
 * \param path The path to the ODS to be destroyed.
 * \retval 0		The object store was destroyed
 * \retval EINVAL	A incorrect parameter was specified
 * \retval EPERM	Permission denied.
 */
extern int ods_destroy(const char *path);

typedef enum ods_perm_e {
	ODS_PERM_RO = 0,
	ODS_PERM_RW
} ods_perm_t;

/**
 * \brief Return the path used to open/create the ods
 * \retval The path;
 */
extern const char *ods_path(ods_t ods);

/**
 * \brief Open and optionally create an ODS object store
 *
 * \param path	The path to the ODS to be opened.
 * \param o_perm The requested read/write permissions.
 * \retval !0	The ODS handle
 * \retval 0	An error occured opening/creating the ODS
 */
extern ods_t ods_open(const char *path, ods_perm_t o_perm);

/**
 * \brief Obtain file stats for the ODS database
 *
 * \param ods	The ODS handle
 * \param sb	Pointer to a struct stat buffer, see the stat() system call.
 * \retval 0	The stats were successfully queried.
 * \retval !0	See the stat() system call.
 */
extern int ods_stat(ods_t ods, struct stat *sb);

/**
 * \brief Truncate an ODS to it's minimum size
 *
 * An ODS has both allocated and unallocated space. This function
 * releases unallocated storage and reduces the storage size to the
 * minimum necessary to contain the allocated objects.
 *
 * \param ods  The ODS handle
 * \retval 0   Success
 * \retval !0  An error occured truncating the storage
 */
int ods_pack(ods_t ods);

/**
 * \brief Acquire a pointer to the user-data for the ODS
 *
 * An ODS has storage pre-allocated for the storager of
 * meta-data. This function returns a pointer to that storage.
 *
 * \param ods	The ODS ods handle
 * \return A pointer to the user data.
 */
#ifdef ODS_OBJ_DEBUG
extern ods_obj_t _ods_get_user_data(ods_t ods);
#define ods_get_user_data(ods) ({		\
	ods_obj_t o = _ods_get_user_data(ods);	\
	if (ods_obj_track && o) {		\
		o->thread = pthread_self();	\
		o->alloc_line = __LINE__;	\
		o->alloc_func = __func__;	\
	}					\
	o;					\
})
#else
extern ods_obj_t ods_get_user_data(ods_t ods);
#endif
#define ODS_COMMIT_ASYNC	0
#define ODS_COMMIT_SYNC		1
/**
 * \brief Commit changes to stable storage
 *
 * This function initiates and optionally waits for these changes to
 * be committed to stable storage.
 *
 * The 'flags' parameter determines whether the function returns
 * immediately or waits for changes to be commited to stable
 * storage. If flags is set to ODS_COMMIT_SYNC is set, the function
 * will wait for the commit to complete before returning to the
 * caller.
 *
 * \param ods	The ODS handle
 * \param flags	The commit flags.
 */
extern void ods_commit(ods_t ods, int flags);

/**
 * \brief Close an ODS store
 *
 * Close the ODS store and flush all commits to persistent
 * storage. If the 'flags' parameter is set to ODS_COMMIT_SYNC, the
 * function waits until the changes are commited to stable storage
 * before returning.
 *
 * This function protects against releasing a NULL store.
 *
 * \param ods	The ods handle.
 * \param flags	Set to ODS_COMMIT_SYNC for a synchronous close.
 */
extern void ods_close(ods_t ods, int flags);

/**
 * \brief Allocate an object of the requested size
 *
 * Allocates space in the persistent store for an object of at least
 * \c sz bytes and initializes an in-memory object that refers to this
 * ODS object. The application should use the ods_obj_get()
 * function to add references to the object and ods_obj_put() to
 * release these references. The ODS storage for the object is freed
 * using the the ods_obj_delete() function.
 *
 * \param ods	The ODS handle
 * \param sz	The desired size
 * \return	Pointer to an object of the requested size or NULL if there
 *		is an error.
 */
#ifdef ODS_OBJ_DEBUG
extern ods_obj_t _ods_obj_alloc(ods_t ods, size_t sz);
#define ods_obj_alloc(ods, sz) ({		\
	ods_obj_t o = _ods_obj_alloc(ods, sz);	\
	if (ods_obj_track && o) {				\
		o->thread = pthread_self();	\
		o->alloc_line = __LINE__;	\
		o->alloc_func = __func__;	\
	}					\
	o;					\
})
#else
extern ods_obj_t ods_obj_alloc(ods_t ods, size_t sz);
#endif
/**
 * \brief Allocate a memory object of the requested size
 *
 * Allocates space in memory for an object of at least
 * <tt>sz</tt> bytes and initializes an in-memory object that refers to this
 * ODS object. The application should use the ods_obj_get()
 * function to add references to the object and ods_obj_put() to
 * release these references.
 *
 * \param sz	The desired size
 * \return	Pointer to an object of the requested size or NULL if there
 *		is an error.
 */
#ifdef ODS_OBJ_DEBUG
extern ods_obj_t _ods_obj_malloc(size_t sz);
#define ods_obj_malloc(sz) ({		\
	ods_obj_t o = _ods_obj_malloc(sz);	\
	if (ods_obj_track && o) {			\
		o->thread = pthread_self();	\
		o->alloc_line = __LINE__;	\
		o->alloc_func = __func__;	\
	}					\
	o;					\
})
#else
extern ods_obj_t ods_obj_malloc(size_t sz);
#endif
#ifdef ODS_OBJ_DEBUG
#define ODS_OBJ(_name_, _data_, _sz_)		\
	struct ods_obj_s _name_ = {		\
		.refcount = 0,			\
		.ods = NULL,			\
		.size = _sz_,			\
		.ref = 0,			\
		.as.ptr = _data_,		\
		.map = NULL,			\
		.thread = pthread_self(),	\
		.alloc_func = __func__ ,	\
		.alloc_line = __LINE__,		\
		.put_line = 0,			\
		.put_func = NULL		\
	}
#else
#define ODS_OBJ(_name_, _data_, _sz_)		\
	struct ods_obj_s _name_ = {		\
		.refcount = 0,			\
		.ods = NULL,			\
		.size = _sz_,			\
		.ref = 0,			\
		.as.ptr = _data_,		\
		.map = NULL,			\
	}
#endif
/**
 * \brief Free the storage for this object in the ODS
 *
 * This frees ODS resources associated with the object. The object
 * reference and pointer will be set to zero. Use the \c
 * ods_obj_put() function to release the in-memory resources for the
 * object.
 *
 * \param obj	Pointer to the object
 */
extern void ods_obj_delete(ods_obj_t obj);

/**
 * \brief Free the storage for this reference in the ODS
 *
 * This frees ODS resources associated with the object.
 *
 * \param ods	The ODS handle
 * \param ref	The ODS object reference
 */
extern void ods_ref_delete(ods_t ods, ods_ref_t ref);

#define ODS_REF_STATUS_INVALID	0x80000000
#define ODS_REF_STATUS_FREE	0x40000000
#define ODS_REF_STATUS_INTERIOR	0x20000000
#define ODS_REF_STATUS_CORRUPT	0x10000000
#define ODS_REF_SIZE_MASK	0x0FFFFFFF

uint32_t ods_ref_status(ods_t ods, ods_ref_t ref);

/**
 * Return true if the reference is valid in the ODS
 *
 * \param ods The ODS handle
 * \param ref The object reference
 */
int ods_ref_valid(ods_t ods, ods_ref_t ref);

/**
 * \brief Drop a reference to the object
 *
 * Decrement the reference count on the object. If the reference count
 * goes to zero, the in-memory resrouces associated with the object
 * will be freed.
 *
 * \param obj	Pointer to the object
 */
#ifdef ODS_OBJ_DEBUG
void _ods_obj_put(ods_obj_t obj);
#define ods_obj_put(o)				\
	_ods_obj_put(o);			\
	if (ods_obj_track && o) {			\
		o->thread = pthread_self();	\
		o->put_line = __LINE__;		\
		o->put_func = __func__;		\
	}
#else
extern void ods_obj_put(ods_obj_t obj);
#endif
/**
 * \brief Return the ODS in which an object resides
 *
 * \param obj The object handle
 * \retval The ODS handle
 */
extern ods_t ods_obj_ods(ods_obj_t obj);

/**
 * \brief Extend the object store by the specified amount
 *
 * This function increases the size of the object store by the
 * specified amount.
 *
 * \param ods	The ODS handle
 * \param sz	The requsted additional size in bytes
 * \return 0	Success
 * \return ENOMEM There was insufficient storage to satisfy the request.
 */
extern int ods_extend(ods_t ods, size_t sz);

/**
 * \brief Return the size of the ODS in bytes
 *
 * \param ods	The ODS handle
 * \returns The size of the ODS in bytes
 */
extern size_t ods_size(ods_t ods);

/**
 * \brief Dump the meta data for the ODS
 *
 * This function prints information about the object store such as its
 * size and current allocated and free object locations.
 *
 * \param ods	The ODS handle
 * \param fp	The FILE* to which the information should be sent
 */
extern void ods_dump(ods_t ods, FILE *fp);

/**
 * \brief The callback function called by the ods_obj_iter() function
 *
 * Called by the ods_obj_iter() function for each object in the ODS. If
 * the function wishes to cancel iteration, return !0, otherwise,
 * return 0.
 *
 * \param ods	The ODS handle
 * \param obj	The object handle
 * \param sz	The size of the object
 * \param arg	The 'arg' passed into ods_obj_iter()
 * \retval 0	Continue iterating
 * \retval !0	Stop iterating
 */
typedef int (*ods_obj_iter_fn_t)(ods_t ods, ods_obj_t obj, void *arg);

typedef struct ods_obj_iter_pos_s {
	int page_no;
	int blk;
} *ods_obj_iter_pos_t;

void ods_obj_iter_pos_init(ods_obj_iter_pos_t pos);

/**
 * \brief Iterate over objects in the ODS
 *
 * This function iterates over objects allocated in the ODS and
 * calls the specified 'iter_fn' for each object. See the
 * ods_obj_iter_fn_t() for the function definition.
 *
 * If the the <tt>pos</tt> argument is not NULL, it should be
 * initialized with the ods_obj_iter_pos_init() funuction. The
 * <tt>pos</tt> argument will updated with the location of the next
 * object in the store when ods_obj_iter() returns. This facilitates
 * walking through a portion of the objects at a time, continuing
 * later where the function left off.
 *
 * The ods_obj_iter_fn_t() function indicates that the iteration
 * should stop by returning !0. Otherwise, the ods_obj_iter() function
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
int ods_obj_iter(ods_t ods, ods_obj_iter_pos_t pos, ods_obj_iter_fn_t iter_fn, void *arg);

ods_atomic_t ods_obj_count(ods_t ods);

/*
 * Take a reference on an object
 */
ods_obj_t ods_obj_get(ods_obj_t obj);

/*
 * Create a memory object from a persistent reference
 */
#ifdef ODS_OBJ_DEBUG
ods_obj_t _ods_ref_as_obj(ods_t ods, ods_ref_t ref);
#define ods_ref_as_obj(ods, ref) ({			\
	ods_obj_t o = _ods_ref_as_obj(ods, ref);	\
	if (ods_obj_track && o) {					\
		o->thread = pthread_self();		\
		o->alloc_line = __LINE__;		\
		o->alloc_func = __func__;		\
	}						\
	o;						\
})
#else
ods_obj_t ods_ref_as_obj(ods_t ods, ods_ref_t ref);
#endif
/*
 * Return an object's reference
 */
ods_ref_t ods_obj_ref(ods_obj_t obj);

union ods_obj_type_u {
	void *ptr;
	int8_t *int8;
	uint8_t *uint8;
	int16_t *int16;
	uint16_t *uint16;
	int32_t *int32;
	uint32_t *uint32;
	int64_t *int64;
	uint64_t *uint64;
	char *str;
	unsigned char *bytes;
	ods_atomic_t *lock;
};

struct ods_obj_s {
	ods_atomic_t refcount;
	ods_t ods;
	size_t size;		/* allocated size in store */
	ods_ref_t ref;		/* persistent reference */
	union ods_obj_type_u as;
	ods_map_t map;
	pthread_t thread;
	int alloc_line;
	const char *alloc_func;
	int put_line;
	const char *put_func;
	LIST_ENTRY(ods_obj_s) entry;
};
static inline void *ods_obj_as_ptr(ods_obj_t obj) {
	return obj->as.ptr;
}
/**
 * \brief Return the size of an object
 *
 * This function returns the size of the object pointed to by the
 * 'obj' parameter.
 *
 * \param obj	Pointer to the object
 * \return sz	The allocated size of the object in bytes
 * \return -1	The 'obj' parameter does not point to an ODS object.
 */
static inline size_t ods_obj_size(ods_obj_t obj) {
	return obj->size;
}

/**
 * \brief Verify that an object is valid
 *
 * Check that the provided object is valid for the given ODS.
 *
 * \param ods The ODS handle
 * \param obj The object to validate
 * \retval TRUE (!0) if the object is valid
 * \retval FALSE (0) if the object is not valid
 */
int ods_obj_valid(ods_t ods, ods_obj_t obj);

#define ODS_PTR(_typ_, _obj_) ((_typ_)_obj_->as.ptr)

/**
 * \brief Lock the specified \em lock_id
 *
 * Attempt to lock the specified lock. If the lock is available, the
 * lock is taken and the function returns immediately. If the lock is
 * not available, the function will block and wait until the lock
 * becomes available. The duration of the wait is the current time() +
 * \em wait. If the \em wait parameter is NULL, the function waits
 * forever. The \em wait parameter is a timespec structure as defined
 * in <time.h> as follows:
 *
 *    struct timespec {
 *        time_t tv_sec;
 *        int tv_nsec;
 *    };
 *
 * \param ods The ODS handle
 * \param lock_id The lock idenfifier
 * \param wait The timeout wait period or NULL
 */
extern int ods_lock(ods_t ods, int lock_id, struct timespec *wait);

/**
 * \brief Unlock the specified \em lock_id
 *
 * \param ods The ODS handle
 * \param lock_id The lock identifier
 */
extern void ods_unlock(ods_t ods, int lock_id);

/**
 * \brief Returns the number of available locks
 *
 * Each ODS object repository has a number of shared memory locks that
 * can be used to coordinate access among multiple threads and
 * processes.
 *
 * Each lock is identified by it's \em lock_id which is an integer
 * between zero and the number reported by this function - 1. The
 * association of a \em lock_id with a particular lock domain is
 * application specific.
 *
 * \param ods The ODS handle
 * \retval The number of user spin locks
 */
extern size_t ods_lock_count(ods_t);

/**
 * \brief Print information about locks held on the ODS
 * \param path The path to the ODS
 * \param fp FILE* where information will be printed
 */
int ods_lock_info(const char *path, FILE *fp);

/**
 * \brief Release locks held by dead processes
 * \param path The path to the ODS
 */
int ods_lock_cleanup(const char *path);

/**
 * \brief Print debug information about the repository
 * \param ods The ODS handle
 * \param fp A FILE* pointer to receive the output
 */
void ods_info(ods_t ods, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif
