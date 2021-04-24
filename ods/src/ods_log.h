/*
 * Copyright (c) 2021 Open Grid Computing, Inc. All rights reserved.
 *
 * See the file COPYING at the top of the source tree for license
 * detail.
 */
#ifndef _ODS_LOG_H
#define _ODS_LOG_H
/* ODS Debug True/False */
extern int __ods_debug;

/* Log file pointer and mask */
extern FILE *__ods_log_fp;
extern uint64_t __ods_log_mask;

static inline void ods_log(int level, const char *func, int line, char *fmt, ...)
{
	va_list ap;

	if (!__ods_log_fp)
		return;

	if (0 ==  (level & __ods_log_mask))
		return;

	va_start(ap, fmt);
	fprintf(__ods_log_fp, "ods[%d] @ %s:%d | ", level, func, line);
	vfprintf(__ods_log_fp, fmt, ap);
	fflush(__ods_log_fp);
}

#define ods_lfatal(fmt, ...) ods_log(ODS_LOG_FATAL, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define ods_lerror(fmt, ...) ods_log(ODS_LOG_ERROR, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define ods_lwarn(fmt, ...) ods_log(ODS_LOG_WARN, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define ods_linfo(fmt, ...) ods_log(ODS_LOG_INFO, __func__, __LINE__, fmt, ##__VA_ARGS__)
#define ods_ldebug(fmt, ...) ods_log(ODS_LOG_DEBUG, __func__, __LINE__, fmt, ##__VA_ARGS__)
#endif
