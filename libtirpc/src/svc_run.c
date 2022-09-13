/*
 * Copyright (c) 2009, Sun Microsystems, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of Sun Microsystems, Inc. nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This is the rpc server side idle loop
 * Wait for input, call server program.
 */
#include <pthread.h>
#include <reentrant.h>
#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/poll.h>


#include <rpc/rpc.h>
#include "rpc_com.h"
#include <sys/select.h>
#include <sys/wait.h>

void *wait_proc(void *arg)
{
  int wstatus;
  pthread_t t = pthread_self();
  pthread_setname_np(t, "svc_run:waitpid");
  for (;;) {
    pid_t pid = waitpid(-1, &wstatus, 0);
    if (pid == -1)
      {
	if (errno != ECHILD)
	  {
	    fprintf (stderr, "wait_proc: waitpid returned with error %d\n", errno);
	  }
	sleep(10);
	continue;
      }
  }
  return NULL;
}

void
svc_run()
{
  int i, quit;
  struct pollfd *my_pollfd = NULL;
  int last_max_pollfd = 0;
  pthread_t wait_thread;

  /* Add a thread to wait for exiting children */
  i = pthread_create(&wait_thread, NULL, wait_proc, NULL);
  if (i)
    {
      warn ("svc_run: - could not create wait thread\n");
      return;
    }
  for (;;) {
    int max_pollfd = svc_max_pollfd;
    if (max_pollfd == 0 && svc_pollfd == NULL)
        break;

      if (last_max_pollfd != max_pollfd)
        {
          struct pollfd *new_pollfd
            = realloc (my_pollfd, sizeof (struct pollfd) * max_pollfd);

          if (new_pollfd == NULL)
            {
              warn ("svc_run: - out of memory");
              break;
            }

          my_pollfd = new_pollfd;
          last_max_pollfd = max_pollfd;
        }
      quit = 1;
      for (i = 0; i < max_pollfd; ++i)
        {
          my_pollfd[i].fd = svc_pollfd[i].fd;
          if (svc_pollfd[i].fd != -1)
            quit = 0;
          my_pollfd[i].events = svc_pollfd[i].events | POLLRDHUP;
          my_pollfd[i].revents = 0;
        }
      if (quit)
        {
          /* Exiting because pollfd is empty */
          break;
        }

      switch (i = poll (my_pollfd, max_pollfd, -1))
        {
        case -1:
          if (errno == EINTR)
            continue;
          warn ("svc_run: - poll failed");
          break;
        case 0:
          continue;
        default:
          svc_getreq_poll (my_pollfd, i);
          continue;
        }
      break;
    }

  free (my_pollfd);
}

/*
 *      This function causes svc_run() to exit by telling it that it has no
 *      more work to do.
 */
void
svc_exit()
{
	extern rwlock_t svc_fd_lock;

	rwlock_wrlock(&svc_fd_lock);
	free (svc_pollfd);
	svc_pollfd = NULL;
	svc_max_pollfd = 0;
	rwlock_unlock(&svc_fd_lock);
}
