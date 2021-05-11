

#ifndef _SYSTEM_H_OAI_
#define _SYSTEM_H_OAI_
#include <stdint.h>
#include <pthread.h>
#ifdef __cplusplus
extern "C" {
#endif


/****************************************************
 * send a command to the background process
 *     return -1 on error, 0 on success
 ****************************************************/

int background_system(char *command);

/****************************************************
 * initialize the background process
 *     to be called very early
 ****************************************************/

void start_background_system(void);

void set_latency_target(void);
void configure_linux(void);

void threadCreate(pthread_t* t, void * (*func)(void*), void * param, char* name, int affinity, int priority);
 
#define SCHED_OAI SCHED_RR
#define OAI_PRIORITY_RT_LOW sched_get_priority_min(SCHED_OAI)
#define OAI_PRIORITY_RT ((sched_get_priority_min(SCHED_OAI)+sched_get_priority_max(SCHED_OAI))/2)
#define OAI_PRIORITY_RT_MAX sched_get_priority_max(SCHED_OAI)-2

void thread_top_init(char *thread_name,
                     int affinity,
                     uint64_t runtime,
                     uint64_t deadline,
                     uint64_t period);

#ifdef __cplusplus
}
#endif


#endif /* _SYSTEM_H_OAI_ */
