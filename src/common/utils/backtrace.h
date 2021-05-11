

#include <signal.h>

#ifndef BACKTRACE_H_
#define BACKTRACE_H_
#ifdef __cplusplus
extern "C" {
#endif

void display_backtrace(void);

void backtrace_handle_signal(siginfo_t *info);
#ifdef __cplusplus
}
#endif


#endif /* BACKTRACE_H_ */
