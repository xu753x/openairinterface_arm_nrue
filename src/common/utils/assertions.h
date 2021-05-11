

#ifndef __COMMON_UTILS_ASSERTIONS__H__
#define __COMMON_UTILS_ASSERTIONS__H__

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <platform_types.h>

#if defined(ENB_MODE)
# define display_backtrace()
#else
# include "backtrace.h"
#endif

void output_log_mem(void);
#define _Assert_Exit_                           \
    fprintf(stderr, "\nExiting execution\n");   \
    display_backtrace();                        \
    fflush(stdout);                             \
    fflush(stderr);                             \
    abort();

#define _Assert_(cOND, aCTION, fORMAT, aRGS...)             \
do {                                                        \
    if (!(cOND)) {                                          \
        fprintf(stderr, "\nAssertion ("#cOND") failed!\n"   \
                "In %s() %s:%d\n" fORMAT,                   \
                __FUNCTION__, __FILE__, __LINE__, ##aRGS);  \
        aCTION;                                             \
    }						\
} while(0)

#define AssertFatal(cOND, fORMAT, aRGS...)          _Assert_(cOND, _Assert_Exit_, fORMAT, ##aRGS)

#define AssertError(cOND, aCTION, fORMAT, aRGS...)  _Assert_(cOND, aCTION, fORMAT, ##aRGS)



#define DevCheck(cOND, vALUE1, vALUE2, vALUE3)                                                          \
_Assert_(cOND, _Assert_Exit_, #vALUE1 ": %" PRIdMAX "\n" #vALUE2 ": %" PRIdMAX "\n" #vALUE3 ": %" PRIdMAX "\n\n",  \
         (intmax_t)vALUE1, (intmax_t)vALUE2, (intmax_t)vALUE3)

#define DevCheck4(cOND, vALUE1, vALUE2, vALUE3, vALUE4)                                                                         \
_Assert_(cOND, _Assert_Exit_, #vALUE1": %" PRIdMAX "\n" #vALUE2 ": %" PRIdMAX "\n" #vALUE3 ": %" PRIdMAX "\n" #vALUE4 ": %" PRIdMAX "\n\n",   \
         (intmax_t)vALUE1, (intmax_t)vALUE2, (intmax_t)vALUE3, (intmax_t)vALUE4)

#define DevParam(vALUE1, vALUE2, vALUE3)    DevCheck(0, vALUE1, vALUE2, vALUE3)

#define DevAssert(cOND)                     _Assert_(cOND, _Assert_Exit_, "")

#define DevMessage(mESSAGE)                 _Assert_(0, _Assert_Exit_, #mESSAGE)

#define CHECK_INIT_RETURN(fCT)                                  \
do {                                                            \
    int fct_ret;                                                \
    if ((fct_ret = (fCT)) != 0) {                               \
        fprintf(stderr, "Function "#fCT" has failed\n"          \
        "returning %d\n", fct_ret);                             \
        fflush(stdout);                                         \
        fflush(stderr);                                         \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

#endif /* __COMMON_UTILS_ASSERTIONS__H__ */
