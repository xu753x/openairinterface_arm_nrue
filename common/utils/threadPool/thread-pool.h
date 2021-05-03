/*
* Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The OpenAirInterface Software Alliance licenses this file to You under
* the OAI Public License, Version 1.1  (the "License"); you may not use this file
* except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.openairinterface.org/?page_id=698
*
* Author and copyright: Laurent Thomas, open-cells.com
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*-------------------------------------------------------------------------------
* For more information about the OpenAirInterface (OAI) Software Alliance:
*      contact@openairinterface.org
*/


#ifndef THREAD_POOL_H
#define THREAD_POOL_H
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <assertions.h>
#include <LOG/log.h>
#include <common/utils/system.h>

#ifdef DEBUG
  #define THREADINIT   PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP
#else
  #define THREADINIT   PTHREAD_MUTEX_INITIALIZER
#endif
#define mutexinit(mutex)   {int ret=pthread_mutex_init(&mutex,NULL); \
                            AssertFatal(ret==0,"ret=%d\n",ret);}
#define condinit(signal)   {int ret=pthread_cond_init(&signal,NULL); \
                            AssertFatal(ret==0,"ret=%d\n",ret);}
#define mutexlock(mutex)   {int ret=pthread_mutex_lock(&mutex); \
                            AssertFatal(ret==0,"ret=%d\n",ret);}
#define mutextrylock(mutex)   pthread_mutex_trylock(&mutex)
#define mutexunlock(mutex) {int ret=pthread_mutex_unlock(&mutex); \
                            AssertFatal(ret==0,"ret=%d\n",ret);}
#define condwait(condition, mutex) {int ret=pthread_cond_wait(&condition, &mutex); \
                                    AssertFatal(ret==0,"ret=%d\n",ret);}
#define condbroadcast(signal) {int ret=pthread_cond_broadcast(&signal); \
                               AssertFatal(ret==0,"ret=%d\n",ret);}
#define condsignal(signal)    {int ret=pthread_cond_broadcast(&signal); \
                               AssertFatal(ret==0,"ret=%d\n",ret);}
#define tpool_nbthreads(tpool)   (tpool.nbThreads)
typedef struct notifiedFIFO_elt_s {
  struct notifiedFIFO_elt_s *next;
  uint64_t key; //To filter out elements
  struct notifiedFIFO_s *reponseFifo;
  void (*processingFunc)(void *);
  bool malloced;
  uint64_t creationTime;
  uint64_t startProcessingTime;
  uint64_t endProcessingTime;
  uint64_t returnTime;
  void *msgData;
}  notifiedFIFO_elt_t;

typedef struct notifiedFIFO_s {
  notifiedFIFO_elt_t *outF;
  notifiedFIFO_elt_t *inF;
  pthread_mutex_t lockF;
  pthread_cond_t  notifF;
} notifiedFIFO_t;

// You can use this allocator or use any piece of memory
static inline notifiedFIFO_elt_t *newNotifiedFIFO_elt(int size,
    uint64_t key,
    notifiedFIFO_t *reponseFifo,
    void (*processingFunc)(void *)) {
  notifiedFIFO_elt_t *ret;
  AssertFatal( NULL != (ret=(notifiedFIFO_elt_t *) malloc(sizeof(notifiedFIFO_elt_t)+size+32)), "");
  ret->next=NULL;
  ret->key=key;
  ret->reponseFifo=reponseFifo;
  ret->processingFunc=processingFunc;
  // We set user data piece aligend 32 bytes to be able to process it with SIMD
  ret->msgData=(void *)((uint8_t*)ret+(sizeof(notifiedFIFO_elt_t)/32+1)*32);
  ret->malloced=true;
  return ret;
}

static inline void *NotifiedFifoData(notifiedFIFO_elt_t *elt) {
  return elt->msgData;
}

static inline void delNotifiedFIFO_elt(notifiedFIFO_elt_t *elt) {
  if (elt->malloced) {
    elt->malloced=false;
    free(elt);
  } else
    printf("delNotifiedFIFO on something not allocated by newNotifiedFIFO\n");

  //LOG_W(UTIL,"delNotifiedFIFO on something not allocated by newNotifiedFIFO\n");
}

static inline void initNotifiedFIFO_nothreadSafe(notifiedFIFO_t *nf) {
  nf->inF=NULL;
  nf->outF=NULL;
}
static inline void initNotifiedFIFO(notifiedFIFO_t *nf) {
  mutexinit(nf->lockF);
  condinit (nf->notifF);
  initNotifiedFIFO_nothreadSafe(nf);
  // No delete function: the creator has only to free the memory
}

static inline void pushNotifiedFIFO_nothreadSafe(notifiedFIFO_t *nf, notifiedFIFO_elt_t *msg) {
  msg->next=NULL;

  if (nf->outF == NULL)
    nf->outF = msg;

  if (nf->inF != NULL)
    nf->inF->next = msg;

  nf->inF = msg;
}

static inline void pushNotifiedFIFO(notifiedFIFO_t *nf, notifiedFIFO_elt_t *msg) {
  mutexlock(nf->lockF);
  pushNotifiedFIFO_nothreadSafe(nf,msg);
  condbroadcast(nf->notifF);
  mutexunlock(nf->lockF);
}

static inline  notifiedFIFO_elt_t *pullNotifiedFIFO_nothreadSafe(notifiedFIFO_t *nf) {
  if (nf->outF == NULL)
    return NULL;

  notifiedFIFO_elt_t *ret=nf->outF;

  if (nf->outF==nf->outF->next)
    LOG_E(TMR,"Circular list in thread pool: push several times the same buffer is forbidden\n");

  nf->outF=nf->outF->next;

  if (nf->outF==NULL)
    nf->inF=NULL;

  return ret;
}

static inline  notifiedFIFO_elt_t *pullNotifiedFIFO(notifiedFIFO_t *nf) {
  mutexlock(nf->lockF);
  notifiedFIFO_elt_t *ret;

  while((ret=pullNotifiedFIFO_nothreadSafe(nf)) == NULL)
    condwait(nf->notifF, nf->lockF);

  mutexunlock(nf->lockF);
  return ret;
}

static inline  notifiedFIFO_elt_t *pollNotifiedFIFO(notifiedFIFO_t *nf) {
  int tmp=mutextrylock(nf->lockF);

  if (tmp != 0 )
    return NULL;

  notifiedFIFO_elt_t *ret=pullNotifiedFIFO_nothreadSafe(nf);
  mutexunlock(nf->lockF);
  return ret;
}

// This function aborts all messages matching the key
// If the queue is used in thread pools, it doesn't cancels already running processing
// because the message has already been picked
static inline int abortNotifiedFIFO(notifiedFIFO_t *nf, uint64_t key) {
  mutexlock(nf->lockF);
  int nbDeleted=0;
  notifiedFIFO_elt_t **start=&nf->outF;

  while(*start!=NULL) {
    if ( (*start)->key == key ) {
      notifiedFIFO_elt_t *request=*start;
      *start=(*start)->next;
      delNotifiedFIFO_elt(request);
      nbDeleted++;
    } else
      start=&(*start)->next;
  }

  if (nf->outF == NULL)
    nf->inF=NULL;

  mutexunlock(nf->lockF);
  return nbDeleted;
}

struct one_thread {
  pthread_t  threadID;
  int id;
  int coreID;
  char name[256];
  uint64_t runningOnKey;
  bool abortFlag;
  struct thread_pool *pool;
  struct one_thread *next;
};

typedef struct thread_pool {
  int activated;
  bool measurePerf;
  int traceFd;
  int dummyTraceFd;
  uint64_t cpuCyclesMicroSec;
  uint64_t startProcessingUE;
  int nbThreads;
  bool restrictRNTI;
  notifiedFIFO_t incomingFifo;
  struct one_thread *allthreads;
} tpool_t;

static inline void pushTpool(tpool_t *t, notifiedFIFO_elt_t *msg) {
  if (t->measurePerf) msg->creationTime=rdtsc();

  if ( t->activated)
    pushNotifiedFIFO(&t->incomingFifo, msg);
  else {
    if (t->measurePerf)
      msg->startProcessingTime=rdtsc();

    msg->processingFunc(NotifiedFifoData(msg));

    if (t->measurePerf)
      msg->endProcessingTime=rdtsc();

    if (msg->reponseFifo)
      pushNotifiedFIFO(msg->reponseFifo, msg);
  }
}

static inline notifiedFIFO_elt_t *pullTpool(notifiedFIFO_t *responseFifo, tpool_t *t) {
  notifiedFIFO_elt_t *msg= pullNotifiedFIFO(responseFifo);
  AssertFatal(t->traceFd, "Thread pool used while not initialized");
  if (t->measurePerf)
    msg->returnTime=rdtsc();

  if (t->traceFd > 0)
    if(write(t->traceFd, msg, sizeof(*msg)));

  return msg;
}

static inline notifiedFIFO_elt_t *tryPullTpool(notifiedFIFO_t *responseFifo, tpool_t *t) {
  notifiedFIFO_elt_t *msg= pollNotifiedFIFO(responseFifo);
  AssertFatal(t->traceFd, "Thread pool used while not initialized");
  if (msg == NULL)
    return NULL;

  if (t->measurePerf)
    msg->returnTime=rdtsc();

  if (t->traceFd)
    if(write(t->traceFd, msg, sizeof(*msg)));

  return msg;
}

static inline int abortTpool(tpool_t *t, uint64_t key) {
  int nbRemoved=0;
  notifiedFIFO_t *nf=&t->incomingFifo;
  mutexlock(nf->lockF);
  notifiedFIFO_elt_t **start=&nf->outF;

  while(*start!=NULL) {
    if ( (*start)->key == key ) {
      notifiedFIFO_elt_t *request=*start;
      *start=(*start)->next;
      delNotifiedFIFO_elt(request);
      nbRemoved++;
    } else
      start=&(*start)->next;
  }

  if (t->incomingFifo.outF==NULL)
    t->incomingFifo.inF=NULL;

  struct one_thread *ptr=t->allthreads;

  while(ptr!=NULL) {
    if (ptr->runningOnKey==key) {
      ptr->abortFlag=true;
      nbRemoved++;
    }

    ptr=ptr->next;
  }

  mutexunlock(nf->lockF);
  return nbRemoved;
}
void initNamedTpool(char *params,tpool_t *pool, bool performanceMeas, char *name);
#define  initTpool(PARAMPTR,TPOOLPTR, MEASURFLAG) initNamedTpool(PARAMPTR,TPOOLPTR, MEASURFLAG, NULL)
#endif
