//===-- KTest.h --------------------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __COMMON_KTEST_H__
#define __COMMON_KTEST_H__

#if defined(CRETE_CONFIG)
#include "crete-replayer/crete_debug.h"
#include <stdint.h>
#endif // CRETE_CONFIG

#ifdef __cplusplus
extern "C" {
#endif

  typedef struct KTestObject KTestObject;
  struct KTestObject {
    char *name;
    unsigned numBytes;
    unsigned char *bytes;

#if defined(CRETE_DEBUG_CONCOLIC_TG)
    uint64_t address;
#endif // CRETE_DEBUG_CONCOLIC_TG
  };
  
  typedef struct KTest KTest;
  struct KTest {
    /* file format version */
    unsigned version; 
    
    unsigned numArgs;
    char **args;

    unsigned symArgvs;
    unsigned symArgvLen;

    unsigned numObjects;
    KTestObject *objects;
  };

  
  /* returns the current .ktest file format version */
  unsigned kTest_getCurrentVersion();
  
  /* return true iff file at path matches KTest header */
  int   kTest_isKTestFile(const char *path);

  /* returns NULL on (unspecified) error */
  KTest* kTest_fromFile(const char *path);

  /* returns 1 on success, 0 on (unspecified) error */
  int   kTest_toFile(KTest *, const char *path);
  
  /* returns total number of object bytes */
  unsigned kTest_numBytes(KTest *);

  void  kTest_free(KTest *);

#if defined(CRETE_DEBUG_CONCOLIC_TG)
  /* returns 1 on success, 0 on (unspecified) error */
  int crete_kTest_toFile(KTest *bo, const char *path,
          const void *trace_tag_explored);
#endif // CRETE_DEBUG_CONCOLIC_TG

#ifdef __cplusplus
}
#endif

#endif
