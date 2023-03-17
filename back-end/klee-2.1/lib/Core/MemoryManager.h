//===-- MemoryManager.h -----------------------------------------*- C++ -*-===//
//
//                     The KLEE Symbolic Virtual Machine
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef KLEE_MEMORYMANAGER_H
#define KLEE_MEMORYMANAGER_H

#include <cstddef>
#include <set>
#include <cstdint>
#include <unordered_map>

#if defined(CRETE_CONFIG)

#define PAGE_ALIGN_BITS     (12)
#define PAGE_SIZE           (1<<PAGE_ALIGN_BITS)
#define PAGE_ADDRESS_MASK   (0xffffffffffffffffLL << PAGE_ALIGN_BITS)
#define PAGE_OFFSET_MASK    (~PAGE_ADDRESS_MASK)

#endif //defined(CRETE_CONFIG)


namespace llvm {
class Value;
}

namespace klee {
class MemoryObject;
class ArrayCache;

class MemoryManager {
private:
  typedef std::set<MemoryObject *> objects_ty;
  objects_ty objects;
  ArrayCache *const arrayCache;

  char *deterministicSpace;
  char *nextFreeSlot;
  size_t spaceSize;

public:
  MemoryManager(ArrayCache *arrayCache);
  ~MemoryManager();

  /**
   * Returns memory object which contains a handle to real virtual process
   * memory.
   */
  MemoryObject *allocate(uint64_t size, bool isLocal, bool isGlobal,
                         const llvm::Value *allocSite, size_t alignment);
  MemoryObject *allocateFixed(uint64_t address, uint64_t size,
                              const llvm::Value *allocSite);
  void deallocate(const MemoryObject *mo);
  void markFreed(MemoryObject *mo);
  ArrayCache *getArrayCache() const { return arrayCache; }

  /*
   * Returns the size used by deterministic allocation in bytes
   */
  size_t getUsedDeterministicSize();

#if defined(CRETE_CONFIG)
public:
bool find_dynamic_page_mo(uint64_t static_addr, MemoryObject *&ret_mo)
{
  //    fprintf(stderr, "PAGE_SIZE = %lu, PAGE_ADDRESS_MASK = %p, PAGE_OFFSET_MASK = %p\n",
//            (uint64_t)PAGE_SIZE, (void *)(uint64_t)PAGE_ADDRESS_MASK, (void *)(uint64_t)PAGE_OFFSET_MASK);

    bool ret = true;
    uint64_t static_page_addr = static_addr & PAGE_ADDRESS_MASK;

    if(m_dyn_addr_map.find(static_page_addr) == m_dyn_addr_map.end())
    {
        ret = false;
        m_dyn_addr_map[static_page_addr] = allocate(PAGE_SIZE, false, true, 0, PAGE_SIZE);

//        fprintf(stderr, "find_dynamic_page_mo(): create new page mo: static_page_addr = %p, page_mo_addr = %p\n",
//                (void *)static_page_addr, (void *)m_dyn_addr_map[static_page_addr]->address);
    }

    ret_mo = m_dyn_addr_map[static_page_addr];
    return ret;
}
#endif
#if defined(CRETE_CONFIG)
  private:
    //  <static_addr, MemoryObject (with dynamic address)>
    std::unordered_map<uint64_t, MemoryObject *> m_dyn_addr_map;

#endif //defined(CRETE_CONFIG)

};

} // End klee namespace

#endif /* KLEE_MEMORYMANAGER_H */
