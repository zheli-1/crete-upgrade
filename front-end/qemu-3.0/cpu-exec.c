/*
 *  emulator main execution loop
 *
 *  Copyright (c) 2003-2005 Fabrice Bellard
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */
#include "config.h"
#include "cpu.h"
#include "trace.h"
#include "disas/disas.h"
#include "tcg.h"
#include "qemu/atomic.h"
#include "sysemu/qtest.h"
#include "qemu/timer.h"
#include "exec/address-spaces.h"
#include "exec/memory-internal.h"
#include "qemu/rcu.h"

#if defined(CRETE_CONFIG) || 1
#include "runtime-dump/runtime-dump.h"
#include "runtime-dump/tci_analyzer.h"
#include "runtime-dump/crete-debug.h"
#endif //#if defined(CRETE_CONFIG)

/* -icount align implementation. */

typedef struct SyncClocks {
    int64_t diff_clk;
    int64_t last_cpu_icount;
    int64_t realtime_clock;
} SyncClocks;

#if !defined(CONFIG_USER_ONLY)
/* Allow the guest to have a max 3ms advance.
 * The difference between the 2 clocks could therefore
 * oscillate around 0.
 */
#define VM_CLOCK_ADVANCE 3000000
#define THRESHOLD_REDUCE 1.5
#define MAX_DELAY_PRINT_RATE 2000000000LL
#define MAX_NB_PRINTS 100

static void align_clocks(SyncClocks *sc, const CPUState *cpu)
{
    int64_t cpu_icount;

    if (!icount_align_option) {
        return;
    }

    cpu_icount = cpu->icount_extra + cpu->icount_decr.u16.low;
    sc->diff_clk += cpu_icount_to_ns(sc->last_cpu_icount - cpu_icount);
    sc->last_cpu_icount = cpu_icount;

    if (sc->diff_clk > VM_CLOCK_ADVANCE) {
#ifndef _WIN32
        struct timespec sleep_delay, rem_delay;
        sleep_delay.tv_sec = sc->diff_clk / 1000000000LL;
        sleep_delay.tv_nsec = sc->diff_clk % 1000000000LL;
        if (nanosleep(&sleep_delay, &rem_delay) < 0) {
            sc->diff_clk = rem_delay.tv_sec * 1000000000LL + rem_delay.tv_nsec;
        } else {
            sc->diff_clk = 0;
        }
#else
        Sleep(sc->diff_clk / SCALE_MS);
        sc->diff_clk = 0;
#endif
    }
}

static void print_delay(const SyncClocks *sc)
{
    static float threshold_delay;
    static int64_t last_realtime_clock;
    static int nb_prints;

    if (icount_align_option &&
        sc->realtime_clock - last_realtime_clock >= MAX_DELAY_PRINT_RATE &&
        nb_prints < MAX_NB_PRINTS) {
        if ((-sc->diff_clk / (float)1000000000LL > threshold_delay) ||
            (-sc->diff_clk / (float)1000000000LL <
             (threshold_delay - THRESHOLD_REDUCE))) {
            threshold_delay = (-sc->diff_clk / 1000000000LL) + 1;
            printf("Warning: The guest is now late by %.1f to %.1f seconds\n",
                   threshold_delay - 1,
                   threshold_delay);
            nb_prints++;
            last_realtime_clock = sc->realtime_clock;
        }
    }
}

static void init_delay_params(SyncClocks *sc,
                              const CPUState *cpu)
{
    if (!icount_align_option) {
        return;
    }
    sc->realtime_clock = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL_RT);
    sc->diff_clk = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) - sc->realtime_clock;
    sc->last_cpu_icount = cpu->icount_extra + cpu->icount_decr.u16.low;
    if (sc->diff_clk < max_delay) {
        max_delay = sc->diff_clk;
    }
    if (sc->diff_clk > max_advance) {
        max_advance = sc->diff_clk;
    }

    /* Print every 2s max if the guest is late. We limit the number
       of printed messages to NB_PRINT_MAX(currently 100) */
    print_delay(sc);
}
#else
static void align_clocks(SyncClocks *sc, const CPUState *cpu)
{
}

static void init_delay_params(SyncClocks *sc, const CPUState *cpu)
{
}
#endif /* CONFIG USER ONLY */

void cpu_loop_exit(CPUState *cpu)
{
    cpu->current_tb = NULL;
    siglongjmp(cpu->jmp_env, 1);
}

/* exit the current TB from a signal handler. The host registers are
   restored in a state compatible with the CPU emulator
 */
#if defined(CONFIG_SOFTMMU)
void cpu_resume_from_signal(CPUState *cpu, void *puc)
{
    /* XXX: restore cpu registers saved in host registers */

    cpu->exception_index = -1;
    siglongjmp(cpu->jmp_env, 1);
}

void cpu_reload_memory_map(CPUState *cpu)
{
    AddressSpaceDispatch *d;

    if (qemu_in_vcpu_thread()) {
        /* Do not let the guest prolong the critical section as much as it
         * as it desires.
         *
         * Currently, this is prevented by the I/O thread's periodinc kicking
         * of the VCPU thread (iothread_requesting_mutex, qemu_cpu_kick_thread)
         * but this will go away once TCG's execution moves out of the global
         * mutex.
         *
         * This pair matches cpu_exec's rcu_read_lock()/rcu_read_unlock(), which
         * only protects cpu->as->dispatch.  Since we reload it below, we can
         * split the critical section.
         */
        rcu_read_unlock();
        rcu_read_lock();
    }

    /* The CPU and TLB are protected by the iothread lock.  */
    d = atomic_rcu_read(&cpu->as->dispatch);
    cpu->memory_dispatch = d;
    tlb_flush(cpu, 1);
}
#endif

/* Execute a TB, and fix up the CPU state afterwards if necessary */
static inline tcg_target_ulong cpu_tb_exec(CPUState *cpu, uint8_t *tb_ptr)
{
    CPUArchState *env = cpu->env_ptr;
    uintptr_t next_tb;

#if defined(CRETE_DEBUG_GENERAL)
    if( is_in_list_crete_dbg_tb_pc(rt_dump_tb->pc))
    {
        fprintf(stderr, "\tcpu_tb_exec(): ");
        {
            uint64_t i;
            const uint8_t *ptr_addr = (const uint8_t*)(&env->CRETE_DBG_REG);
            uint8_t byte_value;
            fprintf(stderr, "env->" CRETE_GET_STRING(CRETE_DBG_REG) " = [");
            for(i = 0; i < sizeof(env->CRETE_DBG_REG); ++i) {
                byte_value = *(ptr_addr + i);
                fprintf(stderr, " 0x%02x", byte_value);
            }
            fprintf(stderr, "]\n");
        }
    }
#endif

#if defined(DEBUG_DISAS)
    if (qemu_loglevel_mask(CPU_LOG_TB_CPU)) {
#if defined(TARGET_I386)
        log_cpu_state(cpu, CPU_DUMP_CCOP);
#elif defined(TARGET_M68K)
        /* ??? Should not modify env state for dumping.  */
        cpu_m68k_flush_flags(env, env->cc_op);
        env->cc_op = CC_OP_FLAGS;
        env->sr = (env->sr & 0xffe0) | env->cc_dest | (env->cc_x << 4);
        log_cpu_state(cpu, 0);
#else
        log_cpu_state(cpu, 0);
#endif
    }
#endif /* DEBUG_DISAS */

    cpu->can_do_io = 0;

#if defined(CRETE_DEP_ANALYSIS) || 1
    if(f_crete_enabled)
    {
        next_tb = crete_tcg_qemu_tb_exec(env, tb_ptr);

#if defined(CRETE_DEBUG)
        fprintf(stderr, "+++\n");
        tci_analyzer_print();
        fprintf(stderr, "+++\n");
#endif
    }
    else
    {
        next_tb = tcg_qemu_tb_exec(env, tb_ptr);
    }
#else // !defined(CRETE_DEP_ANALYSIS)
    next_tb = tcg_qemu_tb_exec(env, tb_ptr);
#endif // defined(CRETE_DEP_ANALYSIS)

    cpu->can_do_io = 1;
    trace_exec_tb_exit((void *) (next_tb & ~TB_EXIT_MASK),
                       next_tb & TB_EXIT_MASK);

    if ((next_tb & TB_EXIT_MASK) > TB_EXIT_IDX1) {
        /* We didn't start executing this TB (eg because the instruction
         * counter hit zero); we must restore the guest PC to the address
         * of the start of the TB.
         */
        CPUClass *cc = CPU_GET_CLASS(cpu);
        TranslationBlock *tb = (TranslationBlock *)(next_tb & ~TB_EXIT_MASK);
        if (cc->synchronize_from_tb) {
            cc->synchronize_from_tb(cpu, tb);
        } else {
            assert(cc->set_pc);
            cc->set_pc(cpu, tb->pc);
        }
    }
    if ((next_tb & TB_EXIT_MASK) == TB_EXIT_REQUESTED) {
        /* We were asked to stop executing TBs (probably a pending
         * interrupt. We've now stopped, so clear the flag.
         */
        cpu->tcg_exit_req = 0;
    }
    return next_tb;
}

/* Execute the code without caching the generated code. An interpreter
   could be used if available. */
static void cpu_exec_nocache(CPUArchState *env, int max_cycles,
                             TranslationBlock *orig_tb)
{
    CPUState *cpu = ENV_GET_CPU(env);
    TranslationBlock *tb;
    target_ulong pc = orig_tb->pc;
    target_ulong cs_base = orig_tb->cs_base;
    uint64_t flags = orig_tb->flags;

    /* Should never happen.
       We only end up here when an existing TB is too long.  */
    if (max_cycles > CF_COUNT_MASK)
        max_cycles = CF_COUNT_MASK;

    /* tb_gen_code can flush our orig_tb, invalidate it now */
    tb_phys_invalidate(orig_tb, -1);
    tb = tb_gen_code(cpu, pc, cs_base, flags,
                     max_cycles | CF_NOCACHE);
    cpu->current_tb = tb;
    /* execute the generated code */
    trace_exec_tb_nocache(tb, tb->pc);
    cpu_tb_exec(cpu, tb->tc_ptr);
    cpu->current_tb = NULL;
    tb_phys_invalidate(tb, -1);
    tb_free(tb);
}

static TranslationBlock *tb_find_slow(CPUArchState *env,
                                      target_ulong pc,
                                      target_ulong cs_base,
                                      uint64_t flags)
{
    CPUState *cpu = ENV_GET_CPU(env);
    TranslationBlock *tb, **ptb1;
    unsigned int h;
    tb_page_addr_t phys_pc, phys_page1;
    target_ulong virt_page2;

    tcg_ctx.tb_ctx.tb_invalidated_flag = 0;

    /* find translated block using physical mappings */
    phys_pc = get_page_addr_code(env, pc);
    phys_page1 = phys_pc & TARGET_PAGE_MASK;
    h = tb_phys_hash_func(phys_pc);
    ptb1 = &tcg_ctx.tb_ctx.tb_phys_hash[h];
    for(;;) {
        tb = *ptb1;
        if (!tb)
            goto not_found;
        if (tb->pc == pc &&
            tb->page_addr[0] == phys_page1 &&
            tb->cs_base == cs_base &&
            tb->flags == flags) {
            /* check next page if needed */
            if (tb->page_addr[1] != -1) {
                tb_page_addr_t phys_page2;

                virt_page2 = (pc & TARGET_PAGE_MASK) +
                    TARGET_PAGE_SIZE;
                phys_page2 = get_page_addr_code(env, virt_page2);
                if (tb->page_addr[1] == phys_page2)
                    goto found;
            } else {
                goto found;
            }
        }
        ptb1 = &tb->phys_hash_next;
    }
 not_found:
   /* if no translated code available, then translate it now */
    tb = tb_gen_code(cpu, pc, cs_base, flags, 0);

 found:
    /* Move the last found TB to the head of the list */
    if (likely(*ptb1)) {
        *ptb1 = tb->phys_hash_next;
        tb->phys_hash_next = tcg_ctx.tb_ctx.tb_phys_hash[h];
        tcg_ctx.tb_ctx.tb_phys_hash[h] = tb;
    }
    /* we add the TB in the virtual pc hash table */
    cpu->tb_jmp_cache[tb_jmp_cache_hash_func(pc)] = tb;
    return tb;
}

static inline TranslationBlock *tb_find_fast(CPUArchState *env)
{
    CPUState *cpu = ENV_GET_CPU(env);
    TranslationBlock *tb;
    target_ulong cs_base, pc;
    int flags;

    /* we record a subset of the CPU state. It will
       always be the same before a given translated block
       is executed. */
    cpu_get_tb_cpu_state(env, &pc, &cs_base, &flags);
    tb = cpu->tb_jmp_cache[tb_jmp_cache_hash_func(pc)];
    if (unlikely(!tb || tb->pc != pc || tb->cs_base != cs_base ||
                 tb->flags != flags)) {
        tb = tb_find_slow(env, pc, cs_base, flags);
    }
    return tb;
}

static void cpu_handle_debug_exception(CPUArchState *env)
{
    CPUState *cpu = ENV_GET_CPU(env);
    CPUClass *cc = CPU_GET_CLASS(cpu);
    CPUWatchpoint *wp;

    if (!cpu->watchpoint_hit) {
        QTAILQ_FOREACH(wp, &cpu->watchpoints, entry) {
            wp->flags &= ~BP_WATCHPOINT_HIT;
        }
    }

    cc->debug_excp_handler(cpu);
}

/* main execution loop */

volatile sig_atomic_t exit_request;

int cpu_exec(CPUArchState *env)
{
    CPUState *cpu = ENV_GET_CPU(env);
    CPUClass *cc = CPU_GET_CLASS(cpu);
#ifdef TARGET_I386
    X86CPU *x86_cpu = X86_CPU(cpu);
#endif
    int ret, interrupt_request;
    TranslationBlock *tb;
    uint8_t *tc_ptr;
    uintptr_t next_tb;
    SyncClocks sc;

    /* This must be volatile so it is not trashed by longjmp() */
    volatile bool have_tb_lock = false;

    if (cpu->halted) {
        if (!cpu_has_work(cpu)) {
            return EXCP_HALTED;
        }

        cpu->halted = 0;
    }

    current_cpu = cpu;

    /* As long as current_cpu is null, up to the assignment just above,
     * requests by other threads to exit the execution loop are expected to
     * be issued using the exit_request global. We must make sure that our
     * evaluation of the global value is performed past the current_cpu
     * value transition point, which requires a memory barrier as well as
     * an instruction scheduling constraint on modern architectures.  */
    smp_mb();

    rcu_read_lock();

    if (unlikely(exit_request)) {
        cpu->exit_request = 1;
    }

    cc->cpu_exec_enter(cpu);

    /* Calculate difference between guest clock and host clock.
     * This delay includes the delay of the last cycle, so
     * what we have to do is sleep until it is 0. As for the
     * advance/delay we gain here, we try to fix it next time.
     */
    init_delay_params(&sc, cpu);

    /* prepare setjmp context for exception handling */
    for(;;) {
        CRETE_DBG_GEN(
        fprintf(stderr, "=== Start: cpu_exec() ext loop: "
                "interrupt_request = %u, exception_index = %d\n",
                cpu->interrupt_request, cpu->exception_index);
        );

        if (sigsetjmp(cpu->jmp_env, 0) == 0) {
            /* if an exception is pending, we execute it here */
            if (cpu->exception_index >= 0) {
                if (cpu->exception_index >= EXCP_INTERRUPT) {
                    /* exit request from the cpu execution loop
                     * BOBO: caused by a hardware interrupt */
                    ret = cpu->exception_index;
                    if (ret == EXCP_DEBUG) {
                        cpu_handle_debug_exception(env);
                    }
                    cpu->exception_index = -1;
                    break;
                } else {
#if defined(CONFIG_USER_ONLY)
                    /* if user mode only, we simulate a fake exception
                       which will be handled outside the cpu execution
                       loop */
#if defined(TARGET_I386)
                    cc->do_interrupt(cpu);
#endif
                    ret = cpu->exception_index;
                    cpu->exception_index = -1;
                    break;
#else
                    /* BOBO: handling exception within cpu execution loop */
                    cc->do_interrupt(cpu);
                    cpu->exception_index = -1;
#endif
                }
            }

            next_tb = 0; /* force lookup of first TB */
            for(;;) {
                CRETE_DBG_GEN(
                fprintf(stderr, "--- Start: cpu_exec() inner loop: "
                        "interrupt_request = %u, exception_index = %d\n",
                        cpu->interrupt_request, cpu->exception_index);
                );

                interrupt_request = cpu->interrupt_request;
                if (unlikely(interrupt_request)) {
                    if (unlikely(cpu->singlestep_enabled & SSTEP_NOIRQ)) {
                        /* Mask out external interrupts for this step. */
                        interrupt_request &= ~CPU_INTERRUPT_SSTEP_MASK;
                    }
                    if (interrupt_request & CPU_INTERRUPT_DEBUG) {
                        cpu->interrupt_request &= ~CPU_INTERRUPT_DEBUG;
                        cpu->exception_index = EXCP_DEBUG;
                        cpu_loop_exit(cpu);
                    }
                    if (interrupt_request & CPU_INTERRUPT_HALT) {
                        cpu->interrupt_request &= ~CPU_INTERRUPT_HALT;
                        cpu->halted = 1;
                        cpu->exception_index = EXCP_HLT;
                        cpu_loop_exit(cpu);
                    }
#if defined(TARGET_I386)
                    if (interrupt_request & CPU_INTERRUPT_INIT) {
                        cpu_svm_check_intercept_param(env, SVM_EXIT_INIT, 0);
                        do_cpu_init(x86_cpu);
                        cpu->exception_index = EXCP_HALTED;
                        cpu_loop_exit(cpu);
                    }
#else
                    if (interrupt_request & CPU_INTERRUPT_RESET) {
                        cpu_reset(cpu);
                    }
#endif
                    /* The target hook has 3 exit conditions:
                       False when the interrupt isn't processed,
                       True when it is, and we should restart on a new TB,
                       and via longjmp via cpu_loop_exit.  */
                    if (cc->cpu_exec_interrupt(cpu, interrupt_request)) {
                        next_tb = 0;
                    }
                    /* Don't use the cached interrupt_request value,
                       do_interrupt may have updated the EXITTB flag. */
                    if (cpu->interrupt_request & CPU_INTERRUPT_EXITTB) {
                        cpu->interrupt_request &= ~CPU_INTERRUPT_EXITTB;
                        /* ensure that no TB jump will be modified as
                           the program flow was changed */
                        next_tb = 0;
                    }
                }
                if (unlikely(cpu->exit_request)) {
                    cpu->exit_request = 0;
                    cpu->exception_index = EXCP_INTERRUPT;
                    cpu_loop_exit(cpu);
                }
                spin_lock(&tcg_ctx.tb_ctx.tb_lock);
                have_tb_lock = true;
                tb = tb_find_fast(env);
                /* Note: we do it here to avoid a gcc bug on Mac OS X when
                   doing it in tb_find_slow */
                if (tcg_ctx.tb_ctx.tb_invalidated_flag) {
                    /* as some TB could have been invalidated because
                       of memory exceptions while generating the code, we
                       must recompute the hash index here */
                    next_tb = 0;
                    tcg_ctx.tb_ctx.tb_invalidated_flag = 0;
                }
                if (qemu_loglevel_mask(CPU_LOG_EXEC)) {
                    qemu_log("Trace %p [" TARGET_FMT_lx "] %s\n",
                             tb->tc_ptr, tb->pc, lookup_symbol(tb->pc));
                }
                /* see if we can patch the calling TB. When the TB
                   spans two pages, we cannot safely do a direct
                   jump. */
                if (next_tb != 0 && tb->page_addr[1] == -1) {
                    tb_add_jump((TranslationBlock *)(next_tb & ~TB_EXIT_MASK),
                                next_tb & TB_EXIT_MASK, tb);
                }
                have_tb_lock = false;
                spin_unlock(&tcg_ctx.tb_ctx.tb_lock);

                /* cpu_interrupt might be called while translating the
                   TB, but before it is linked into a potentially
                   infinite loop and becomes env->current_tb. Avoid
                   starting execution if there is a pending interrupt. */
                cpu->current_tb = tb;
                barrier();
                if (likely(!cpu->exit_request)) {
                    trace_exec_tb(tb, tb->pc);
                    tc_ptr = tb->tc_ptr;

#if defined(CRETE_CONFIG) || 1
                    if(cpu->env_ptr != env)
                        assert(0);

                    crete_pre_cpu_tb_exec((void *)cpu->env_ptr, tb);
#endif // #if defined(CRETE_CONFIG)

                    /* execute the generated code */
                    next_tb = cpu_tb_exec(cpu, tc_ptr);

#if defined(CRETE_CONFIG) || 1
                    if(cpu->env_ptr != env)
                        assert(0);

                    crete_post_cpu_tb_exec(cpu->env_ptr, tb, next_tb, 0);
                    assert((next_tb&TB_EXIT_MASK) != TB_EXIT_ICOUNT_EXPIRED);

                    /* Disable run-time dump after the execution of current TB */
                    flag_rt_dump_enable = 0;
#endif // #if defined(CRETE_CONFIG)

                    switch (next_tb & TB_EXIT_MASK) {
                    case TB_EXIT_REQUESTED:
                        /* Something asked us to stop executing
                         * chained TBs; just continue round the main
                         * loop. Whatever requested the exit will also
                         * have set something else (eg exit_request or
                         * interrupt_request) which we will handle
                         * next time around the loop.
                         */
                        next_tb = 0;
                        break;
                    case TB_EXIT_ICOUNT_EXPIRED:
                    {
                        /* Instruction counter expired.  */
                        int insns_left = cpu->icount_decr.u32;
                        if (cpu->icount_extra && insns_left >= 0) {
                            /* Refill decrementer and continue execution.  */
                            cpu->icount_extra += insns_left;
                            insns_left = MIN(0xffff, cpu->icount_extra);
                            cpu->icount_extra -= insns_left;
                            cpu->icount_decr.u16.low = insns_left;
                        } else {
                            if (insns_left > 0) {
                                /* Execute remaining instructions.  */
                                tb = (TranslationBlock *)(next_tb & ~TB_EXIT_MASK);
                                cpu_exec_nocache(env, insns_left, tb);
                                align_clocks(&sc, cpu);
                            }
                            cpu->exception_index = EXCP_INTERRUPT;
                            next_tb = 0;
                            cpu_loop_exit(cpu);
                        }
                        break;
                    }
                    default:
                        break;
                    }
                }
                cpu->current_tb = NULL;
                /* Try to align the host and virtual clocks
                   if the guest is in advance */
                align_clocks(&sc, cpu);
                /* reset soft MMU for next block (it can currently
                   only be set by a memory fault) */

                CRETE_DBG_GEN(
                fprintf(stderr, "--- End: cpu_exec() inner loop : "
                        "interrupt_request = %u, exception_index = %d\n",
                        cpu->interrupt_request, cpu->exception_index);
                );
            } /* for(;;) */
        } else {
            /* Reload env after longjmp - the compiler may have smashed all
             * local variables as longjmp is marked 'noreturn'. */
            cpu = current_cpu;
            env = cpu->env_ptr;
            cc = CPU_GET_CLASS(cpu);
            cpu->can_do_io = 1;
#ifdef TARGET_I386
            x86_cpu = X86_CPU(cpu);
#endif
            if (have_tb_lock) {
                spin_unlock(&tcg_ctx.tb_ctx.tb_lock);
                have_tb_lock = false;
            }
        }

        CRETE_DBG_GEN(
        fprintf(stderr, "=== Finish: cpu_exec() ext loop: "
                "interrupt_request = %u, exception_index = %d\n",
                cpu->interrupt_request, cpu->exception_index);
        );
    } /* for(;;) */

    cc->cpu_exec_exit(cpu);
    rcu_read_unlock();

    /* fail safe : never use current_cpu outside cpu_exec() */
    current_cpu = NULL;
    return ret;
}

#if defined(CRETE_CROSS_CHECK)

#define __ADD_C_CPUSTATE_OFFSET(in_type, in_name)                           \
        offset = offsetof(CPUArchState, in_name);                           \
        size = sizeof(in_type);                                             \
        crete_add_c_cpuState_offset(offset, size);

#define __ADD_C_CPUSTATE_OFFSET_ARRAY(in_type, in_name, array_size)         \
        for(i = 0; i < (array_size); ++i)                                   \
        {                                                                   \
            offset = offsetof(CPUArchState, in_name) + i*sizeof(in_type);   \
            size = sizeof(in_type);                                         \
            crete_add_c_cpuState_offset(offset, size);                      \
        }

void init_C_CPUArchState_offset(void);
void init_C_CPUArchState_offset(void)
{
    uint64_t i = 0;
    uint64_t offset = 0;
    uint64_t size = 0;

    __ADD_C_CPUSTATE_OFFSET(target_ulong, regs);

    __ADD_C_CPUSTATE_OFFSET(target_ulong, eflags)

    /* emulator internal eflags handling */
    // target_ulong cc_dst;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, cc_dst)
    // target_ulong cc_src;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, cc_src)
    // target_ulong cc_src2;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, cc_src2)
    //uint32_t cc_op;
    __ADD_C_CPUSTATE_OFFSET(uint32_t, cc_op);
    // int32_t df;
    __ADD_C_CPUSTATE_OFFSET(int32_t, df)
    // uint32_t hflags;
    __ADD_C_CPUSTATE_OFFSET(uint32_t, hflags)
    // uint32_t hflags2;
    __ADD_C_CPUSTATE_OFFSET(uint32_t, hflags2)

    /* segments */
    // SegmentCache segs[6];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(SegmentCache, segs, 6)

    // SegmentCache ldt;
    __ADD_C_CPUSTATE_OFFSET(SegmentCache, ldt)
    // SegmentCache tr;
    __ADD_C_CPUSTATE_OFFSET(SegmentCache, tr)
    // SegmentCache gdt;
    __ADD_C_CPUSTATE_OFFSET(SegmentCache, gdt)
    // SegmentCache idt;
    __ADD_C_CPUSTATE_OFFSET(SegmentCache, idt)

    // xxx: not traced
    // target_ulong cr[5];
    //    __ADD_C_CPUSTATE_OFFSET_ARRAY(target_ulong, cr, 5)
    // int32_t a20_mask;
    __ADD_C_CPUSTATE_OFFSET(int32_t, a20_mask)

    // BNDReg bnd_regs[4];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(BNDReg, bnd_regs, 4)
    // BNDCSReg bndcs_regs;
    __ADD_C_CPUSTATE_OFFSET(BNDCSReg, bndcs_regs)
    // uint64_t msr_bndcfgs;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_bndcfgs)

    /* Beginning of state preserved by INIT (dummy marker).  */
    // xxx: not traced
    //    struct {} start_init_save;
    //    __ADD_C_CPUSTATE_OFFSET(struct {}, start_init_save)

    /* FPU state */
    // unsigned int fpstt;
    __ADD_C_CPUSTATE_OFFSET(unsigned int, fpstt)
    // uint16_t fpus;
    __ADD_C_CPUSTATE_OFFSET(uint16_t, fpus)
    // uint16_t fpuc;
    __ADD_C_CPUSTATE_OFFSET(uint16_t, fpuc)
    // uint8_t fptags[8];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(uint8_t, fptags, 8)
    // FPReg fpregs[8];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(FPReg, fpregs, 8)
    /* KVM-only so far */
    // uint16_t fpop;
    __ADD_C_CPUSTATE_OFFSET(uint16_t, fpop)
    // uint64_t fpip;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, fpip)
    // uint64_t fpdp;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, fpdp)

    /* emulator internal variables */
    // float_status fp_status;
    __ADD_C_CPUSTATE_OFFSET(float_status, fp_status)
    // floatx80 ft0;
    __ADD_C_CPUSTATE_OFFSET(floatx80, ft0)

    // float_status mmx_status;
    __ADD_C_CPUSTATE_OFFSET(float_status, mmx_status)
    // float_status sse_status;
    __ADD_C_CPUSTATE_OFFSET(float_status, sse_status)
    // uint32_t mxcsr;
    __ADD_C_CPUSTATE_OFFSET(uint32_t, mxcsr)
    // XMMReg xmm_regs[CPU_NB_REGS == 8 ? 8 : 32];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(XMMReg, xmm_regs, CPU_NB_REGS == 8 ? 8 : 32)
    // XMMReg xmm_t0;
    __ADD_C_CPUSTATE_OFFSET(XMMReg, xmm_t0)
    // MMXReg mmx_t0;
    __ADD_C_CPUSTATE_OFFSET(MMXReg, mmx_t0)

    // uint64_t opmask_regs[NB_OPMASK_REGS];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(uint64_t, opmask_regs, NB_OPMASK_REGS)

    /* sysenter registers */
    // uint32_t sysenter_cs;
    __ADD_C_CPUSTATE_OFFSET(uint32_t, sysenter_cs)
    // target_ulong sysenter_esp;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, sysenter_esp)
    // target_ulong sysenter_eip;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, sysenter_eip)
    // uint64_t efer;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, efer)
    // uint64_t star;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, star)

    // uint64_t vm_hsave;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, vm_hsave)

#ifdef TARGET_X86_64
    // target_ulong lstar;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, lstar)
    // target_ulong cstar;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, cstar)
    // target_ulong fmask;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, fmask)
    // target_ulong kernelgsbase;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, kernelgsbase)
#endif

    // uint64_t tsc;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, tsc)
    // uint64_t tsc_adjust;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, tsc_adjust)
    // uint64_t tsc_deadline;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, tsc_deadline)

    // uint64_t mcg_status;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, mcg_status)
    // uint64_t msr_ia32_misc_enable;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_ia32_misc_enable)
    // uint64_t msr_ia32_feature_control;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_ia32_feature_control)

    // uint64_t msr_fixed_ctr_ctrl;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_fixed_ctr_ctrl)
    // uint64_t msr_global_ctrl;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_global_ctrl)
    // uint64_t msr_global_status;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_global_status)
    // uint64_t msr_global_ovf_ctrl;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_global_ovf_ctrl)
    // uint64_t msr_fixed_counters[MAX_FIXED_COUNTERS];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(uint64_t, msr_fixed_counters, MAX_FIXED_COUNTERS)
    // uint64_t msr_gp_counters[MAX_GP_COUNTERS];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(uint64_t, msr_gp_counters, MAX_GP_COUNTERS)
    // uint64_t msr_gp_evtsel[MAX_GP_COUNTERS];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(uint64_t, msr_gp_evtsel, MAX_GP_COUNTERS)

    // uint64_t pat;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, pat)
    // uint32_t smbase;
    __ADD_C_CPUSTATE_OFFSET(uint32_t, smbase)

    /* End of state preserved by INIT (dummy marker).  */
    // xxx: not traced
    //    struct {} end_init_save;
    //    __ADD_C_CPUSTATE_OFFSET(struct {}, end_init_save)

    // uint64_t system_time_msr;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, system_time_msr)
    // uint64_t wall_clock_msr;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, wall_clock_msr)
    // uint64_t steal_time_msr;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, steal_time_msr)
    // uint64_t async_pf_en_msr;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, async_pf_en_msr)
    // uint64_t pv_eoi_en_msr;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, pv_eoi_en_msr)

    // uint64_t msr_hv_hypercall;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_hv_hypercall)
    // uint64_t msr_hv_guest_os_id;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_hv_guest_os_id)
    // uint64_t msr_hv_vapic;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_hv_vapic)
    // uint64_t msr_hv_tsc;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, msr_hv_tsc)

    /* exception/interrupt handling */
    //xxx: not traced
    // int error_code;
    //    __ADD_C_CPUSTATE_OFFSET(int, error_code)
    // int exception_is_int;
    __ADD_C_CPUSTATE_OFFSET(int, exception_is_int)
    // target_ulong exception_next_eip;
    __ADD_C_CPUSTATE_OFFSET(target_ulong, exception_next_eip)
    // target_ulong dr[8];
    __ADD_C_CPUSTATE_OFFSET_ARRAY(target_ulong, dr, 8)
    //xxx: not traced
    //    union {
    //        struct CPUBreakpoint *cpu_breakpoint[4];
    //        struct CPUWatchpoint *cpu_watchpoint[4];
    //    };
    // int old_exception;
    __ADD_C_CPUSTATE_OFFSET(int, old_exception)
    // uint64_t vm_vmcb;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, vm_vmcb)
    // uint64_t tsc_offset;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, tsc_offset)
    // uint64_t intercept;
    __ADD_C_CPUSTATE_OFFSET(uint64_t, intercept)
    // uint16_t intercept_cr_read;
    __ADD_C_CPUSTATE_OFFSET(uint16_t, intercept_cr_read)
    // uint16_t intercept_cr_write;
    __ADD_C_CPUSTATE_OFFSET(uint16_t, intercept_cr_write)
    // uint16_t intercept_dr_read;
    __ADD_C_CPUSTATE_OFFSET(uint16_t, intercept_dr_read)
    // uint16_t intercept_dr_write;
    __ADD_C_CPUSTATE_OFFSET(uint16_t, intercept_dr_write)
    // uint32_t intercept_exceptions;
    __ADD_C_CPUSTATE_OFFSET(uint32_t, intercept_exceptions)
    // uint8_t v_tpr;
    __ADD_C_CPUSTATE_OFFSET(uint8_t, v_tpr)

    /* KVM states, automatically cleared on reset */
    // uint8_t nmi_injected;
    __ADD_C_CPUSTATE_OFFSET(uint8_t, nmi_injected)
    // uint8_t nmi_pending;
    __ADD_C_CPUSTATE_OFFSET(uint8_t, nmi_pending)

}
#endif //#if defined(CRETE_CROSS_CHECK)