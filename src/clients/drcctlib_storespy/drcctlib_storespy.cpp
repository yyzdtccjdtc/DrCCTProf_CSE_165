/* 
 *  Copyright (c) 2020-2021 Xuhpclab. All rights reserved.
 *  Licensed under the MIT License.
 *  See LICENSE file for more information.
 */

#include <cstddef>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <unordered_map>
#include <sys/mman.h>
#include <functional>
#include <vector>
#include <algorithm>
#include <xmmintrin.h>
#include <immintrin.h>

#include "dr_api.h"
#include "drmgr.h"
#include "drreg.h"
#include "drutil.h"
#include "drcctlib.h"
#include "shadow_memory_ml.h"

using namespace std;

/***********************************************
 ******  shadow memory
 ************************************************/
ConcurrentShadowMemoryMl<uint8_t, uint64_t, context_handle_t> sm; 

#define DRCCTLIB_PRINTF(_FORMAT, _ARGS...) \
    DRCCTLIB_PRINTF_TEMPLATE("memory_with_addr_and_refsize_clean_call", _FORMAT, ##_ARGS)
#define DRCCTLIB_EXIT_PROCESS(_FORMAT, _ARGS...)                                           \
    DRCCTLIB_CLIENT_EXIT_PROCESS_TEMPLATE("memory_with_addr_and_refsize_clean_call", _FORMAT, \
                                          ##_ARGS)

static int tls_idx;
static int memOp_num; 
static int zero = 0;

uint64_t unreadCount;
uint64_t unreadBytes;

thread_local bool Sample_flag = true;
thread_local long long NUM_INS = 0;

#define TLS_MEM_REF_BUFF_SIZE 100
#ifdef SAMPLE_RUN
#   define WINDOW_ENABLE 1000000
#   define WINDOW_DISABLE 100000000
#endif
#define MAX_WRITE_OP_LENGTH (512)
#define MAX_WRITE_OPS_IN_INS (8)
#define THREAD_MAX (1024)

/* infrastructure for shadow memory */
/* MACROs */
// 64KB shadow pages
#define PAGE_OFFSET_BITS (16LL)
#define PAGE_OFFSET(addr) ( addr & 0xFFFF)
#define PAGE_OFFSET_MASK ( 0xFFFF)

#define PAGE_SIZE (1 << PAGE_OFFSET_BITS)
#define delta 0.01
// 2 level page table
#define PTR_SIZE (sizeof(struct Status *))
#define LEVEL_1_PAGE_TABLE_BITS  (20)
#define LEVEL_1_PAGE_TABLE_ENTRIES  (1 << LEVEL_1_PAGE_TABLE_BITS )
#define LEVEL_1_PAGE_TABLE_SIZE  (LEVEL_1_PAGE_TABLE_ENTRIES * PTR_SIZE )

#define LEVEL_2_PAGE_TABLE_BITS  (12)
#define LEVEL_2_PAGE_TABLE_ENTRIES  (1 << LEVEL_2_PAGE_TABLE_BITS )
#define LEVEL_2_PAGE_TABLE_SIZE  (LEVEL_2_PAGE_TABLE_ENTRIES * PTR_SIZE )

#define LEVEL_1_PAGE_TABLE_SLOT(addr) (((addr) >> (LEVEL_2_PAGE_TABLE_BITS + PAGE_OFFSET_BITS)) & 0xfffff)
#define LEVEL_2_PAGE_TABLE_SLOT(addr) (((addr) >> (PAGE_OFFSET_BITS)) & 0xFFF)

#define IS_ACCESS_WITHIN_PAGE_BOUNDARY(accessAddr, accessLen) (PAGE_OFFSET((accessAddr)) <= (PAGE_OFFSET_MASK - (accessLen)))

#define MAKE_CONTEXT_PAIR(a, b) (((uint64_t)(a) << 32) | ((uint64_t)(b)))

#define DECODE_DEAD(data) static_cast<context_handle_t>(((data) & 0xffffffffffffffff) >> 32)
#define DECODE_KILL(data) (static_cast<context_handle_t>( (data) & 0x00000000ffffffff))

#define MAX_CONTEXTS (50)

#define XMM_VEC_LEN (16)
#define YMM_VEC_LEN (32)
#define ZMM_VEC_LEN (64)

static file_t gTraceFile;
static uint8_t** gL1PageTable[LEVEL_1_PAGE_TABLE_SIZE];


enum {
    INSTRACE_TLS_OFFS_BUF_PTR,
    INSTRACE_TLS_COUNT, /* total number of TLS slots allocated */
};
static reg_id_t tls_seg;
static uint tls_offs;
#define TLS_SLOT(tls_base, enum_val) (void **)((byte *)(tls_base) + tls_offs + (enum_val))
#define BUF_PTR(tls_base, type, offs) *(type **)TLS_SLOT(tls_base, offs)
#define MINSERT instrlist_meta_preinsert
#ifdef ARM_CCTLIB
#    define OPND_CREATE_CCT_INT OPND_CREATE_INT
#else
#    define OPND_CREATE_CCT_INT OPND_CREATE_INT32
#endif

typedef struct _mem_ref_t {
    app_pc addr;
    size_t size;
} mem_ref_t;

typedef struct op_ref{
    uint64_t *opAddr;
    uint32_t opSize;
}op_ref;

typedef struct _per_thread_t {
    mem_ref_t *cur_buf_list;
    void *cur_buf;
    op_ref opList[MAX_WRITE_OPS_IN_INS];
    uint64_t value[MAX_WRITE_OPS_IN_INS];
    uint64_t bytesWritten;
    int32_t float_instr_and_ok_to_appx;
    int32_t is_float;
    bool sample_mem;
    uint32_t instr_num;
} per_thread_t;

typedef struct RedanduncyData{
    context_handle_t dead;
    context_handle_t kill;
    uint64_t frequency;
} RedanduncyData;

void *lock;

static unordered_map<uint64_t, uint64_t> RedMap[THREAD_MAX];
static unordered_map<uint64_t, uint64_t> ApproxRedMap[THREAD_MAX];

static void AddToRedTable(uint64_t key, uint16_t value, int threadID){
    // if the pair exits, then update the total bytes. otherwise add a new element
    unordered_map<uint64_t, uint64_t>::iterator it = RedMap[threadID].find(key);
    if (it == RedMap[threadID].end()) {
        RedMap[threadID][key] = value;
    } else {
        it->second += value;
        //dr_fprintf(gTraceFile, "RedTable->first = %llu, RedTable->second = %llu\n", it->first, it->second);
    }
}

static void AddToApproxRedTable(uint64_t key, uint16_t value, int threadID){
    // if the pair exits, then update the total bytes. otherwise add a new element
    unordered_map<uint64_t, uint64_t>::iterator it = ApproxRedMap[threadID].find(key);
    if (it == ApproxRedMap[threadID].end()) {
        ApproxRedMap[threadID][key] = value;
    } else {
        it->second += value;
        //dr_fprintf(gTraceFile, "RedTable->first = %llu, RedTable->second = %llu\n", it->first, it->second);
    }
}

template<int start, int end, int incr>
struct UnrolledLoop{
    static void Body(function<void (const int)> func){
        func(start); // Real loop body
        UnrolledLoop<start+incr, end, incr>::Body(func); //unroll next iteration
    }
};

template<int end, int incr>
struct UnrolledLoop<end, end, incr>{
    static void Body(function<void (const int)> func){
        // empty body
    }
};

template<int start, int end, int incr>
struct UnrolledConjunction{
    static bool Body(function<bool (const int)> func){
        return func(start) && UnrolledConjunction<start+incr, end, incr>::Body(func); // unroll next iteration
    }
};

template<int end, int incr>
struct UnrolledConjunction<end, end, incr>{
    static bool Body(function<void (const int)> func){
        return true;
    }
};

// helper functions for shadow memory
static uint8_t* GetOrCreateShadowBaseAddress(uint64_t addr){
    uint8_t *shadowPage;
    uint8_t** *l1Ptr = &gL1PageTable[LEVEL_1_PAGE_TABLE_SLOT(addr)];

    if(*l1Ptr == 0){
        *l1Ptr = (uint8_t**)mmap(0, LEVEL_2_PAGE_TABLE_SIZE, PROT_WRITE | PROT_READ, MAP_NORESERVE | MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
        shadowPage = (*l1Ptr)[LEVEL_2_PAGE_TABLE_SLOT(addr)] = (uint8_t*) mmap(0, PAGE_SIZE * (sizeof(uint64_t)), PROT_WRITE | PROT_READ, MAP_NORESERVE | MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    }else if((shadowPage = (*l1Ptr)[LEVEL_2_PAGE_TABLE_SLOT(addr)]) == 0 ){
        shadowPage = (*l1Ptr)[LEVEL_2_PAGE_TABLE_SLOT(addr)] = (uint8_t*) mmap(0, PAGE_SIZE * (sizeof(uint64_t)), PROT_WRITE | PROT_READ, MAP_NORESERVE | MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    }
    return shadowPage;
}

template<class T, uint16_t AccessLen, uint32_t bufferOffset, bool isApprox>
struct RedSpyAnalysis{
    static void RecordNByteValueBeforeWrite(void *addr, void* drcontext, uint32_t memOp){
        //record the value in the memory before the memory write
        per_thread_t *pt = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);
        pt->bytesWritten += AccessLen;
        // create the shadow address
        tuple<uint8_t[SHADOW_PAGE_SIZE_ML], uint64_t[SHADOW_PAGE_SIZE_ML], context_handle_t[SHADOW_PAGE_SIZE_ML]> &t = sm.GetOrCreateShadowBaseAddress((uint64_t)addr);
        uint8_t * __restrict__ shadowAddr = &(get<0>(t)[PAGE_OFFSET_ML((uint64_t)addr)]); // the shadow addres of a data object
        context_handle_t * __restrict__ pre_ctxt_hndl = &(get<2>(t)[PAGE_OFFSET_ML((uint64_t)addr)]);
        //read the value by using the addr and memory operation size and the store it into shadow memory
        if (isApprox) {
            if(AccessLen == ZMM_VEC_LEN || AccessLen == YMM_VEC_LEN || AccessLen == XMM_VEC_LEN){
                T * __restrict__ oldValue = reinterpret_cast<T*> (shadowAddr);
                T * __restrict__ newValue = reinterpret_cast<T*> (addr);
                
                for(uint32_t i = 0; i < AccessLen/ sizeof(T); i++) {
                    T tmp_new;
                    if(dr_safe_read(newValue + i, sizeof(T), &tmp_new, NULL)){                    
                        oldValue[i] = tmp_new;
                    }else{
                        unreadCount += 1;
                        unreadBytes += sizeof(T);
                    }
                }
            }else if(AccessLen == 10){
                memcpy(shadowAddr, addr, AccessLen);
            }else{
                assert (AccessLen < 10);
                
                *((T*)(shadowAddr)) = *(static_cast<T*>(addr));
                
            } 
        }else{
            T cur_value;
            if(!dr_safe_read(addr, AccessLen, &cur_value, NULL)){
                return;
            }
            *((T*)(shadowAddr)) = cur_value;
        }
        //*** For LoadSpy *** : 
        //1. read the value by using the addr and memory operation size
        //2. if this is the first access of this memory address, create the shawdow memory for it 
        //   store it into shadow memory
        //3. if this is't the first access of this memory address, reaf the previous value frome 
        //   shawdow memory. If the current value is same or approx same with the previous value 
        //   add the redundact pair into redendant table. If the current value is different
        //   with the previous value, just update the value in the shawdow memory.
    }
    
    static void CheckNByteValueAfterWrite(void* opAddr, void* drcontext, context_handle_t cur_ctxt_hndl, uint32_t memOp){
        // if(!Sample_flag){
        //     return;
        // }
        
        //compare the value after memory write with the value in the shadow memory
        bool isRedundantWrite = false;
        per_thread_t *pt = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);
        
        //get the shadow address
        tuple<uint8_t[SHADOW_PAGE_SIZE_ML], uint64_t[SHADOW_PAGE_SIZE_ML], context_handle_t[SHADOW_PAGE_SIZE_ML]> &t = sm.GetOrCreateShadowBaseAddress((uint64_t)opAddr);
        uint8_t * __restrict__ shadowAddr = &(get<0>(t)[PAGE_OFFSET_ML((uint64_t)opAddr)]); // the shadow addres of a data object
        context_handle_t * __restrict__ pre_ctxt_hndl = &(get<2>(t)[PAGE_OFFSET_ML((uint64_t)opAddr)]);
        // do the comparison and return a bool value isRedundantWrite
        if(isApprox) {
            // return false;
            if(AccessLen == ZMM_VEC_LEN || AccessLen == YMM_VEC_LEN || AccessLen == XMM_VEC_LEN){
                T * __restrict__ oldValue = reinterpret_cast<T*> (shadowAddr);
                T * __restrict__ newValue = reinterpret_cast<T*> (opAddr);
                
                int redCount = 0;
                
                for(uint32_t i = 0; i < AccessLen/ sizeof(T); i++) {

                    T tmp_new;
                    if(dr_safe_read(newValue + i, sizeof(T), &tmp_new, NULL)){
                    
                        T tmp = (tmp_new - oldValue[i])/oldValue[i];
                        if (tmp < ((T) 0))
                            tmp = -tmp;
                        redCount += tmp <= ((T)delta);
                    }else{
                        unreadCount += 1;
                        unreadBytes += sizeof(T);
                    }
                }
                if(redCount != AccessLen/ sizeof(T)) {
                    isRedundantWrite = false;
                }
            }else if(AccessLen == 10){
                uint8_t newValue[10];
                memcpy(newValue, opAddr, AccessLen);
                
                uint64_t * upperOld = (uint64_t*)&(shadowAddr[2]);
                uint64_t * upperNew = (uint64_t*)&(newValue[2]);
                
                uint16_t * lowOld = (uint16_t*)&(shadowAddr[0]);
                uint16_t * lowNew = (uint16_t*)&(newValue[0]);
                
                if((*lowOld & 0xfff0) == (*lowNew & 0xfff0) && *upperNew == *upperOld){
                    isRedundantWrite = true;
                }
            }else{
                assert (AccessLen < 10);
                T newValue = *(static_cast<T*>(opAddr));
                T oldValue = *((T*)(shadowAddr));
                
                T rate = (newValue - oldValue)/oldValue;
                if( rate <= delta && rate >= -delta ) 
                {
                    isRedundantWrite = true;
                }
            }
        }else{
            T cur_value, prev_value;
            if(!dr_safe_read(opAddr, AccessLen, &cur_value, NULL)){
                return;
            }
            prev_value = *((T*)(shadowAddr));
            isRedundantWrite = (cur_value == prev_value);
        }

        uint8_t *status = GetOrCreateShadowBaseAddress((uint64_t)opAddr);
        int threadID = drcctlib_get_thread_id();
        context_handle_t *prevIP = (context_handle_t*)(status + PAGE_OFFSET((uint64_t)opAddr) * sizeof(context_handle_t));
        bool isAccessWithinPageBoundary = IS_ACCESS_WITHIN_PAGE_BOUNDARY((uint64_t)opAddr, AccessLen);
        // if is redundant write then add to redundent table
        if (isRedundantWrite) {
            // redundancy detected
            if(isAccessWithinPageBoundary){
                // All from the same context ?
                if (UnrolledConjunction<0, AccessLen, 1>::Body( [&] (int index) -> bool {return (prevIP[index] == prevIP[0]); } )) {
                    // repory to RedTable
                    if(isApprox){
                        AddToApproxRedTable(MAKE_CONTEXT_PAIR(prevIP[0], cur_ctxt_hndl), AccessLen, threadID);
                    }else{
                        AddToRedTable(MAKE_CONTEXT_PAIR(prevIP[0], cur_ctxt_hndl), AccessLen, threadID);
                    }

                    // update context
                    UnrolledLoop<0, AccessLen, 1>::Body( [&] (int index) -> void {
                        prevIP[index] = cur_ctxt_hndl;
                    });
                } else {
                    // different contexts
                    UnrolledLoop<0, AccessLen, 1>::Body( [&] (int index) -> void {
                        // report in RedTable
                        if(isApprox){
                            AddToApproxRedTable(MAKE_CONTEXT_PAIR(prevIP[index], cur_ctxt_hndl), 1, threadID);
                        }else{
                            AddToRedTable(MAKE_CONTEXT_PAIR(prevIP[index], cur_ctxt_hndl), 1, threadID);
                        }
                        // update context
                        prevIP[index] = cur_ctxt_hndl;
                    });
                }
            } else {
                // write across a 64k page boundary
                // first byte is on this page though
                if(isApprox){
                    AddToApproxRedTable(MAKE_CONTEXT_PAIR(prevIP[0], cur_ctxt_hndl), 1, threadID);
                }else{
                    AddToRedTable(MAKE_CONTEXT_PAIR(prevIP[0], cur_ctxt_hndl), 1, threadID);
                }
                // update context
                prevIP[0] = cur_ctxt_hndl;
                // remaining bytes (from 1 to AccessLen) across a 64k page boundary
                UnrolledLoop<1, AccessLen, 1>::Body( [&] (int index) -> void {
                    status = GetOrCreateShadowBaseAddress((uint64_t)opAddr + index);
                    prevIP = (context_handle_t*)(status + PAGE_OFFSET(((uint64_t)opAddr + index)) * sizeof(context_handle_t));
                    // report in RedTable
                    if(isApprox){
                        AddToApproxRedTable(MAKE_CONTEXT_PAIR(prevIP[0], cur_ctxt_hndl), 1, threadID);
                    }else{
                        AddToRedTable(MAKE_CONTEXT_PAIR(prevIP[0], cur_ctxt_hndl), 1, threadID);
                    }
                    // update context
                    prevIP[0] = cur_ctxt_hndl;
                });
            }
        } else {
            // no redundancy, just update context
            if (isAccessWithinPageBoundary) {
                UnrolledLoop<0, AccessLen, 1>::Body( [&] (int index) -> void {
                    // all from the same context
                    // update context
                    prevIP[index] = cur_ctxt_hndl;
                });
            } else {
                // write across a 64k page boundary
                UnrolledLoop<0, AccessLen, 1>::Body( [&] (int index) -> void {
                    status = GetOrCreateShadowBaseAddress((uint64_t)opAddr + index);
                    prevIP = (context_handle_t*)(status + PAGE_OFFSET(((uint64_t)opAddr + index)) * sizeof(context_handle_t));
                    // update context
                    prevIP[0] = cur_ctxt_hndl;
                });
            }
        }
    }
};

static void RecordValueBeforeLargeWrite(void *addr, uint16_t AccessLen, void* drcontext, uint32_t memOp){
    //special case for large memroy write
    
    per_thread_t *pt = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);
    pt->bytesWritten += AccessLen;

    tuple<uint8_t[SHADOW_PAGE_SIZE_ML], uint64_t[SHADOW_PAGE_SIZE_ML], context_handle_t[SHADOW_PAGE_SIZE_ML]> &t = sm.GetOrCreateShadowBaseAddress((uint64_t)addr);
    uint8_t * __restrict__ shadowAddr = &(get<0>(t)[PAGE_OFFSET_ML((uint64_t)addr)]);
    memcpy(shadowAddr, addr, AccessLen);

}

static void CheckAfterLargeWrite(void* opAddr, uint16_t AccessLen, void* drcontext, context_handle_t cur_ctxt_hndl, uint32_t memOp){
    //special case for large memroy write
    bool isRedundantWrite = false;
    per_thread_t *pt = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);

    tuple<uint8_t[SHADOW_PAGE_SIZE_ML], uint64_t[SHADOW_PAGE_SIZE_ML], context_handle_t[SHADOW_PAGE_SIZE_ML]> &t = sm.GetOrCreateShadowBaseAddress((uint64_t)opAddr);
    uint8_t * __restrict__ shadowAddr = &(get<0>(t)[PAGE_OFFSET_ML((uint64_t)opAddr)]);
    context_handle_t * __restrict__ pre_ctxt_hndl = &(get<2>(t)[PAGE_OFFSET_ML((uint64_t)opAddr)]);
    
    uint8_t *status = GetOrCreateShadowBaseAddress((uint64_t)opAddr);
    int threadID = drcctlib_get_thread_id();
    context_handle_t *prevIP = (context_handle_t*)(status + PAGE_OFFSET((uint64_t)opAddr) * sizeof(context_handle_t));
    
    if(memcmp(shadowAddr, opAddr, AccessLen) == 0) {
        context_handle_t prevCtx = prevIP[0];
        for(uint32_t index = 0; index < AccessLen; index++) {
            if(prevCtx != prevIP[index]) { 
                prevCtx = prevIP[index]; 
            } 
            AddToRedTable(MAKE_CONTEXT_PAIR(prevIP[0], cur_ctxt_hndl), 1 , threadID);
            prevIP[index] = cur_ctxt_hndl;
        }
    } else {
        for(uint32_t index = 0; index < AccessLen; index++) {
            prevIP[index] = cur_ctxt_hndl;
        }
    }          
}


template<uint32_t readBufferSlotIndex>
struct RedSpyInstrument{
    static void InstrumentValueBeforeWriting(void *drcontext, context_handle_t cur_ctxt_hndl, mem_ref_t *ref, uint32_t memOp, int32_t float_instr_and_ok_to_appx, int32_t is_float){
        //different cases for different data type and size
        void *addr = ref->addr;
        uint32_t refSize = ref->size;
        if(float_instr_and_ok_to_appx == 1) {
            switch(refSize) {
                case 1:
                case 2: 
                    assert(0 && "memory read floating data with unexptected small size");
                case 4:
                    RedSpyAnalysis<float, 4, readBufferSlotIndex, true>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                    break;
                case 8:
                    RedSpyAnalysis<double, 8, readBufferSlotIndex, true>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                    break;
                case 10: 
                    RedSpyAnalysis<uint8_t, 10, readBufferSlotIndex, true>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                    break;
                case 16: {
                    if(is_float == 1) { 
                        RedSpyAnalysis<float, 16, readBufferSlotIndex, true>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                        break;
                    }
                    else{
                        RedSpyAnalysis<double, 16, readBufferSlotIndex, true>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                        break;
                    }
                } break;
                case 32: {
                    if(is_float == 1) { 
                        RedSpyAnalysis<float, 32, readBufferSlotIndex, true>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                        break;
                    }
                    else{
                        RedSpyAnalysis<double, 32, readBufferSlotIndex, true>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                        break;
                    }
                } break;
                default: 
                    break;
            }
        }else{
            switch(refSize) {
                case 1:
                    RedSpyAnalysis<uint8_t, 1, readBufferSlotIndex, false>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                    break;
                case 2:
                    RedSpyAnalysis<uint16_t, 2, readBufferSlotIndex, false>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                    break;
                case 4:
                    RedSpyAnalysis<uint32_t, 4, readBufferSlotIndex, false>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                    break;
                case 8:
                    RedSpyAnalysis<uint64_t, 8, readBufferSlotIndex, false>::RecordNByteValueBeforeWrite(addr, drcontext, memOp);
                    break;
                default:
                    RecordValueBeforeLargeWrite(addr, refSize, drcontext, memOp);
                    break;
            }
        }

    }
    static void InstrumentValueAfterWriting(void *drcontext, context_handle_t cur_ctxt_hndl, op_ref *opList, uint32_t memOp, int32_t float_instr_and_ok_to_appx, int32_t is_float){
        
        //different cases for different data type and size
        void *opAddr = opList->opAddr;
        uint32_t opSize = opList->opSize;
        if(float_instr_and_ok_to_appx == 1) {
            switch(opSize) {
                case 1:
                case 2: 
                    assert(0 && "memory read floating data with unexptected small size");
                case 4:
                    RedSpyAnalysis<float, 4, readBufferSlotIndex, true>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                    break;
                case 8:
                    RedSpyAnalysis<double, 8, readBufferSlotIndex, true>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                    break;
                case 10: 
                    RedSpyAnalysis<uint8_t, 10, readBufferSlotIndex, true>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                    break;
                case 16: {
                    if(is_float == 1) { 
                        RedSpyAnalysis<float, 16, readBufferSlotIndex, true>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                        break;
                    }
                    else{
                        RedSpyAnalysis<double, 16, readBufferSlotIndex, true>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                        break;
                    }
                } break;
                case 32: {
                    if(is_float == 1) { 
                        RedSpyAnalysis<float, 32, readBufferSlotIndex, true>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                        break;
                    }
                    else{
                        RedSpyAnalysis<double, 32, readBufferSlotIndex, true>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                        break;
                    }
                } break;
                default: 
                    break;
            }
        }else{
            switch(opSize) {
                case 1:
                    RedSpyAnalysis<uint8_t, 1, readBufferSlotIndex, false>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                    break;
                case 2:
                    RedSpyAnalysis<uint16_t, 2, readBufferSlotIndex, false>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                    break;
                case 4:
                    RedSpyAnalysis<uint32_t, 4, readBufferSlotIndex, false>::CheckNByteValueAfterWrite(opAddr, drcontext, cur_ctxt_hndl, memOp);
                    break;
                case 8:
                    RedSpyAnalysis<uint64_t, 8, readBufferSlotIndex, false>::CheckNByteValueAfterWrite(opAddr, drcontext,  cur_ctxt_hndl, memOp);
                    break;
                default:
                    CheckAfterLargeWrite(opAddr, opSize, drcontext, cur_ctxt_hndl, memOp);
                    break;
            }
        }
    }
};


// client want to do
void
BeforeWrite(void *drcontext, context_handle_t cur_ctxt_hndl, mem_ref_t *ref, int32_t num, int32_t num_write, int32_t float_instr_and_ok_to_appx, int32_t is_float)
{
    int readBufferSlotIndex = 0;
    for(int32_t memOp = 0; memOp < num_write; memOp++){
        switch(readBufferSlotIndex){
            case 0:
                // Read the value at location before this instruction
                RedSpyInstrument<0>::InstrumentValueBeforeWriting(drcontext, cur_ctxt_hndl, ref, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            case 1:
                RedSpyInstrument<1>::InstrumentValueBeforeWriting(drcontext, cur_ctxt_hndl, ref, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            case 2:
                RedSpyInstrument<2>::InstrumentValueBeforeWriting(drcontext, cur_ctxt_hndl, ref, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            case 3:
                RedSpyInstrument<3>::InstrumentValueBeforeWriting(drcontext, cur_ctxt_hndl, ref, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            case 4:
                RedSpyInstrument<4>::InstrumentValueBeforeWriting(drcontext, cur_ctxt_hndl, ref, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            default:
                //assert(0 && "NYI");
                break;
        }
        // use next slot for the next write operand
        readBufferSlotIndex++;   
    }
}

void
AfterWrite(void *drcontext, context_handle_t cur_ctxt_hndl, op_ref *opList, int32_t num, int32_t num_write, int32_t float_instr_and_ok_to_appx, int32_t is_float){
    
    int readBufferSlotIndex = 0;
    for(int32_t memOp = 0; memOp < num_write; memOp++){
        // read the value at this location after write
        switch(readBufferSlotIndex){
            case 0:
                RedSpyInstrument<0>::InstrumentValueAfterWriting(drcontext, cur_ctxt_hndl, opList, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            case 1:
                RedSpyInstrument<1>::InstrumentValueAfterWriting(drcontext, cur_ctxt_hndl, opList, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            case 2:
                RedSpyInstrument<2>::InstrumentValueAfterWriting(drcontext, cur_ctxt_hndl, opList, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            case 3:
                RedSpyInstrument<3>::InstrumentValueAfterWriting(drcontext, cur_ctxt_hndl, opList, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            case 4:
                RedSpyInstrument<4>::InstrumentValueAfterWriting(drcontext, cur_ctxt_hndl, opList, memOp, float_instr_and_ok_to_appx, is_float);
                break;
            default:
                break;
        }
        // use next slot for the next write op
        readBufferSlotIndex++;
    }
}

// dr clean call
void
InsertCleancall(int32_t slot, int32_t num, int32_t num_write, int32_t float_instr_and_ok_to_appx, int32_t is_float, int32_t IsCld)
{
    //get the drcontext
    void *drcontext = dr_get_current_drcontext();
    //get pt from drcontext
    per_thread_t *pt = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);
    //get context_handle from drcontext
    context_handle_t cur_ctxt_hndl = drcctlib_get_context_handle(drcontext, slot);
    // context_handle_t prev_ctxt_hndl = drcctlib_get_context_handle(drcontext, (slot - 1));
    
    if (IsCld == 1){
        BUF_PTR(pt->cur_buf, mem_ref_t, INSTRACE_TLS_OFFS_BUF_PTR) = pt->cur_buf_list;
        return;
    }
#ifdef SAMPLE_RUN
    if(!pt->sample_mem){
        BUF_PTR(pt->cur_buf, mem_ref_t, INSTRACE_TLS_OFFS_BUF_PTR) = pt->cur_buf_list;
        return;
    }
#endif
    // do the analysis after the memory writes
    //*** For LoadSpy *** : we don't neet it
    for (int i = 0; i < memOp_num; i++){
        if(pt->opList[i].opAddr != 0) {
            AfterWrite(drcontext, cur_ctxt_hndl, &pt->opList[i], num, num_write, pt->float_instr_and_ok_to_appx, pt->is_float);
            // AfterWrite(drcontext, prev_ctxt_hndl, &pt->opList[i], num, num_write);
        }
    }
    // do the analysis before the memory write
    //*** For LoadSpy *** : we need to compare the value before the memory read
    for (int i = 0; i < num; i++) {
        if (pt->cur_buf_list[i].addr != 0) {
            // store the addr and size of ops in opList[]
            // check the values in addr after running ins
            pt->opList[i].opAddr = (uint64_t*)((&pt->cur_buf_list[i])->addr);
            pt->opList[i].opSize = (uint32_t)((&pt->cur_buf_list[i])->size);
            pt->float_instr_and_ok_to_appx = float_instr_and_ok_to_appx;
            pt->is_float = is_float;
            BeforeWrite(drcontext, cur_ctxt_hndl, &pt->cur_buf_list[i], num, num_write, float_instr_and_ok_to_appx, is_float);
        } else {
            pt->opList[i].opAddr = 0;
            pt->opList[i].opSize = 0;
        }
    }
    memOp_num = num;
    BUF_PTR(pt->cur_buf, mem_ref_t, INSTRACE_TLS_OFFS_BUF_PTR) = pt->cur_buf_list;
}

// insert
static void
InstrumentMem(void *drcontext, instrlist_t *ilist, instr_t *where, opnd_t ref)
{
    /* We need two scratch registers */
    //reserve two register for instrumentation
    reg_id_t reg_mem_ref_ptr, free_reg;
    if (drreg_reserve_register(drcontext, ilist, where, NULL, &reg_mem_ref_ptr) !=
            DRREG_SUCCESS ||
        drreg_reserve_register(drcontext, ilist, where, NULL, &free_reg) !=
            DRREG_SUCCESS) {
        DRCCTLIB_EXIT_PROCESS("InstrumentMem drreg_reserve_register != DRREG_SUCCESS");
    }
    //get the memory address and store it into free_reg
    if (!drutil_insert_get_mem_addr(drcontext, ilist, where, ref, free_reg,
                                    reg_mem_ref_ptr)) {
        MINSERT(ilist, where,
                XINST_CREATE_load_int(drcontext, opnd_create_reg(free_reg),
                                      OPND_CREATE_CCT_INT(0)));
    }
    dr_insert_read_raw_tls(drcontext, ilist, where, tls_seg,
                           tls_offs + INSTRACE_TLS_OFFS_BUF_PTR, reg_mem_ref_ptr);
    // store address in free_reg to mem_ref_t->addr
    MINSERT(ilist, where,
            XINST_CREATE_store(
                drcontext, OPND_CREATE_MEMPTR(reg_mem_ref_ptr, offsetof(mem_ref_t, addr)),
                opnd_create_reg(free_reg)));


    //get the size and store it into free_reg
    MINSERT(ilist, where,
            XINST_CREATE_load_int(drcontext, opnd_create_reg(free_reg),
                                  OPND_CREATE_CCT_INT(drutil_opnd_mem_size_in_bytes(ref, where))));
    // store size in free_reg to mem_ref_t->size
    MINSERT(ilist, where,
            XINST_CREATE_store(drcontext, OPND_CREATE_MEMPTR(reg_mem_ref_ptr, offsetof(mem_ref_t, size)),
                             opnd_create_reg(free_reg)));

#ifdef ARM_CCTLIB
    MINSERT(ilist, where,
            XINST_CREATE_load_int(drcontext, opnd_create_reg(free_reg),
                                  OPND_CREATE_CCT_INT(sizeof(mem_ref_t))));
    MINSERT(ilist, where,
            XINST_CREATE_add(drcontext, opnd_create_reg(reg_mem_ref_ptr),
                             opnd_create_reg(free_reg)));
#else
    MINSERT(ilist, where,
            XINST_CREATE_add(drcontext, opnd_create_reg(reg_mem_ref_ptr),
                             OPND_CREATE_CCT_INT(sizeof(mem_ref_t))));
#endif
    dr_insert_write_raw_tls(drcontext, ilist, where, tls_seg,
                            tls_offs + INSTRACE_TLS_OFFS_BUF_PTR, reg_mem_ref_ptr);
    /* Restore scratch registers */
    if (drreg_unreserve_register(drcontext, ilist, where, reg_mem_ref_ptr) !=
            DRREG_SUCCESS ||
        drreg_unreserve_register(drcontext, ilist, where, free_reg) != DRREG_SUCCESS) {
        DRCCTLIB_EXIT_PROCESS("InstrumentMem drreg_unreserve_register != DRREG_SUCCESS");
    }
}

bool IsDoubleOrFloat (int opcode){
	return (opcode == OP_vblendmps || opcode == OP_vbroadcastf32x4 || opcode == OP_vbroadcastf32x8 || 
    opcode == OP_vcompressps || opcode == OP_vcvtps2qq || opcode == OP_vcvtps2udq || 
    opcode == OP_vcvtps2uqq || opcode == OP_vcvtqq2ps || opcode == OP_vcvtss2usi || 
    opcode == OP_vcvttps2qq || opcode == OP_vcvttps2udq || opcode == OP_vcvttps2uqq || 
    opcode == OP_vcvttss2usi || opcode == OP_vcvtudq2ps || opcode == OP_vcvtuqq2ps || 
    opcode == OP_vcvtusi2ss || opcode == OP_vexp2ps || opcode == OP_vexpandps || 
    opcode == OP_vextractf32x4 || opcode == OP_vextractf32x8 || opcode == OP_vfixupimmps || 
    opcode == OP_vfixupimmss || opcode == OP_vfpclassps || opcode == OP_vfpclassss || 
    opcode == OP_vgatherpf0dps || opcode == OP_vgatherpf0qps || opcode == OP_vgatherpf1dps || 
    opcode == OP_vgatherpf1qps || opcode == OP_vgetexpps || opcode == OP_vgetexpss || 
    opcode == OP_vgetmantps || opcode == OP_vgetmantss || opcode == OP_vinsertf32x4 || 
    opcode == OP_vinsertf32x8 || opcode == OP_vpermi2ps || opcode == OP_vpermt2ps || 
    opcode == OP_vrangeps || opcode == OP_vrangess || opcode == OP_vrcp14ps || 
    opcode == OP_vrcp14ss || opcode == OP_vrcp28ps || opcode == OP_vrcp28ss || 
    opcode == OP_vreduceps || opcode == OP_vreducess || opcode == OP_vrndscaleps || 
    opcode == OP_vrndscaless || opcode == OP_vrsqrt14ps || opcode == OP_vrsqrt14ss || 
    opcode == OP_vrsqrt28ps || opcode == OP_vrsqrt28ss || opcode == OP_vscalefps || 
    opcode == OP_vscalefss || opcode == OP_vscatterdps || opcode == OP_vscatterqps || 
    opcode == OP_vscatterpf0dps || opcode == OP_vscatterpf0qps || opcode == OP_vscatterpf1dps || 
    opcode == OP_vscatterpf1qps || opcode == OP_vshuff32x4 || opcode == OP_vbroadcastf32x2);	
}

static inline bool IsOkToApproximate(instr_t* instr) {
    int op = instr_get_opcode(instr);
    switch(op) {
        case OP_fldenv: //XED_ICLASS_FLDENV:
        case OP_fnstenv: //XED_ICLASS_FNSTENV:
        case OP_fnsave: //XED_ICLASS_FNSAVE:
        case OP_fldcw: //XED_ICLASS_FLDCW:
        case OP_fnstcw: //XED_ICLASS_FNSTCW:
        case OP_fxrstor32: //XED_ICLASS_FXRSTOR:
        case OP_fxrstor64: //XED_ICLASS_FXRSTOR64:
        case OP_fxsave32: //XED_ICLASS_FXSAVE:
        case OP_fxsave64: //XED_ICLASS_FXSAVE64:
            return false;
        default:
            return true;
    }
}

static inline bool IsFloatInstructionAndOkToApproximate(instr_t* instr) {
    if(instr_is_floating(instr)){
        int op = instr_get_opcode(instr);
        switch(op) {
            /* added in Intel Sandy Bridge */
            case OP_xgetbv:
            case OP_xsetbv:
            case OP_xsave32: 
            case OP_xrstor32: 
            case OP_xsaveopt32:

            case OP_fxsave64:
            case OP_fxrstor64: 
            case OP_xsave64: 
            case OP_xrstor64:
            case OP_xsaveopt64:
                return false;
            default:
                return IsOkToApproximate(instr);
        }
    }
    else{
        return false;
    }

}

// analysis
void
InstrumentInsCallback(void *drcontext, instr_instrument_msg_t *instrument_msg)
{
    //basic block
    instrlist_t *bb = instrument_msg->bb;
    //instrcution information
    instr_t *instr = instrument_msg->instr;
    //insrtuction index information
    int32_t slot = instrument_msg->slot;

    int num = 0;
    int num_write = 0;
    // when an insturction's destination (where it will write to) is memory reference
    // it is a memoru write instuction, then we need to Instrument it
    for (int i = 0; i < instr_num_dsts(instr); i++) {
        if (opnd_is_memory_reference(instr_get_dst(instr, i))) { //dst = write
            num++;
            num_write++;
            //instrument the memory
            InstrumentMem(drcontext, bb, instr, instr_get_dst(instr, i));
        }
    }
    bool FloatInstrAndOkToAppx;
    bool IsFloat;
    int IsCld = 0;

    //Filter out some instructions that cannot be recognized
    int op = instr_get_opcode(instr);
    if (op == OP_cld){
        IsCld = 1;
    }
    // check the date type of the instruction (int or float) and (float or double) 
    IsFloat = IsDoubleOrFloat(op);
    FloatInstrAndOkToAppx = IsFloatInstructionAndOkToApproximate(instr);
    int float_instr_and_ok_to_appx; 
    int is_float;

    if (IsFloat){
        is_float = 1;
    }else{
        is_float = 0;
    }

    if (FloatInstrAndOkToAppx){
        float_instr_and_ok_to_appx = 1;
    }else{
        float_instr_and_ok_to_appx = 0;
    }
    //insert the analusis function and its arguments
    dr_insert_clean_call(drcontext, bb, instr, (void *)InsertCleancall, false, 6,
                         OPND_CREATE_CCT_INT(slot), OPND_CREATE_CCT_INT(num), OPND_CREATE_CCT_INT(num_write), 
                         OPND_CREATE_CCT_INT(float_instr_and_ok_to_appx), OPND_CREATE_CCT_INT(is_float), OPND_CREATE_CCT_INT(IsCld));
}


static bool RedundancyCompare(const struct RedanduncyData &first, const struct RedanduncyData &second) {
    return first.frequency > second.frequency ? true : false;
}

void PrintRedundancyPairs(void * drcontext,int threadID) {
    vector<RedanduncyData> tmpList;
    vector<RedanduncyData>::iterator tmpIt;
    uint64_t grandTotalRedundantBytes = 0;
    dr_fprintf(gTraceFile, "********** Dump Data from Thread %d **********\n", threadID);//thread 0 appears two times and look at for loop
    for (unordered_map<uint64_t, uint64_t>::iterator it = RedMap[threadID].begin(); it != RedMap[threadID].end(); ++it) {

        RedanduncyData tmp = { DECODE_DEAD ((*it).first), DECODE_KILL((*it).first), (*it).second};
        tmpList.push_back(tmp);
        grandTotalRedundantBytes += tmp.frequency;
    }
    per_thread_t *pt = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);
    dr_fprintf(gTraceFile, "\nTotal write bytes = %lld\n", (pt->bytesWritten));
    dr_fprintf(gTraceFile, "\nTotal redundant bytes = %lld\n", grandTotalRedundantBytes);
    dr_fprintf(gTraceFile, "\nTotal redundant bytes ratio= %f %%\n", grandTotalRedundantBytes * 100.0 / (pt->bytesWritten));

    sort(tmpList.begin(), tmpList.end(), RedundancyCompare);
    vector<RedanduncyData>::iterator listIt;
    int cntxNum = 0;
    for (listIt = tmpList.begin(); listIt != tmpList.end(); listIt++) {
        if (cntxNum < MAX_CONTEXTS) {
            dr_fprintf(gTraceFile, "\n========== (%f) %% ==========\n", (*listIt).frequency * 100.0 / grandTotalRedundantBytes);
            dr_fprintf(gTraceFile, "Redundancy bytes = %lld\n", (*listIt).frequency);
            if ((*listIt).dead == 0) {
                dr_fprintf(gTraceFile, "\nPrepopulated with by OS\n");
            } else {
                drcctlib_print_backtrace(gTraceFile, (*listIt).dead, true, true, -1);
                dr_fprintf(gTraceFile, "dead context: %lu\n", (*listIt).dead);
            }
            dr_fprintf(gTraceFile, "--------------------Redundantly written by--------------------\n");
            drcctlib_print_backtrace(gTraceFile, (*listIt).kill, true, true, -1);
            dr_fprintf(gTraceFile, "kill context: %lu\n", (*listIt).kill);
        }
        else {
            break;
        }
        cntxNum++;
    }
}

static void PrintApproxRedundancyPairs(void * drcontext, int threadId) {
    vector<RedanduncyData> tmpList;
    vector<RedanduncyData>::iterator tmpIt;
    
    uint64_t grandTotalRedundantBytes = 0;
    dr_fprintf(gTraceFile, "\n*************** Dump Data(delta=%.2f%%) from Thread %d ****************\n", delta*100,threadId);
    
#ifdef MERGING
    for(unordered_map<uint64_t, uint64_t>::iterator it = ApproxRedMap[threadId].begin(); it != ApproxRedMap[threadId].end(); ++it) {
        context_handle_t dead = DECODE_DEAD((*it).first);
        context_handle_t kill = DECODE_KILL((*it).first);
        
        for(tmpIt = tmpList.begin(); tmpIt != tmpList.end(); ++tmpIt) {
            if(dead == 0 || ((*tmpIt).dead) == 0){
                continue;
            }

            bool ct1 = drcctlib_have_same_source_line(dead,(*tmpIt).dead);
            bool ct2 = drcctlib_have_same_source_line(kill,(*tmpIt).kill);
            if(ct1 && ct2){
                (*tmpIt).bytes += (*it).second.bytes;
                grandTotalRedundantBytes += (*it).second.bytes;
                break;
            }
        }
        if(tmpIt == tmpList.end()) {
            RedanduncyData tmp = {dead, kill, (*it).second.bytes}; 
            tmpList.push_back(tmp);
            grandTotalRedundantBytes += tmp.bytes;
        }
    }

#else
    for(unordered_map<uint64_t, uint64_t>::iterator it = ApproxRedMap[threadId].begin(); it != ApproxRedMap[threadId].end(); ++it) {
        RedanduncyData tmp = {DECODE_DEAD ((*it).first), DECODE_KILL((*it).first), (*it).second};
        tmpList.push_back(tmp);
        grandTotalRedundantBytes += tmp.frequency;
    }
#endif
    
    per_thread_t *pt = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);
    dr_fprintf(gTraceFile, "\nTotal write bytes = %lld\n", (pt->bytesWritten));
    dr_fprintf(gTraceFile, "\nTotal redundant bytes = %lld\n", grandTotalRedundantBytes);
    dr_fprintf(gTraceFile, "\nTotal redundant bytes ratio= %f %%\n", grandTotalRedundantBytes * 100.0 / (pt->bytesWritten));

    sort(tmpList.begin(), tmpList.end(), RedundancyCompare);
    vector<RedanduncyData>::iterator listIt;
    int cntxNum = 0;
    for (listIt = tmpList.begin(); listIt != tmpList.end(); listIt++) {
        if (cntxNum < MAX_CONTEXTS) {
            dr_fprintf(gTraceFile, "\n========== (%f) %% ==========\n", (*listIt).frequency * 100.0 / grandTotalRedundantBytes);
            dr_fprintf(gTraceFile, "Appr Redundancy bytes = %lld\n", (*listIt).frequency);
            if ((*listIt).dead == 0) {
                dr_fprintf(gTraceFile, "\nPrepopulated with by OS\n");
            } else {
                drcctlib_print_backtrace(gTraceFile, (*listIt).dead, true, true, -1);
                dr_fprintf(gTraceFile, "dead context: %lu\n", (*listIt).dead);
            }
            dr_fprintf(gTraceFile, "--------------------Appr Redundantly written by--------------------\n");
            drcctlib_print_backtrace(gTraceFile, (*listIt).kill, true, true, -1);
            dr_fprintf(gTraceFile, "kill context: %lu\n", (*listIt).kill);
        }
        else {
            break;
        }
        cntxNum++;
    }
}

// #ifdef SAMPLE_RUN
// void
// InstrumentBBStartInsertCallback(void *drcontext, int32_t slot_num, int32_t mem_ref_num)
// {
//     per_thread_t *pt = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);
//     uint32_t tot_num = pt->instr_num;
//     bool flag = pt->sample_mem;


//     if(flag){
//         tot_num += mem_ref_num;
//         if(tot_num > WINDOW_ENABLE){
//             flag = false;
//             tot_num = 0;
//         }
//     }else{
//         tot_num += mem_ref_num;
//         if(tot_num > WINDOW_DISABLE){
//             flag = true;
//             tot_num = 0;
//         }
//     }

//     pt->sample_mem = flag;
//     pt->instr_num = tot_num;

// }
// #endif

static void
ClientThreadStart(void *drcontext)
{
    //malloc space for pt(per thread structure)
    per_thread_t *pt = (per_thread_t *)dr_thread_alloc(drcontext, sizeof(per_thread_t));
    if (pt == NULL) {
        DRCCTLIB_EXIT_PROCESS("pt == NULL");
    }
    //set the place for pt in drcontext
    drmgr_set_tls_field(drcontext, tls_idx, (void *)pt);
    //initilized the values in pt
    pt->cur_buf = dr_get_dr_segment_base(tls_seg);
    pt->cur_buf_list =
        (mem_ref_t *)dr_global_alloc(TLS_MEM_REF_BUFF_SIZE * sizeof(mem_ref_t));
    BUF_PTR(pt->cur_buf, mem_ref_t, INSTRACE_TLS_OFFS_BUF_PTR) = pt->cur_buf_list;
    pt->bytesWritten= 0;
    pt->sample_mem = true;
    pt->instr_num = 0;
}

static void
ClientThreadEnd(void *drcontext)
{
    // get pt
    per_thread_t *pt = (per_thread_t *)drmgr_get_tls_field(drcontext, tls_idx);
    //free the memory spave in pt
    dr_global_free(pt->cur_buf_list, TLS_MEM_REF_BUFF_SIZE * sizeof(mem_ref_t));
    dr_thread_free(drcontext, pt, sizeof(per_thread_t));

    int threadID = drcctlib_get_thread_id();
    
    // need lock for drcctlib_have_same_source_line
    dr_mutex_lock(lock);
    //print out the redundancy information
    PrintRedundancyPairs(drcontext, threadID);
    PrintApproxRedundancyPairs(drcontext, threadID);
    dr_mutex_unlock(lock);
    // clear the map
    RedMap[threadID].clear();
    ApproxRedMap[threadID].clear();
}

static void
ClientInit(int argc, const char *argv[])
{
    //Create a log file name 
    char name[MAXIMUM_PATH] = "";
    DRCCTLIB_INIT_LOG_FILE_NAME(name, "red", "out");
    //set the output txt tgile 
    gTraceFile = dr_open_file(name, DR_FILE_WRITE_OVERWRITE | DR_FILE_ALLOW_LARGE);
    DR_ASSERT(gTraceFile != INVALID_FILE);
    //write the head
    dr_fprintf(gTraceFile, "ClientInit\n");
// #ifdef SAMPLE_RUN
//     dr_fprintf(gTraceFile, "SAMPLING ON\n");
// #endif
    lock = dr_mutex_create();
    
}

static void
ClientExit(void)
{
    // add output module here
    dr_fprintf(gTraceFile, "ClientExit\n");
    drcctlib_exit();
    // unregister the modules
    if (!dr_raw_tls_cfree(tls_offs, INSTRACE_TLS_COUNT)) {
        DRCCTLIB_EXIT_PROCESS(
            "ERROR: drcctlib_memory_with_addr_and_refsize_clean_call dr_raw_tls_calloc fail");
    }
    if (!drmgr_unregister_thread_init_event(ClientThreadStart) ||
        !drmgr_unregister_thread_exit_event(ClientThreadEnd) ||
        !drmgr_unregister_tls_field(tls_idx)) {
        DRCCTLIB_PRINTF("ERROR: drcctlib_memory_with_addr_and_refsize_clean_call failed to "
                        "unregister in ClientExit");
    }
    drmgr_exit();
    if (drreg_exit() != DRREG_SUCCESS) {
        DRCCTLIB_PRINTF("failed to exit drreg");
    }
    drutil_exit();
    //destory the lock
    dr_mutex_destroy(lock);
}


bool
CustomFilter(instr_t *instr)
{
    //Instructions filter, only return the instructions that write to memory
    //*** For LoadSpy *** : return the instructions that read from memory
    return (instr_writes_memory(instr));
}


#ifdef __cplusplus
extern "C" {
#endif



DR_EXPORT void
dr_client_main(client_id_t id, int argc, const char *argv[])
{
    dr_set_client_name("DynamoRIO Client 'drcctlib_memory_with_addr_and_refsize_clean_call'",
                       "http://dynamorio.org/issues");
    ClientInit(argc, argv);

    // Initializes the drmgr extension. Must be called prior to any of the other routines
    if (!drmgr_init()) {
        DRCCTLIB_EXIT_PROCESS("ERROR: drcctlib_memory_with_addr_and_refsize_clean_call "
                              "unable to initialize drmgr");
    }

    //Specifies the options when initializing drreg.
    drreg_options_t ops = { sizeof(ops), 4 /*max slots needed*/, false };
    //Initializes the drreg extension. Must be called prior to any of the other routines.
    if (drreg_init(&ops) != DRREG_SUCCESS) {
        DRCCTLIB_EXIT_PROCESS("ERROR: drcctlib_memory_with_addr_and_refsize_clean_call "
                              "unable to initialize drreg");
    }
    //Initializes the Instrumentation Utilities extension.
    if (!drutil_init()) {
        DRCCTLIB_EXIT_PROCESS("ERROR: drcctlib_memory_with_addr_and_refsize_clean_call "
                              "unable to initialize drutil");
    }
    // register the thread init function (what will do when a thread starts)
    drmgr_register_thread_init_event(ClientThreadStart);
    // register the thread end function (what will do when a thread ends)
    drmgr_register_thread_exit_event(ClientThreadEnd);

    //Reserves a thread-local storage (tls) slot for every thread. Returns the index of the slot.(tls_idx) 
    tls_idx = drmgr_register_tls_field();
    if (tls_idx == -1) {
        DRCCTLIB_EXIT_PROCESS("ERROR: drcctlib_memory_with_addr_and_refsize_clean_call "
                              "drmgr_register_tls_field fail");
    }

    //Create a set of (TLS) slots that can be directly accessed via tls_seg and tls_offs
    if (!dr_raw_tls_calloc(&tls_seg, &tls_offs, INSTRACE_TLS_COUNT, 0)) {
        DRCCTLIB_EXIT_PROCESS(
            "ERROR: drcctlib_memory_with_addr_and_refsize_clean_call dr_raw_tls_calloc fail");
    }
// #ifdef SAMPLE_RUN
//     drcctlib_init_ex(CustomFilter, INVALID_FILE,
//                      InstrumentInsCallback, InstrumentBBStartInsertCallback, NULL,
//                      DRCCTLIB_DEFAULT);
// #else
    drcctlib_init(CustomFilter, INVALID_FILE, InstrumentInsCallback, false);
// #endif

    //register the clinet exit event (what will do when client ends)
    dr_register_exit_event(ClientExit);
}

#ifdef __cplusplus
}
#endif