#pragma once
#include <cstddef>
#include <cstdint>

// 每个句柄固定由一个 CPU 线程拥有；返回视图只在下次调用前有效。
extern "C" {
void *hj_actor_create();
int hj_actor_call(void *, uint32_t op, const uint8_t *, size_t, const uint8_t **, size_t *);
void hj_actor_close(void *);
const char *hj_identity();
size_t hj_memory_bytes();
}
