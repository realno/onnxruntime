#include "core/platform/threadpool.h"
#include <memory>
#include <string>

namespace onnxruntime {
namespace concurrency {

std::unique_ptr<ThreadPool> CreateThreadPool(int thread_pool_size, bool allow_spinning,
                                                   ThreadPool::ThreadEnvironment& env, EigenAllocator* allocator);
}  // namespace concurrency
}  // namespace onnxruntime