#include <sycl/sycl.hpp>
int main() {
  sycl::queue q;
  float *buf = sycl::malloc_shared<float>(1024, q);
  q.submit([&](sycl::handler &h) {
    h.parallel_for(sycl::range<1>(256), [=](sycl::id<1> i) {
      float x = buf[i];
      float t = sycl::tanh(x);              // probe: native or emulated?
      float s = 1.0f / (1.0f + sycl::exp(-x));  // sigmoid
      buf[i] = t + s;
    });
  }).wait();
  return buf[0] == 12345.f ? 1 : 0;
}
