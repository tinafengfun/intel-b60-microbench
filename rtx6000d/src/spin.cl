__kernel void spin(__global float* out, int iters) {
  float a = get_global_id(0) % 1000;
  for (int i = 0; i < iters; i++) a = fma(a, 0.9999f, 0.0001f);
  if (a == 12345.f) out[0] = a;
}
