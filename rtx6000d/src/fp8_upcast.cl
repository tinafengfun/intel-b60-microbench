// FP8(e4m3) -> BF16 upcast probe: what instructions does IGC emit on BMG (Xe2)?
// If Xe2 had a native FP8 convert instruction, the conversion below would map
// to it. If not, we should see pure integer/FP16 ALU sequences.
__kernel void upcast_e4m3_bf16(__global uchar *in, __global ushort *out) {
  int i = get_global_id(0);
  uchar v = in[i];
  // e4m3 -> fp16: place bits (S EEEE MMM -> S EEEE MMM 0000000), rebias by *2^(15-7)
  ushort h = ((ushort)v) << 8;
  half f = as_half(h) * 256.0h;
  // fp16 -> fp32 -> bf16 (round-to-nearest-even)
  uint u = as_uint((float)f);
  u += 0x7fffu + ((u >> 16) & 1u);
  out[i] = (ushort)(u >> 16);
}

// e5m2 -> bf16 variant (e5m2 has fp16-compatible exponent, shift-only to fp16)
__kernel void upcast_e5m2_bf16(__global uchar *in, __global ushort *out) {
  int i = get_global_id(0);
  uchar v = in[i];
  ushort h = ((ushort)v) << 8;
  half f = as_half(h);          // e5m2 -> fp16 exact bit placement
  uint u = as_uint((float)f);
  u += 0x7fffu + ((u >> 16) & 1u);
  out[i] = (ushort)(u >> 16);
}

// throughput flavor: loop to keep EU busy, consume result to defeat DCE
__kernel void upcast_loop(__global uchar *in, __global ushort *out, int iters) {
  int i = get_global_id(0);
  uchar v = in[i & 4095];
  uint acc = 0;
  for (int k = 0; k < iters; k++) {
    ushort h = ((ushort)(v ^ (uchar)k)) << 8;
    half f = as_half(h) * 256.0h;
    uint u = as_uint((float)f);
    u += 0x7fffu + ((u >> 16) & 1u);
    acc ^= (u >> 16);
    v ^= 1;
  }
  if (acc == 12345u) out[0] = (ushort)acc;
}
