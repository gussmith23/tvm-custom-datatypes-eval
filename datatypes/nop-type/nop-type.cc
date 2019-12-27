#include <cstdint>

extern "C" float Nop32ToFloat(uint32_t in) {
  return 1.0;
}

extern "C" uint32_t FloatToNop32(float in) {
  return 1;
}

// TODO(gus) how wide should the input be?
extern "C" uint32_t IntToNop32(int in) {
  return in;
}

extern "C" uint32_t Nop32Add(uint32_t a, uint32_t b) {
  return a;
}

extern "C" uint32_t Nop32Sub(uint32_t a, uint32_t b) {
  return a;
}

extern "C" uint32_t Nop32Mul(uint32_t a, uint32_t b) {
  return a;
}

extern "C" uint32_t Nop32Div(uint32_t a, uint32_t b) {
  return a;
}

extern "C" uint32_t Nop32Max(uint32_t a, uint32_t b) {
  return a;
}

extern "C" uint32_t Nop32Sqrt(uint32_t a) {
  return a;
}

extern "C" uint32_t Nop32Exp(uint32_t a) {
  return a;
}
