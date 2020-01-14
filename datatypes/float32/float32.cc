#include <cstdint>
#include <cmath>

extern "C" float OurFloat32ToFloat(uint32_t in) {
  return *reinterpret_cast<float *>(&in);
}

extern "C" uint32_t FloatToOurFloat32(float in) {
  return *reinterpret_cast<uint32_t *>(&in);
}

// TODO(gus) how wide should the input be?
extern "C" uint32_t IntToOurFloat32(int in) {
  return (float)in;
}

extern "C" uint32_t OurFloat32Add(uint32_t a, uint32_t b) {
  return FloatToOurFloat32(OurFloat32ToFloat(a) + OurFloat32ToFloat(b));
}

extern "C" uint32_t OurFloat32Sub(uint32_t a, uint32_t b) {
  return FloatToOurFloat32(OurFloat32ToFloat(a) - OurFloat32ToFloat(b));
}

extern "C" uint32_t OurFloat32Mul(uint32_t a, uint32_t b) {
  return FloatToOurFloat32(OurFloat32ToFloat(a) * OurFloat32ToFloat(b));
}

extern "C" uint32_t OurFloat32Div(uint32_t a, uint32_t b) {
  return FloatToOurFloat32(OurFloat32ToFloat(a) / OurFloat32ToFloat(b));
}

extern "C" uint32_t OurFloat32Max(uint32_t a, uint32_t b) {
  return (OurFloat32ToFloat(a) > OurFloat32ToFloat(b)) ? a : b;
}

extern "C" uint32_t OurFloat32Sqrt(uint32_t a) {
  return FloatToOurFloat32(sqrt(OurFloat32ToFloat(a)));
}

extern "C" uint32_t OurFloat32Exp(uint32_t a) {
  return FloatToOurFloat32(exp(OurFloat32ToFloat(a)));
}
