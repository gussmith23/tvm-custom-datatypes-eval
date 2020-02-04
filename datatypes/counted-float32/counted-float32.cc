#include <cstdint>
#include <cmath>

extern "C" size_t num_adds = 0;
extern "C" size_t num_subs = 0;
extern "C" size_t num_muls = 0;
extern "C" size_t num_divs = 0;
extern "C" size_t num_maxes = 0;
extern "C" size_t num_sqrts = 0;
extern "C" size_t num_exps = 0;

extern "C" size_t GetNumAdds() { return num_adds; }
extern "C" size_t GetNumSubs() { return num_subs; }
extern "C" size_t GetNumMuls() { return num_muls; }
extern "C" size_t GetNumDivs() { return num_divs; }
extern "C" size_t GetNumMaxes() { return num_maxes; }
extern "C" size_t GetNumSqrts() { return num_sqrts; }
extern "C" size_t GetNumExps() { return num_exps; }

extern "C" float CountedFloat32ToFloat(uint32_t in) {
  return *reinterpret_cast<float *>(&in);
}

extern "C" uint32_t FloatToCountedFloat32(float in) {
  return *reinterpret_cast<uint32_t *>(&in);
}

// TODO(gus) how wide should the input be?
extern "C" uint32_t IntToCountedFloat32(int in) {
  return (float)in;
}

extern "C" uint32_t CountedFloat32Add(uint32_t a, uint32_t b) {
  num_adds++;
  return FloatToCountedFloat32(CountedFloat32ToFloat(a) + CountedFloat32ToFloat(b));
}

extern "C" uint32_t CountedFloat32Sub(uint32_t a, uint32_t b) {
  num_subs++;
  return FloatToCountedFloat32(CountedFloat32ToFloat(a) - CountedFloat32ToFloat(b));
}

extern "C" uint32_t CountedFloat32Mul(uint32_t a, uint32_t b) {
  num_muls++;
  return FloatToCountedFloat32(CountedFloat32ToFloat(a) * CountedFloat32ToFloat(b));
}

extern "C" uint32_t CountedFloat32Div(uint32_t a, uint32_t b) {
  num_divs++;
  return FloatToCountedFloat32(CountedFloat32ToFloat(a) / CountedFloat32ToFloat(b));
}

extern "C" uint32_t CountedFloat32Max(uint32_t a, uint32_t b) {
  num_maxes++;
  return (CountedFloat32ToFloat(a) > CountedFloat32ToFloat(b)) ? a : b;
}

extern "C" uint32_t CountedFloat32Sqrt(uint32_t a) {
  num_sqrts++;
  return FloatToCountedFloat32(sqrt(CountedFloat32ToFloat(a)));
}

extern "C" uint32_t CountedFloat32Exp(uint32_t a) {
  num_exps++;
  return FloatToCountedFloat32(exp(CountedFloat32ToFloat(a)));
}
