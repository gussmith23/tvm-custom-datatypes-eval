#include <cstdint>
#include <iostream>
#include <posit.h>

// The extern "C" functions in this file are named with an underscore so as not
// to collide with the similarly-named functions which may or may not also
// exist in TVM.
posit8_t Uint8ToLibPosit_Posit8es0_(uint8_t in) {
  return posit8_reinterpret(in);
}

uint8_t LibPosit_Posit8es0_toUint8(posit8_t in) { return posit8_bits(in); }

extern "C" float _LibPosit_Posit8es0_ToFloat(uint8_t in) {
  return posit8_tof(Uint8ToLibPosit_Posit8es0_(in));
}

extern "C" uint8_t _FloatToLibPosit_Posit8es0(float in) {
  return LibPosit_Posit8es0_toUint8(posit8_fromf(in));
}

// TODO(gus) how wide should the input be?
extern "C" uint8_t _IntToLibPosit_Posit8es0(int in) {
  return LibPosit_Posit8es0_toUint8(posit8_fromsl(in));
}

extern "C" uint8_t _LibPosit_Posit8es0_Add(uint8_t a, uint8_t b) {
  return LibPosit_Posit8es0_toUint8(
      posit8_add(Uint8ToLibPosit_Posit8es0_(a), Uint8ToLibPosit_Posit8es0_(b)));
}

extern "C" uint8_t _LibPosit_Posit8es0_Sub(uint8_t a, uint8_t b) {
  return LibPosit_Posit8es0_toUint8(
      posit8_sub(Uint8ToLibPosit_Posit8es0_(a), Uint8ToLibPosit_Posit8es0_(b)));
}

extern "C" uint8_t _LibPosit_Posit8es0_Mul(uint8_t a, uint8_t b) {
  return LibPosit_Posit8es0_toUint8(
      posit8_mul(Uint8ToLibPosit_Posit8es0_(a), Uint8ToLibPosit_Posit8es0_(b)));
}

extern "C" uint8_t _LibPosit_Posit8es0_Div(uint8_t a, uint8_t b) {
  return LibPosit_Posit8es0_toUint8(
      posit8_div(Uint8ToLibPosit_Posit8es0_(a), Uint8ToLibPosit_Posit8es0_(b)));
}

extern "C" uint8_t _LibPosit_Posit8es0_Max(uint8_t a, uint8_t b) {
  auto a_p = Uint8ToLibPosit_Posit8es0_(a);
  auto b_p = Uint8ToLibPosit_Posit8es0_(b);
  return LibPosit_Posit8es0_toUint8(posit8_cmp(a_p, b_p) == 1 ? a_p : b_p);
}

extern "C" uint8_t _LibPosit_Posit8es0_Sqrt(uint8_t a) {
  return LibPosit_Posit8es0_toUint8(posit8_sqrt(Uint8ToLibPosit_Posit8es0_(a)));
}

extern "C" uint8_t _LibPosit_Posit8es0_Exp(uint8_t a) {
  return LibPosit_Posit8es0_toUint8(posit8_exp(Uint8ToLibPosit_Posit8es0_(a)));
}

posit16_t Uint16ToLibPosit_Posit16es1_(uint16_t in) {
  return posit16_reinterpret(in);
}

uint16_t LibPosit_Posit16es1_toUint16(posit16_t in) { return posit16_bits(in); }

extern "C" float _LibPosit_Posit16es1_ToFloat(uint16_t in) {
  return posit16_tof(Uint16ToLibPosit_Posit16es1_(in));
}

extern "C" uint16_t _FloatToLibPosit_Posit16es1(float in) {
  return LibPosit_Posit16es1_toUint16(posit16_fromf(in));
}

// TODO(gus) how wide should the input be?
extern "C" uint16_t _IntToLibPosit_Posit16es1(int in) {
  return LibPosit_Posit16es1_toUint16(posit16_fromsl(in));
}

extern "C" uint16_t _LibPosit_Posit16es1_Add(uint16_t a, uint16_t b) {
  return LibPosit_Posit16es1_toUint16(posit16_add(
      Uint16ToLibPosit_Posit16es1_(a), Uint16ToLibPosit_Posit16es1_(b)));
}

extern "C" uint16_t _LibPosit_Posit16es1_Sub(uint16_t a, uint16_t b) {
  return LibPosit_Posit16es1_toUint16(posit16_sub(
      Uint16ToLibPosit_Posit16es1_(a), Uint16ToLibPosit_Posit16es1_(b)));
}

extern "C" uint16_t _LibPosit_Posit16es1_Mul(uint16_t a, uint16_t b) {
  return LibPosit_Posit16es1_toUint16(posit16_mul(
      Uint16ToLibPosit_Posit16es1_(a), Uint16ToLibPosit_Posit16es1_(b)));
}

extern "C" uint16_t _LibPosit_Posit16es1_Div(uint16_t a, uint16_t b) {
  return LibPosit_Posit16es1_toUint16(posit16_div(
      Uint16ToLibPosit_Posit16es1_(a), Uint16ToLibPosit_Posit16es1_(b)));
}

extern "C" uint16_t _LibPosit_Posit16es1_Max(uint16_t a, uint16_t b) {
  auto a_p = Uint16ToLibPosit_Posit16es1_(a);
  auto b_p = Uint16ToLibPosit_Posit16es1_(b);
  return LibPosit_Posit16es1_toUint16(posit16_cmp(a_p, b_p) == 1 ? a_p : b_p);
}

extern "C" uint16_t _LibPosit_Posit16es1_Sqrt(uint16_t a) {
  return LibPosit_Posit16es1_toUint16(
      posit16_sqrt(Uint16ToLibPosit_Posit16es1_(a)));
}

extern "C" uint16_t _LibPosit_Posit16es1_Exp(uint16_t a) {
  return LibPosit_Posit16es1_toUint16(
      posit16_exp(Uint16ToLibPosit_Posit16es1_(a)));
}

posit32_t Uint32ToLibPosit_Posit32es2_(uint32_t in) {
  return posit32_reinterpret(in);
}

uint32_t LibPosit_Posit32es2_toUint32(posit32_t in) { return posit32_bits(in); }

extern "C" float _LibPosit_Posit32es2_ToFloat(uint32_t in) {
  return posit32_tof(Uint32ToLibPosit_Posit32es2_(in));
}

extern "C" uint32_t _FloatToLibPosit_Posit32es2(float in) {
  return LibPosit_Posit32es2_toUint32(posit32_fromf(in));
}

// TODO(gus) how wide should the input be?
extern "C" uint32_t _IntToLibPosit_Posit32es2(int in) {
  return LibPosit_Posit32es2_toUint32(posit32_fromsl(in));
}

extern "C" uint32_t _LibPosit_Posit32es2_Add(uint32_t a, uint32_t b) {
  return LibPosit_Posit32es2_toUint32(posit32_add(
      Uint32ToLibPosit_Posit32es2_(a), Uint32ToLibPosit_Posit32es2_(b)));
}

extern "C" uint32_t _LibPosit_Posit32es2_Sub(uint32_t a, uint32_t b) {
  return LibPosit_Posit32es2_toUint32(posit32_sub(
      Uint32ToLibPosit_Posit32es2_(a), Uint32ToLibPosit_Posit32es2_(b)));
}

extern "C" uint32_t _LibPosit_Posit32es2_Mul(uint32_t a, uint32_t b) {
  return LibPosit_Posit32es2_toUint32(posit32_mul(
      Uint32ToLibPosit_Posit32es2_(a), Uint32ToLibPosit_Posit32es2_(b)));
}

extern "C" uint32_t _LibPosit_Posit32es2_Div(uint32_t a, uint32_t b) {
  return LibPosit_Posit32es2_toUint32(posit32_div(
      Uint32ToLibPosit_Posit32es2_(a), Uint32ToLibPosit_Posit32es2_(b)));
}

extern "C" uint32_t _LibPosit_Posit32es2_Max(uint32_t a, uint32_t b) {
  auto a_p = Uint32ToLibPosit_Posit32es2_(a);
  auto b_p = Uint32ToLibPosit_Posit32es2_(b);
  return LibPosit_Posit32es2_toUint32(posit32_cmp(a_p, b_p) == 1 ? a_p : b_p);
  // TODO(gus): why doesn't the below line work? the following line's
  // implementation of max is not correct, and fails in unexpected ways.
  // return LibPosit_Posit32es2_toUint32(posit32_cmp(a_p, b_p) ? a_p : b_p);
  // This was the implementation that helped me figure out something was wrong.
  // return LibPosit_Posit32es2_toUint32(posit32_tof(a_p) > posit32_tof(b_p) ?
  // a_p : b_p);
}

extern "C" uint32_t _LibPosit_Posit32es2_Sqrt(uint32_t a) {
  auto p = Uint32ToLibPosit_Posit32es2_(a);
  // std::cout << posit32_tof(p) << std::endl;
  auto sqrt = posit32_sqrt(p);
  // std::cout << posit32_tof(sqrt) << std::endl;
  return LibPosit_Posit32es2_toUint32(sqrt);
}

extern "C" uint32_t _LibPosit_Posit32es2_Exp(uint32_t a) {
  auto p = Uint32ToLibPosit_Posit32es2_(a);
  // std::cout << posit32_tof(p) << std::endl;
  auto exp = posit32_exp(p);
  // std::cout << posit32_tof(exp) << std::endl;
  auto out = LibPosit_Posit32es2_toUint32(exp);
  return out;
}
