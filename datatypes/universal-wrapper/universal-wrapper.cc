// Configure the posit template environment
// TODO(gus) have to disable 32 bit; segfaults on my Docker image when enabled,
// not sure why.
// Note: I can run the fast posit32 on my laptop. When I run it, it doesn't seem
// to be any faster, and it actually produces different results. This is
// worrying, but it makes me feel better about not enabling the fast version.
#define POSIT_FAST_POSIT_32_2 0
#define POSIT_FAST_POSIT_16_1 1
#define POSIT_FAST_POSIT_8_0 1
#define POSIT_THROW_ARITHMETIC_EXCEPTION 0

#include "universal/posit/posit"
#include <cstdint>

// The extern "C" functions in this file are named with an underscore so as not
// to collide with the similarly-named functions which may or may not also
// exist in TVM.

sw::unum::posit<8, 0> Uint8ToPosit8es0(uint8_t in) {
  sw::unum::bitblock<8> bb;
  bb = static_cast<unsigned long long>(in);
  return sw::unum::posit<8, 0>().set(bb);
}

uint8_t Posit8es0toUint8(sw::unum::posit<8, 0> in) {
  return static_cast<uint8_t>(in.get().to_ullong());
}

extern "C" float _Posit8es0ToFloat(uint8_t in) {
  return Uint8ToPosit8es0(in).operator float();
}

extern "C" uint8_t _FloatToPosit8es0(float in) {
  auto posit = sw::unum::posit<8, 0>(in);
  return Posit8es0toUint8(posit);
}

// TODO(gus) how wide should the input be?
extern "C" uint8_t _IntToPosit8es0(int in) {
  return Posit8es0toUint8(sw::unum::posit<8, 0>(in));
}

extern "C" uint8_t _Posit8es0Add(uint8_t a, uint8_t b) {
  return Posit8es0toUint8(Uint8ToPosit8es0(a) + Uint8ToPosit8es0(b));
}

extern "C" uint8_t _Posit8es0Sub(uint8_t a, uint8_t b) {
  return Posit8es0toUint8(Uint8ToPosit8es0(a) - Uint8ToPosit8es0(b));
}

extern "C" uint8_t _Posit8es0Mul(uint8_t a, uint8_t b) {
  return Posit8es0toUint8(Uint8ToPosit8es0(a) * Uint8ToPosit8es0(b));
}

extern "C" uint8_t _Posit8es0Div(uint8_t a, uint8_t b) {
  return Posit8es0toUint8(Uint8ToPosit8es0(a) / Uint8ToPosit8es0(b));
}

extern "C" uint8_t _Posit8es0Max(uint8_t a, uint8_t b) {
  auto a_p = Uint8ToPosit8es0(a);
  auto b_p = Uint8ToPosit8es0(b);
  return Posit8es0toUint8(a_p > b_p ? a_p : b_p);
}

extern "C" uint8_t _Posit8es0Sqrt(uint8_t a) {
  return Posit8es0toUint8(sw::unum::sqrt(Uint8ToPosit8es0(a)));
}

extern "C" uint8_t _Posit8es0Exp(uint8_t a) {
  return Posit8es0toUint8(sw::unum::exp(Uint8ToPosit8es0(a)));
}

sw::unum::posit<16, 1> Uint16ToPosit16es1(uint16_t in) {
  sw::unum::bitblock<16> bb;
  bb = static_cast<unsigned long long>(in);
  return sw::unum::posit<16, 1>().set(bb);
}

uint16_t Posit16es1toUint16(sw::unum::posit<16, 1> in) {
  return static_cast<uint16_t>(in.get().to_ullong());
}

extern "C" float _Posit16es1ToFloat(uint16_t in) {
  return Uint16ToPosit16es1(in).operator float();
}

extern "C" uint16_t _FloatToPosit16es1(float in) {
  auto posit = sw::unum::posit<16, 1>(in);
  return Posit16es1toUint16(posit);
}

// TODO(gus) how wide should the input be?
extern "C" uint16_t _IntToPosit16es1(int in) {
  return Posit16es1toUint16(sw::unum::posit<16, 1>(in));
}

extern "C" uint16_t _Posit16es1Add(uint16_t a, uint16_t b) {
  return Posit16es1toUint16(Uint16ToPosit16es1(a) + Uint16ToPosit16es1(b));
}

extern "C" uint16_t _Posit16es1Sub(uint16_t a, uint16_t b) {
  return Posit16es1toUint16(Uint16ToPosit16es1(a) - Uint16ToPosit16es1(b));
}

extern "C" uint16_t _Posit16es1Mul(uint16_t a, uint16_t b) {
  return Posit16es1toUint16(Uint16ToPosit16es1(a) * Uint16ToPosit16es1(b));
}

extern "C" uint16_t _Posit16es1Div(uint16_t a, uint16_t b) {
  return Posit16es1toUint16(Uint16ToPosit16es1(a) / Uint16ToPosit16es1(b));
}

extern "C" uint16_t _Posit16es1Max(uint16_t a, uint16_t b) {
  auto a_p = Uint16ToPosit16es1(a);
  auto b_p = Uint16ToPosit16es1(b);
  return Posit16es1toUint16(a_p > b_p ? a_p : b_p);
}

extern "C" uint16_t _Posit16es1Sqrt(uint16_t a) {
  return Posit16es1toUint16(sw::unum::sqrt(Uint16ToPosit16es1(a)));
}

extern "C" uint16_t _Posit16es1Exp(uint16_t a) {
  return Posit16es1toUint16(sw::unum::exp(Uint16ToPosit16es1(a)));
}

sw::unum::posit<32, 2> Uint32ToPosit32es2(uint32_t in) {
  sw::unum::bitblock<32> bb;
  bb = static_cast<unsigned long long>(in);
  return sw::unum::posit<32, 2>().set(bb);
}

uint32_t Posit32es2ToUint32(sw::unum::posit<32, 2> in) {
  return static_cast<uint32_t>(in.get().to_ullong());
}

extern "C" float _Posit32es2ToFloat(uint32_t in) {
  return Uint32ToPosit32es2(in).operator float();
}

extern "C" uint32_t _FloatToPosit32es2(float in) {
  auto posit = sw::unum::posit<32, 2>(in);
  return Posit32es2ToUint32(posit);
}

// TODO(gus) how wide should the input be?
extern "C" uint32_t _IntToPosit32es2(int in) {
  return Posit32es2ToUint32(sw::unum::posit<32, 2>(in));
}

extern "C" uint32_t _Posit32es2Add(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) + Uint32ToPosit32es2(b));
}

extern "C" uint32_t _Posit32es2Sub(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) - Uint32ToPosit32es2(b));
}

extern "C" uint32_t _Posit32es2Mul(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) * Uint32ToPosit32es2(b));
}

extern "C" uint32_t _Posit32es2Div(uint32_t a, uint32_t b) {
  return Posit32es2ToUint32(Uint32ToPosit32es2(a) / Uint32ToPosit32es2(b));
}

extern "C" uint32_t _Posit32es2Max(uint32_t a, uint32_t b) {
  auto a_p = Uint32ToPosit32es2(a);
  auto b_p = Uint32ToPosit32es2(b);
  return Posit32es2ToUint32(a_p > b_p ? a_p : b_p);
}

extern "C" uint32_t _Posit32es2Sqrt(uint32_t a) {
  return Posit32es2ToUint32(sw::unum::sqrt(Uint32ToPosit32es2(a)));
}

extern "C" uint32_t _Posit32es2Exp(uint32_t a) {
  return Posit32es2ToUint32(sw::unum::exp(Uint32ToPosit32es2(a)));
}
