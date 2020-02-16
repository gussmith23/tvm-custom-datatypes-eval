/*
    Copyright (c) 2019 by Contributors
   \file tvm/src/codegen/custom_datatypes/mybfloat16.cc
   \brief Small bfloat16 library for use in unittests

  Code originally from TensorFlow; taken and simplified. Original license:

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
  ==============================================================================*/

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "biovault_bfloat16/biovault_bfloat16.h"

extern "C" {

uint16_t FloatToBiovaultBFloat16(float in) {
  return biovault::bfloat16_t(in).raw_bits();
}

float BiovaultBFloat16ToFloat(uint16_t in) {
  return biovault::bfloat16_t(in).operator float();
}

uint16_t IntToBiovaultBfloat16(int in) {
  return biovault::bfloat16_t((float)in).raw_bits();
}

uint16_t BiovaultBFloat16Add(uint16_t a, uint16_t b) {
  return FloatToBiovaultBFloat16(BiovaultBFloat16ToFloat(a) +
                                 BiovaultBFloat16ToFloat(b));
}

uint16_t BiovaultBFloat16Sub(uint16_t a, uint16_t b) {
  return FloatToBiovaultBFloat16(BiovaultBFloat16ToFloat(a) -
                                 BiovaultBFloat16ToFloat(b));
}

uint16_t BiovaultBFloat16Mul(uint16_t a, uint16_t b) {
  return FloatToBiovaultBFloat16(BiovaultBFloat16ToFloat(a) *
                                 BiovaultBFloat16ToFloat(b));
}

uint16_t BiovaultBFloat16Div(uint16_t a, uint16_t b) {
  return FloatToBiovaultBFloat16(BiovaultBFloat16ToFloat(a) /
                                 BiovaultBFloat16ToFloat(b));
}

uint16_t BiovaultBFloat16Max(uint16_t a, uint16_t b) {
  auto a_f = BiovaultBFloat16ToFloat(a);
  auto b_f = BiovaultBFloat16ToFloat(b);
  return FloatToBiovaultBFloat16(a_f > b_f ? a_f : b_f);
}

uint16_t BiovaultBFloat16Sqrt(uint16_t a) {
  return FloatToBiovaultBFloat16(sqrt(BiovaultBFloat16ToFloat(a)));
}

uint16_t BiovaultBFloat16Exp(uint16_t a) {
  return FloatToBiovaultBFloat16(exp(BiovaultBFloat16ToFloat(a)));
}
}
