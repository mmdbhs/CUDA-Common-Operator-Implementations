#include <gtest/gtest.h>
#include "relu.hpp"

TEST(test_layer_, test_relu1) {
    relu_layer relu;
    relu.set_length(5);
    float* input = new float[5];
    float* output = new float[5];
    input[0] = -2;
    input[1] = -1;
    input[2] = 0;
    input[3] = 1;
    input[4] = 2;

    relu.forward(input, output);

    ASSERT_EQ(output[0], 0);
    ASSERT_EQ(output[1], 0);
    ASSERT_EQ(output[2], 0);
    ASSERT_EQ(output[3], 1);
    ASSERT_EQ(output[4], 2);

    delete[] input;
    delete[] output;
}