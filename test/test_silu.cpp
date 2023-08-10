#include <gtest/gtest.h>
#include <random>
#include "silu.hpp"


TEST(test_layer, test_silu) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // 定义随机数分布
    std::uniform_real_distribution<float> dist(-20.0, 20.0);

    silu_layer silu;
    uint test_lenth = 100000;
    silu.set_length(test_lenth);

    float* input = new float[test_lenth];
    float* output = new float[test_lenth];

    for (int i = 0; i < test_lenth; i++) {
        input[i] = dist(gen);
    }

    silu.forward(input, output);

    for (int i = 0; i < test_lenth; i++) {
        ASSERT_FLOAT_EQ(output[i], input[i] * (1.0/(1.0 + expf(input[i]))));
    }

    delete[] input;
    delete[] output;
}