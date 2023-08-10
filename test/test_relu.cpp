#include <gtest/gtest.h>
#include <random>
#include "relu.hpp"

TEST(test_layer, test_relu1) {
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

TEST(test_layer, test_relu2) {
    std::random_device rd;
    std::mt19937 gen(rd());

    // 定义随机数分布
    std::uniform_real_distribution<float> dist(-20.0, 20.0);

    relu_layer relu;
    uint test_lenth = 100000;
    relu.set_length(test_lenth);
    float* input = new float[test_lenth];
    float* output = new float[test_lenth];

    for (int i = 0; i < test_lenth; i++) {
        input[i] = dist(gen);
    }

    relu.forward(input, output);

    for (int i = 0; i < test_lenth; i++) {
        ASSERT_EQ(output[i], (input[i] >=0) ? input[i] : 0);
    }

    delete[] input;
    delete[] output;
}