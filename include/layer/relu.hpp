#ifndef RELU_HPP
#define RELU_HPP

class relu_layer {
public:
    // relu_layer();
    // ~relu_layer();

    void set_length(unsigned int length) { m_length = length; }

    void forward(float *input, float *output);

private:
    unsigned int m_length = 0;
};



#endif /* RELU_HPP */