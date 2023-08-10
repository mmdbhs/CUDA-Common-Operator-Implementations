#ifndef __SILU_HPP__
#define __SILU_HPP__

class silu_layer {
public:
    void forward(float* input, float* output);
    void set_length(unsigned int length) { m_length = length; }

private:
    unsigned int m_length = 0;
};

#endif /* __SILU_HPP__ */