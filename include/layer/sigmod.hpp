#ifndef __SIGMOD_HPP__
#define __SIGMOD_HPP__

class sigmod_layer {
public:
    void forward(float* input, float* output);
    void set_length(unsigned int length) { m_length = length; }

private:
    unsigned int m_length = 0;
};




#endif /* __SIGMOD_HPP__ */