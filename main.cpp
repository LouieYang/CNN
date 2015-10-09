#include "CNN.h"

int main()
{
    MatrixXf m(16, 6);
    m << 1, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 0,
        0, 0, 0, 1, 1, 1,
        1, 0, 0, 0, 1, 1,
        1, 1, 0, 0, 0, 1,
        1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1,
        1, 0, 0, 1, 1, 1,
        1, 1, 0, 0, 1, 1,
        1, 1, 1, 0, 0, 1,
        1, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 1, 1,
        1, 0, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1;
    LayerAttributes attributes;
    attributes.push_back({1, 1, 'c', MatrixXf::Ones(1, 1)});
    attributes.push_back({5, 6, 'c', MatrixXf::Ones(6, 1)});
    attributes.push_back({1, 6, 'p', MatrixXf::Identity(6, 6)});
    attributes.push_back({5, 16, 'c', m});
    attributes.push_back({1, 16, 'p', MatrixXf::Identity(16, 16)});

    
    ConvolutionalNeuronNetwork CNN(attributes, 28, 28, 10, 50000, 10000);
    std::cout << "Contructed" << std::endl;
    

    CNN.Train(2, 50, DEFAULT_LEARNING_RATE);


    int err = 0;
    for (int i = 0; i < 10000; i++)
    {
        if (max_label_index(CNN.FeedForward(CNN.get_i_image(i, "test"))) !=
            max_label_index(CNN.get_i_label(i, "test")))
        {
            err++;
        }
    }
    std::cout << "The total error number in 10000 test is " << err << std::endl;
}