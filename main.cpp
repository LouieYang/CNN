#include "CNN.h"

int main()
{
    Matrices m;
    m.push_back(MatrixXf::Ones(1, 1));
    m.push_back(MatrixXf::Ones(6, 1));
    m.push_back(MatrixXf::Identity(6, 6));
    MatrixXf mm(16, 6);
    mm << 1, 1, 1, 0, 0, 0,
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
    m.push_back(mm);
    m.push_back(MatrixXf::Identity(16, 16));
    std::string s("ccpcp");
    ConvolutionalNeuronNetwork CNN(s, 28, 28, 10, 50000, 10000, m);
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