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
    ConvolutionalNeuronNetwork CNN(s, 112, 92, 40, 360, 40, m);
    std::cout << "Contructed" << std::endl;

    
    for (int i = 0; i < 360; i++)
    {
        CNN.BackPropagate(CNN.FeedForward(CNN.get_i_image(i, "train")), CNN.get_i_label(i, "train"), DEFAULT_LEARNING_RATE);
    }
    std::cout << "Train over" << std::endl;

    int err = 0;
    for (int i = 0; i < 40; i++)
    {
        VectorXf error = CNN.FeedForward(CNN.get_i_image(i, "test")) - CNN.get_i_label(i, "test");
        if (error.sum() > 0.1 || error.sum() < -0.1)
        {
            err++;
            std::cout << CNN.FeedForward(CNN.get_i_image(i, "test"));
        }
    }
    std::cout << "The total error number in 10000 test is " << err << std::endl;
}