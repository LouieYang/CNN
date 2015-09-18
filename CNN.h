#ifndef __CNN__CNN__
#define __CNN__CNN__

#include <vector>
#include <iostream>

#include "LoadMatrix.h"

#define DEFAULT_POOL 2
#define DEFAULT_LEARNING_RATE 0.5
#define DSIGMOID(x) x * (1 - x)


class CNNLayer;
using VectorLayers = std::vector<CNNLayer*>;
using Matrices = std::vector<MatrixXf>;

class ConvolutionalNeuronNetwork
{
public:
    ConvolutionalNeuronNetwork(std::string properties, int ImageRow, int ImageCols, int n_labels, int train_samples, int test_samples, Matrices links);
    
    VectorXf FeedForward(MatrixXf inputImage);
    void BackPropagate(VectorXf actualOutput, VectorXf desireOutput, double etaLearningRate);
    
    MatrixXf get_i_image(int index, std::string set);
    VectorXf get_i_label(int index, std::string set);
    
    void LoadTrainImage();
    void LoadTestImage();
    void LoadTrainLabel();
    void LoadTestLabel();
    
private:
    
    const int n_layers;
    const int n_image_rows;
    const int n_image_cols;
    const int n_output_dim;
    const int n_train_samples;
    const int n_test_samples;
    
    std::vector<unsigned int> m_convs;
    std::vector<unsigned int> m_units;
    
    std::vector<MatrixXf> m_train_Images;
    std::vector<MatrixXf> m_test_Images;
    
    MatrixXf m_train_label;
    MatrixXf m_test_label;
    
    MatrixXf MLPWeightMatrix;
    VectorXf MLPBiasVector;
    VectorXf MLP_input;
    
    VectorLayers m_CNNLayers;
    
};

class CNNLayer
{
public:
    
    CNNLayer(long input_rows, long input_cols,
             int conv, int units, char category,
             MatrixXf linkMatrix, CNNLayer* prev = nullptr);
    
    void Calculate();
    void BackPropagate(Matrices& Err_dXn, Matrices& Err_dXnm1, double etaLearningRate);
    
    char get_property() {   return m_property;  }

    Matrices m_Neuron_Value;

    const int n_units;

    const char m_property;

private:
    
    const int l_conv;
    
    Matrices m_conv_weight;
    Matrices m_conv_bias;

    MatrixXf m_link_matrix;
    
    CNNLayer* prev_CNN;
};

MatrixXf Convolution(MatrixXf input, MatrixXf conv, char mode);
MatrixXf Pool(MatrixXf input, int pool_coeff);
MatrixXf Rot180(MatrixXf input);

void DePool(MatrixXf& input, MatrixXf output, int pool_coeff);
void matrix_sigmoid(MatrixXf& matrix);
double sigmoid(double input);
#endif /* defined(__CNN__CNN__) */
