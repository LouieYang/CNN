#ifndef __CNN__CNN__
#define __CNN__CNN__

#include <vector>
#include <iostream>
#include <memory>
#include <random>

#include "LoadMatrix.h"

#define DEFAULT_POOL 2
#define DEFAULT_LEARNING_RATE 0.1
#define DSIGMOID(x) x * (1 - x)

struct LayerAttribute
{
    unsigned int size_conv;
    unsigned int neurons;
    char LayerType;
    MatrixXf linkage;
};

using LayerAttributes = std::vector<struct LayerAttribute>;

class CNNLayer;
using VectorLayers = std::vector<std::shared_ptr<CNNLayer>>;
using Matrices = std::vector<MatrixXf>;

class ConvolutionalNeuronNetwork
{
public:
    ConvolutionalNeuronNetwork(LayerAttributes attributes, int image_row,
                               int image_col, int labels, int train_samples,
                               int test_samples);
    ConvolutionalNeuronNetwork(LayerAttributes attributes, int image_row, int image_col, int labels);
    ConvolutionalNeuronNetwork(std::string properties, int ImageRow, int ImageCol, int n_labels, Matrices links);
    
    VectorXf FeedForward(MatrixXf inputImage);
    void BackPropagate(VectorXf actualOutput, VectorXf desireOutput);
    void ApplyGradient(int batch_size, double etaLearningRate);
    void CleanGradient();
    void Train(int epoches, int batch_size, double etaLearningRate);
    
    void UploadTrainWeight(std::string dst);
    void UploadTrainBias(std::string dst);
    void DownloadTrainWeight(std::string src);
    void DownloadTrainBias(std::string src);
    
    MatrixXf get_i_image(int index, std::string set);
    VectorXf get_i_label(int index, std::string set);
    
    void LoadTrainImage();
    void LoadTrainImage(Matrices train_image);
    void LoadTestImage();
    void LoadTestImage(Matrices test_image);
    void LoadTrainLabel();
    void LoadTrainLabel(MatrixXf train_label);
    void LoadTestLabel();
    void LoadTestLabel(MatrixXf test_label);
    
    int get_image_rows() {  return n_image_rows;    }
    int get_image_cols() {  return n_image_cols;    }
    
    VectorLayers m_CNNLayers;

private:
    
    const int n_layers;
    const int n_image_rows;
    const int n_image_cols;
    const int n_output_dim;
    
    int n_conv_layer;
    
    int n_train_samples;
    int n_test_samples;
    
    std::vector<MatrixXf> m_train_Images;
    std::vector<MatrixXf> m_test_Images;
    
    MatrixXf m_train_label;
    MatrixXf m_test_label;
    
    MatrixXf MLPWeightMatrix;
    MatrixXf dMLPWeightMatrix;
    VectorXf MLPBiasVector;
    VectorXf dMLPBiasVector;
    VectorXf MLP_input;
};

class CNNLayer
{
public:
    
    CNNLayer(long input_rows, long input_cols,
             int conv, int units, char category,
             MatrixXf linkMatrix, CNNLayer* prev = nullptr);
    
    void Calculate();
    void BackPropagate(Matrices& Err_dXn, Matrices& Err_dXnm1);
    
    char get_property() {   return m_property;  }

    Matrices m_Neuron_Value;

    const int n_units;

    const char m_property;
    
    const int l_conv;
    
    Matrices m_conv_weight;
    Matrices d_conv_weight;

    Matrices m_conv_bias;
    Matrices d_conv_bias;
    
    MatrixXf m_link_matrix;
    
    CNNLayer* prev_CNN;
};

MatrixXf Convolution(MatrixXf input, MatrixXf conv, char mode);
MatrixXf Pool(MatrixXf input, int pool_coeff);
MatrixXf Rot180(MatrixXf input);

void DePool(MatrixXf& input, MatrixXf output, int pool_coeff);
void matrix_sigmoid(MatrixXf& matrix);
double sigmoid(double input);
std::vector<int> get_shuffled_index(int n);

int max_label_index(VectorXf label);
#endif /* defined(__CNN__CNN__) */
