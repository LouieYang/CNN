/*******************************************************************
 *  Copyright(c) 2015
 *  All rights reserved.
 *
 *  Name: Convolutional Neural Network
 *  Description: CNN is a kind of sparse NN to learn the correlation between
    data. It has been widely used in image recognition, natural language. Its
    sparsity comes from its local linkage and shared weight of each neuron
 *  Date: 2015-9-18
 *  Author: Yang
 *  Intruction: Use 50000 handwritten digit images to train the CNN and test
    with 10000 images.  Data from MNIST. Also trained by ORL face database
    from University of Cambridge
 
 ******************************************************************/

#include "CNN.h"

ConvolutionalNeuronNetwork::ConvolutionalNeuronNetwork(std::string properties,
                                                       int ImageRow, int ImageCol,int n_labels, int train_samples, int test_samples, Matrices links):
n_output_dim(n_labels), n_train_samples(train_samples),
n_image_rows(ImageRow), n_image_cols(ImageCol), n_layers(int(properties.size())),
n_test_samples(test_samples)
{
    /*
     *  Description:
     *  A constructed function to initialize the CNN structure
     *
     *  @param properties: The structure of conv layer and pool layer(e.g. "ccpcp")
     *
     *  @param ImageRow: train and test image height
     *  @param Imagecol: train and test image width
     *  @param n_labels: the output dimension(labels)
     *  
     *  @param train_samples: number of train samples
     *  @param test_samples: number of test samples
     *
     *  @param links: vector of matrix to represent the local linkage
     *                  (If the layer is "pool", just an diagonal matrix)
     */
    
    
    /* Convert the data matrix to vector of image vector */
    LoadTrainImage();
    LoadTrainLabel();
    LoadTestImage();
    LoadTestLabel();
    
    /* Input the convolutional matrix size */
    std::cout << "Please input your conv matrice' size" << std::endl;
    for (int i = 0; i < n_layers; i++)
    {
        unsigned int conv;
        std::cin >> conv;
        m_convs.push_back(conv);
    }
    
    std::cout << "Please input your units in each layer" << std::endl;
    for (int i = 0; i < n_layers; i++)
    {
        unsigned int unit;
        std::cin >> unit;
        m_units.push_back(unit);
    }
    
    
    
    /******************************************************************
        Construct the structure of CNN
    ******************************************************************/
    
    /* The first layer must be the input image(i.e. needn't set conv size) */
    CNNLayer* Layer = new CNNLayer(n_image_rows, n_image_cols, 1, 1, 'c',
                                   MatrixXf::Zero(1, 1));
    m_CNNLayers.push_back(Layer);
    for (int i = 1; i < n_layers; i++)
    {
        Layer = new CNNLayer(Layer->m_Neuron_Value[0].rows(), Layer->m_Neuron_Value[0].cols(), m_convs[i], m_units[i], properties[i], links[i], Layer);
        m_CNNLayers.push_back(Layer);
    }
    
    /* The final layer of CNN has to be a completely linked perceptron */
    long n_MLP_input_dim = m_CNNLayers[n_layers - 1]->m_Neuron_Value[0].cols() * m_CNNLayers[n_layers - 1]->m_Neuron_Value[0].rows() * m_CNNLayers[n_layers - 1]->n_units;
    
    double init_weight = sqrt(1./(1 + n_output_dim + n_MLP_input_dim));
    MLPWeightMatrix = MatrixXf::Random(n_output_dim, n_MLP_input_dim) * init_weight;
    MLPBiasVector = VectorXf::Zero(n_output_dim);
}

VectorXf ConvolutionalNeuronNetwork::FeedForward(MatrixXf inputImage)
{
    /*
     *  Description:
     *  Feed forward in the CNN to predict the label of input image
     *
     */
    
    VectorLayers::iterator lit = m_CNNLayers.begin();
    if (lit == m_CNNLayers.end())
    {
        std::cerr << "No Layers!" << std::endl;
    }
    
    /* Put the image into the CNN */
    (*lit)->m_Neuron_Value[0] = inputImage;
    for (lit++; lit < m_CNNLayers.end(); lit++)
    {
        (*lit)->Calculate();
    }
    lit--;
    
    
    /******************************************************************
     Unfold the result of previous CNN(matrix) into vector and put it 
     into perceptron
     ******************************************************************/
    
    MLP_input = VectorXf::Zero(MLPWeightMatrix.cols());
    long rows = (*lit)->m_Neuron_Value[0].rows();
    long cols = (*lit)->m_Neuron_Value[0].cols();
    
    for (int i = 0; i < (*lit)->n_units; i++)
    {
        /* Unfold the matrix to vector */
        MatrixXf matrix = (*lit)->m_Neuron_Value[i];
        matrix.resize(cols * rows, 1);
        
        /* Link the unfolded vector in the last layer */
        MLP_input.segment(i * rows * cols, rows * cols) = matrix;
    }
    VectorXf outputLabel = MLPWeightMatrix * MLP_input + MLPBiasVector;
    for (int i = 0; i < n_output_dim; i++)
    {
        outputLabel(i) = sigmoid(outputLabel(i));
    }
    return outputLabel;
}

void ConvolutionalNeuronNetwork::BackPropagate(VectorXf actualOutput,                                              VectorXf desireOutput, double etaLearningRate)
{
    
    /*
     *  Description:
     *  Back propagate the error in the CNN to update weight and bias
     *
     *  @param actualOutput: label predict by the feed forward
     *  @param desireOutput: corresponding label
     *  @param etaLearningRate: learning rate of the last perceptron
     *
     */
    
    /* perceptron error */
    VectorXf diff = desireOutput - actualOutput;
    for (int i = 0; i < diff.size(); i++)
    {
        diff(i) *= DSIGMOID(actualOutput(i));
    }
    MLPWeightMatrix += etaLearningRate * diff * MLP_input.transpose();
    MLPBiasVector += etaLearningRate * diff;    /* Update weight and bias */

    
    /******************************************************************
     Fold the error vector into matrice to BP in CNN
     ******************************************************************/
    
    /* The last but one error(error of last CNN layer) */
    VectorXf diff_Xlast = MLPWeightMatrix.transpose() * diff;
    
    long rows = m_CNNLayers[n_layers - 1]->m_Neuron_Value[0].rows();
    long cols = m_CNNLayers[n_layers - 1]->m_Neuron_Value[0].cols();
    
    VectorLayers::iterator lit = m_CNNLayers.end() - 1;

    Matrices Err_dXlast;
    Err_dXlast.resize((*lit)->n_units);
    
    for (int i = 0; i < (*lit)->n_units; i++)
    {
        /* Fold procedure */
        MatrixXf err_decompressed = diff_Xlast.segment(i * cols * rows, cols * rows);
        err_decompressed.resize(rows, cols);
        Err_dXlast[i] = err_decompressed;
//        Err_dXlast[i] = MatrixXf::Zero(rows, cols);
//        for (int row = 0; row < rows; row++)
//        {
//            for (int col = 0; col < cols; col++)
//            {
//                Err_dXlast[i](row, col) = DSIGMOID(m_CNNLayers[n_layers - 1]->m_Neuron_Value[i](row, col)) * err_decompressed(row, col);
//            }
//        }
    }
    
    /******************************************************************
     Back propagate in the  CNN layer
     ******************************************************************/
    
    std::vector<Matrices> differentials;
    differentials.resize(n_layers);
    differentials[n_layers - 1] = Err_dXlast;
    
    /* Set every layer error default weight size */
    for (int i = 0; i < n_layers - 1; i++)
    {
        differentials[i].resize(m_CNNLayers[i]->n_units);
        for (int j = 0; j < m_CNNLayers[i]->n_units; j++)
        {
            differentials[i][j] = MatrixXf::Zero(m_CNNLayers[i]->m_Neuron_Value[0].rows(), m_CNNLayers[i]->m_Neuron_Value[0].cols());
        }
    }
    
    int ii = n_layers - 1;
    for (; lit > m_CNNLayers.begin(); lit--)
    {
        (*lit)->BackPropagate(differentials[ii], differentials[ii - 1], DEFAULT_LEARNING_RATE);
        --ii;
    }
    
    differentials.clear();
}

MatrixXf ConvolutionalNeuronNetwork::get_i_image(int index, std::string set)
{
    /*
     *  Description:
     *  Get the i-th image from train set or test set
     */
    
    if (set == "train")
    {
        if (index >= m_train_Images.size())
        {
            std::cerr << "Out of images range" << std::endl;
        }
        return m_train_Images[index];
    }
    else if (set == "test")
    {
        if (index >= m_test_Images.size())
        {
            std::cerr << "Out of images range" << std::endl;
        }
        return m_test_Images[index];
    }
    else
    {
        std::cerr << "Wrong set";
        return MatrixXf::Zero(0, 0);
    }
}

VectorXf ConvolutionalNeuronNetwork::get_i_label(int index, std::string set)
{
    /*
     *  Description:
     *  Get the i-th label from train set or test set
     */

    if (set == "train")
    {
        if (index >= m_train_label.size())
        {
            std::cerr << "Out of range" << std::endl;
        }
        return m_train_label.row(index);
    }
    else if (set == "test")
    {
        if (index >= m_test_label.size())
        {
            std::cerr << "Out of range" << std::endl;
        }
        return m_test_label.row(index);
    }
    else
    {
        std::cerr << "Wrong index" << std::endl;
        return VectorXf::Zero(0);
    }
}

void ConvolutionalNeuronNetwork::LoadTrainImage()
{
    /*
     *  Description:
     *  Load train image and normalize to 1
     */

    MatrixXf m_train_compressed_images;
    LoadMatrix(m_train_compressed_images, "./ORL/train_x", n_image_cols * n_image_rows, n_train_samples);
    m_train_compressed_images /= 256.;
    
    m_train_Images.resize(n_train_samples);
    for (int i = 0; i < n_train_samples; i++)
    {
        m_train_Images[i] = MatrixXf(n_image_rows, n_image_cols);
        for (int j = 0; j < n_image_rows; j++)
        {
            m_train_Images[i].row(j) = m_train_compressed_images.row(i).segment(j * n_image_cols, n_image_cols);
        }
    }
}

void ConvolutionalNeuronNetwork::LoadTestImage()
{
    /*
     *  Description:
     *  Load test image and normalize to 1
     */
    
    MatrixXf m_test_compressed_images;
    LoadMatrix(m_test_compressed_images, "./ORL/test_x", n_image_cols * n_image_rows, n_test_samples);
    m_test_compressed_images /= 256.;
    
    m_test_Images.resize(n_test_samples);
    for (int i = 0; i < n_test_samples; i++)
    {
        m_test_Images[i] = MatrixXf(n_image_rows, n_image_cols);
        for (int j = 0; j < n_image_rows; j++)
        {
            m_test_Images[i].row(j) = m_test_compressed_images.row(i).segment(j * n_image_cols, n_image_cols);
        }
    }
}

void ConvolutionalNeuronNetwork::LoadTrainLabel()
{
    /*
     *  Description:
     *  Load train label
     */
    
    LoadMatrix(m_train_label, "./ORL/train_y", n_output_dim, n_train_samples);
}

void ConvolutionalNeuronNetwork::LoadTestLabel()
{
    /*
     *  Description:
     *  Load test label
     */
    
    LoadMatrix(m_test_label, "./ORL/Desktop/test_y", n_output_dim, n_test_samples);
}



/***************************************************************************/



CNNLayer::CNNLayer(long input_rows, long input_cols,
            int conv, int units, char category, MatrixXf linkMatrix,
                   CNNLayer* prev):
l_conv(conv), m_property(category), n_units(units)
{
    /*
     *  Description:
     *  The construction function of each CNN layer
     *
     *  @param input_rows: the previous layer's output matrice rows
     *  @param input_cols: the previous layer's output matrice cols
     *
     *  @param conv: the convolutional matrix rows and cols(same by
     *  default)
     *  @param category: 'c' means convolutional, 'p' means pool layer
     *
     *  @param linkMatrix: The matrix linkage of the current matrices
     *  and previous matrices(e.g. size of it is current units' number
     *  rows and previous units' number cols
     *
     *  @param prev: pointer point to the previous layer
     */
    
    
    m_Neuron_Value.resize(n_units);
    if (category == 'c')        /* Initial the convolutional layer */
    {
        long l_output_rows = input_rows - l_conv + 1;
        long l_output_cols = input_cols - l_conv + 1;
        
        m_conv_bias.resize(n_units);
        m_conv_weight.resize(n_units);
        double init_weight = sqrt(6./(1 + l_output_rows * l_output_cols + input_cols * input_rows));
        
        for (int i = 0; i < n_units; i++)
        {
            m_Neuron_Value[i] = MatrixXf::Zero(l_output_rows, l_output_cols);
            m_conv_weight[i] = init_weight * MatrixXf::Random(l_conv, l_conv);
            m_conv_bias[i] = MatrixXf::Zero(l_output_rows, l_output_cols);
        }
    }
    else if (category == 'p')   /* Initial the pool layer */
    {
        long l_output_rows = input_rows / DEFAULT_POOL;
        long l_output_cols = input_cols / DEFAULT_POOL;
        
        for (int i = 0; i < n_units; i++)
        {
            m_Neuron_Value[i] = MatrixXf::Zero(l_output_rows, l_output_cols);
        }
    }
    else
    {
        std::cerr << "Bad layer category" << std::endl;
    }
    
    m_link_matrix = linkMatrix;
    prev_CNN = prev;
}

void CNNLayer::Calculate()
{
    /*
     *  Description:
     *  Feed forward in each CNN layer
     */
    
    if (m_property == 'c')      /* If current layer is convolutional */
    {
        for (int i = 0; i < n_units; i++)
        {
            for (int j = 0; j < prev_CNN->n_units; j++)
            {
                /* Find if i-th neuron linkage with the j-th neuron */
                if (m_link_matrix(i, j) != 0)
                {
                    m_Neuron_Value[i] += Convolution(prev_CNN->m_Neuron_Value[j], m_conv_weight[i], 'v');
                }
            }
            m_Neuron_Value[i] += m_conv_bias[i];
            matrix_sigmoid(m_Neuron_Value[i]);  /* Use sigmoid as activation */
        }
    }
    else if (m_property == 'p')
    {
        /* Since the linkage matrix is diagonal, just pool it */
        for (int i = 0; i < n_units; i++)
        {
            m_Neuron_Value[i] = Pool(prev_CNN->m_Neuron_Value[i],
                                     DEFAULT_POOL);
        }
    }
}

void CNNLayer::BackPropagate(Matrices &Err_dXn, Matrices &Err_dXnm1, double etaLearningRate)
{
    /*
     *  Description:
     *  Back propagation in the current layer
     *  
     *  @param Err_dXn: the error of current layer(Known)
     *  @param Err_dXnm1: the error of previous layer(Unkown)
     *  @param etaLearningRate: learning step
     *
     */
    
    if (m_property == 'c')
    {
        for (int i = 0; i < n_units; i++)
        {
            MatrixXf err = Err_dXn[i];
            for (int j = 0; j < prev_CNN->n_units; j++)
            {
                /* Find if j-th error of previous links the i-th of current's*/
                if (m_link_matrix(i, j) != 0)
                {
                    /* The formulation can be seen in .md file */
                    Err_dXnm1[j] += Convolution(err, Rot180(m_conv_weight[i]), 'f');
                    m_conv_weight[i] += etaLearningRate * Convolution(prev_CNN->m_Neuron_Value[j], err, 'v');
                }
            }
            m_conv_bias[i] += etaLearningRate * err;
        }
    }
    else if (m_property == 'p')
    {
        for (int i = 0; i < n_units; i++)
        {
            /* The previous layer have used sigmoid as activation */
            for (int j = 0; j < Err_dXn[i].rows(); j++)
            {
                for (int k = 0; k < Err_dXn[i].cols(); k++)
                {
                    Err_dXn[i](j, k) *= DSIGMOID(m_Neuron_Value[i](j, k));
                }
            }
            /* Average the errors into bigger matrice */
            DePool(Err_dXnm1[i], Err_dXn[i], DEFAULT_POOL);
        }
    }
    
}



/***************************************************************************/




MatrixXf Convolution(MatrixXf input, MatrixXf conv, char mode)
{
    /*
     *  Description:
     *  Function of convolution of two matrix
     *
     *  @param mode: 'v' corresponding the valid mode where the size of
     *  output is (input.rows - conv.rows + 1, input.cols - conv.cols + 1)
     *  'f' corresponding mode full mode where the size of output is 
     *  (input.rows + 2 * conv.rows - 2, input.cols + 2 * conv.cols - 2)
     *
     */
    
    MatrixXf output;
    /* The actual formulation can be seen in .md file */
    if (mode == 'v')        /* valid mode */
    {
        output.resize(input.rows() - conv.rows() + 1, input.cols() - conv.cols() + 1);
        for (int i = 0; i < output.rows(); i++)
        {
            for (int j = 0; j < output.cols(); j++)
            {
                output(i, j) = (input.block(i, j, conv.rows(), conv.cols()).array() * conv.array()).sum();
            }
        }
    }
    else if (mode == 'f')   /* full mode */
    {
        output = MatrixXf::Zero(input.rows() + 2 * conv.rows() - 2, input.cols() + 2 * conv.cols() - 2);
        output.block(conv.rows() - 1, conv.cols() - 1, input.rows(), input.cols()) = input;
        return Convolution(output, conv, 'v');
    }
    return output;
}

MatrixXf Pool(MatrixXf input, int pool_coeff)
{
    /*
     *  Description:
     *  Function of pooling
     *
     *  @param pool_coeff: the size of sampling coefficient
     *
     */
    
    MatrixXf output(input.rows() / pool_coeff, input.cols() / pool_coeff);
    for (int i = 0; i < output.rows(); i++)
    {
        for (int j = 0; j < output.cols(); j++)
        {
            output(i, j) = input.block(pool_coeff * i, pool_coeff * j, pool_coeff, pool_coeff).sum() / pow(pool_coeff, 2);
        }
    }
    return output;
}

MatrixXf Rot180(MatrixXf input)
{
    /*
     *  Description:
     *  Function of rotating the matrix by 180 degree
     */
    
    return input.transpose().reverse();
}

void DePool(MatrixXf& input, MatrixXf output, int pool_coeff)
{
    /*
     *  Description:
     *  Function of depooling the matrix
     */
    
    for (int i = 0; i < output.rows(); i++)
    {
        for (int j = 0; j < output.cols(); j++)
        {
            input.block(pool_coeff * i, pool_coeff * j, pool_coeff, pool_coeff) = MatrixXf::Ones(pool_coeff, pool_coeff) * output(i, j) / pow(pool_coeff, 2);
        }
    }
}

void matrix_sigmoid(MatrixXf& matrix)
{
    /*
     *  Description:
     *  Function of mapping each element in matrix with activation function
     */
    
    for (int i = 0; i < matrix.rows(); i++)
    {
        for (int j = 0; j < matrix.cols(); j++)
        {
            matrix(i, j) = sigmoid(matrix(i, j));
        }
    }
}

double sigmoid(double input)
{
    /*
     *  Description:
     *  Activation function
     */
    
    return 1./(1 + exp(-input));
}