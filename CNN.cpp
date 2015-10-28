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

ConvolutionalNeuronNetwork::ConvolutionalNeuronNetwork(LayerAttributes attributes, int image_row, int image_col, int labels, int train_samples, int test_samples):
n_layers(int(attributes.size())), n_image_rows(image_row), n_image_cols(image_col), n_output_dim(labels), n_train_samples(train_samples), n_test_samples(test_samples)
{
    /*
     *  Description:
     *  A constructed function to initialize the CNN structure
     *
     *  @param attributes: The vector of struct LayerAttribute
     *
     *  @param ImageRow: train and test image height
     *  @param Imagecol: train and test image width
     *  @param n_labels: the output dimension(labels)
     *
     *  @param train_samples: number of train samples
     *  @param test_samples: number of test samples
     *
     */
    
    /* Convert the data matrix to vector of image vector */
    
    LoadTrainImage();
    LoadTrainLabel();
    LoadTestImage();
    LoadTestLabel();
    
    /******************************************************************
     Construct the structure of CNN
     ******************************************************************/
    
    /* The first layer must be the input image(i.e. needn't set conv size) */
    CNNLayer* Layer = new CNNLayer(n_image_rows, n_image_cols, attributes[0].size_conv, attributes[0].neurons, attributes[0].LayerType, attributes[0].linkage);
    
    m_CNNLayers.push_back(std::shared_ptr<CNNLayer>(Layer));
    for (int i = 1; i < n_layers; i++)
    {
        Layer = new CNNLayer(Layer->m_Neuron_Value[0].rows(), Layer->m_Neuron_Value[0].cols(), attributes[i].size_conv, attributes[i].neurons, attributes[i].LayerType, attributes[i].linkage, Layer);
        m_CNNLayers.push_back(std::shared_ptr<CNNLayer>(Layer));
    }
    
    /* The final layer of CNN has to be a completely linked perceptron */
    long n_MLP_input_dim = m_CNNLayers[n_layers - 1]->m_Neuron_Value[0].cols() * m_CNNLayers[n_layers - 1]->m_Neuron_Value[0].rows() * m_CNNLayers[n_layers - 1]->n_units;
    
    double init_weight = sqrt(1./(1 + n_output_dim + n_MLP_input_dim));
    MLPWeightMatrix = MatrixXf::Random(n_output_dim, n_MLP_input_dim) * init_weight;
    dMLPWeightMatrix = MatrixXf::Zero(n_output_dim, n_MLP_input_dim);
    
    MLPBiasVector = VectorXf::Zero(n_output_dim);
    dMLPBiasVector = VectorXf::Zero(n_output_dim);
    
    Layer = nullptr;
    delete Layer;

}

ConvolutionalNeuronNetwork::ConvolutionalNeuronNetwork(LayerAttributes attributes, int image_row, int image_col, int labels): n_layers(int(attributes.size())), n_image_rows(image_row), n_image_cols(image_col), n_output_dim(labels)
{
    /*
     *  Description:
     *  Load the trained CNN
     */
    
    /******************************************************************
     Construct the structure of CNN
     ******************************************************************/
    
    /* The first layer must be the input image(i.e. needn't set conv size) */
    CNNLayer* Layer = new CNNLayer(n_image_rows, n_image_cols, attributes[0].size_conv, attributes[0].neurons, attributes[0].LayerType, attributes[0].linkage);
    
    m_CNNLayers.push_back(std::shared_ptr<CNNLayer>(Layer));
    for (int i = 1; i < n_layers; i++)
    {
        Layer = new CNNLayer(Layer->m_Neuron_Value[0].rows(), Layer->m_Neuron_Value[0].cols(), attributes[i].size_conv, attributes[i].neurons, attributes[i].LayerType, attributes[i].linkage, Layer);
        m_CNNLayers.push_back(std::shared_ptr<CNNLayer>(Layer));
    }
    
    /* The final layer of CNN has to be a completely linked perceptron */
    long n_MLP_input_dim = m_CNNLayers[n_layers - 1]->m_Neuron_Value[0].cols() * m_CNNLayers[n_layers - 1]->m_Neuron_Value[0].rows() * m_CNNLayers[n_layers - 1]->n_units;
    
    double init_weight = sqrt(1./(1 + n_output_dim + n_MLP_input_dim));
    MLPWeightMatrix = MatrixXf::Random(n_output_dim, n_MLP_input_dim) * init_weight;
    dMLPWeightMatrix = MatrixXf::Zero(n_output_dim, n_MLP_input_dim);
    
    MLPBiasVector = VectorXf::Zero(n_output_dim);
    dMLPBiasVector = VectorXf::Zero(n_output_dim);
    
    Layer = nullptr;
    delete Layer;
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
    for (lit++; lit != m_CNNLayers.end(); lit++)
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

void ConvolutionalNeuronNetwork::BackPropagate(VectorXf actualOutput,                                              VectorXf desireOutput)
{
    
    /*
     *  Description:
     *  Back propagate the error in the CNN to update weight and bias
     *
     *  @param actualOutput: label predict by the feed forward
     *  @param desireOutput: corresponding label
     *
     */
    
    /* perceptron error */
    VectorXf diff = actualOutput - desireOutput;
    for (int i = 0; i < diff.size(); i++)
    {
        diff(i) *= DSIGMOID(actualOutput(i));
    }

    dMLPWeightMatrix += diff * MLP_input.transpose();
    dMLPBiasVector += diff;    /* Update weight and bias */

    
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
        (*lit)->BackPropagate(differentials[ii], differentials[ii - 1]);
        --ii;
    }
    
    differentials.clear();
}

void ConvolutionalNeuronNetwork::ApplyGradient(int batch_size, double etaLearningRate)
{
    /*
     *  Description:
     *  Apply the gradient into the convolutional network using batch average
     *  
     *  @param batch_size: The number of samples in each batch
     *  @param etaLearningRate: The learning step length
     *
     */
    
    if (batch_size == 0)
    {
        return;
    }
    
    for (int i = 1; i < n_layers; i++)
    {
        if (m_CNNLayers[i]->m_property == 'c')
        {
            long conv_matrix_row = m_CNNLayers[i]->l_conv;
            long conv_matrix_col = m_CNNLayers[i]->l_conv;
            long bias_matrix_row = m_CNNLayers[i]->m_Neuron_Value[0].rows();
            long bias_matrix_col = m_CNNLayers[i]->m_Neuron_Value[0].cols();
            
            for (int j = 0; j < m_CNNLayers[i]->n_units; j++)
            {
                m_CNNLayers[i]->m_conv_weight[j] -= etaLearningRate * m_CNNLayers[i]->d_conv_weight[j] / batch_size;
                
                m_CNNLayers[i]->m_conv_bias[j] -= etaLearningRate * m_CNNLayers[i]->d_conv_bias[j] / batch_size;
                
                /* Clean the buffer */
                m_CNNLayers[i]->d_conv_weight[j] = MatrixXf::Zero(conv_matrix_row, conv_matrix_col);
                m_CNNLayers[i]->d_conv_bias[j] = MatrixXf::Zero(bias_matrix_row, bias_matrix_col);
            }
        }
    }
    
    /* Update the parameters in fully connected network */
    MLPWeightMatrix -= etaLearningRate * dMLPWeightMatrix / batch_size;
    MLPBiasVector -= etaLearningRate * dMLPBiasVector / batch_size;
    
}

void ConvolutionalNeuronNetwork::Train(int epoches, int batch_size, double etaLearningRate)
{
    /*
     *  Description:
     *  Train schedule with batch size fixed. The total iteration is known as epoches.
     *
     *  @param epoches: The number of iteration
     *  @param batch_size: The fixed batch size
     *  @param etaLearningRate: The length of learning step
     *
     */
    
    for (int ii = 1; ii <= epoches; ii++)
    {
        int iter = n_train_samples / batch_size;
        int left_batch = n_train_samples % batch_size;
        
        std::vector<int> train_image_index = get_shuffled_index(int(m_train_Images.size()));
        
        std::cout << ii << "/" << epoches << " Epoch"<< std::endl;
        
        for (int i = 0; i < iter; i++)
        {
            for (int j = 0; j < batch_size; j++)
            {
                BackPropagate(FeedForward(m_train_Images[train_image_index[i * batch_size + j]]), m_train_label.row(train_image_index[i * batch_size + j]));
            }
            ApplyGradient(batch_size, etaLearningRate);
            
            /* Always remember clean the buffer after each batch */
            CleanGradient();
        }
        
        for (int i = 0; i < left_batch; i++)
        {
            BackPropagate(FeedForward(m_train_Images[train_image_index[epoches * batch_size + i]]), m_train_label.row(train_image_index[iter * batch_size + i]));
        }
        ApplyGradient(left_batch, etaLearningRate);
        CleanGradient();
    }

    std::cout << "train over" << std::endl;
}

void ConvolutionalNeuronNetwork::CleanGradient()
{
    /*
     *  Description:
     *  The function to clean the buffer of the last batch
     *
     */
    
    for (int i = 1; i < n_layers; i++)
    {
        if (m_CNNLayers[i]->m_property == 'c')
        {
            for (int j = 0; j < m_CNNLayers[i]->n_units; j++)
            {
                m_CNNLayers[i]->d_conv_weight[j] = MatrixXf::Zero(m_CNNLayers[i]->l_conv, m_CNNLayers[i]->l_conv);
                m_CNNLayers[i]->d_conv_bias[j] = MatrixXf::Zero(m_CNNLayers[i]->m_conv_bias[0].rows(), m_CNNLayers[i]->m_conv_bias[0].cols());
            }
        }
    }
    
    dMLPBiasVector = VectorXf::Zero(dMLPBiasVector.size());
    dMLPWeightMatrix = MatrixXf::Zero(dMLPWeightMatrix.rows(), dMLPWeightMatrix.cols());
}


    /******************************************************************
     Start Upload and download module
     ******************************************************************/


void ConvolutionalNeuronNetwork::UploadTrainWeight(std::string dst)
{
    /*
     *  Description:
     *  Upload the train matrix to get it use
     *
     *  Writing rules:
     *  First line --- group of conv matrix
     *  First two lines of each group --- size of conv matrix(i.e. rows && cols, dim)
     *  At last, MLP rows and cols and then MLP content
     *
     */
    
    std::ofstream fw(dst);
    fw << n_layers << '\n';
    
    auto lit = m_CNNLayers.begin() + 1;
    for (int i = 1; i < n_layers; i++)
    {
        if ((*lit)->m_property != 'c')
        {
            ++lit;
            continue;
        }
        
        fw << (*lit)->l_conv << '\n';
        for (int j = 0; j < (*lit)->n_units; j++)
        {
            fw << (*lit)->m_conv_weight[j];
            fw << '\n';
        }
        lit++;
    }
    
    fw << MLPWeightMatrix.rows() << '\n' << MLPWeightMatrix.cols() << '\n';
    fw << MLPWeightMatrix;
    
    std::cout << "Successfully upload weight matrix" << std::endl;
}

void ConvolutionalNeuronNetwork::UploadTrainBias(std::string dst)
{
    /*
     *  Description:
     *  Upload the bias matrix
     *  
     *  Writing rules:
     *  First line --- group of bias
     *  First three line of each group --- bias matrix(rows, cols, dim)
     *  At last, upload bias vector(first line of dim)
     *
     */
    std::ofstream fw(dst);
    fw << n_layers << '\n';
    
    auto lit = m_CNNLayers.begin() + 1;
    for (int i = 1; i < n_layers; i++)
    {
        if ((*lit)->m_property != 'c')
        {
            lit++;
            continue;
        }

        fw << (*lit)->m_conv_bias[0].rows() << '\n'
            << (*lit)->m_conv_bias[0].cols() << '\n';
        
        for (int j = 0; j < (*lit)->n_units; j++)
        {
            fw << (*lit)->m_conv_bias[j];
            fw << '\n';
        }
        lit++;
    }
    
    fw << MLPBiasVector.size() << '\n';
    fw << MLPBiasVector;
    
    std::cout << "Successfully upload bias matrice and vectors\n";
}

void ConvolutionalNeuronNetwork::DownloadTrainWeight(std::string src)
{
    /*
     *  Description:
     *  Download the trained weight matrix(the same content as shown above)
     */
    
    std::ifstream fr(src);

    std::string str;
    getline(fr, str);
    
    /* Group size */
    assert(stoi(str) == int(n_layers));
    
    
    /* Download the conv matrix with string splitted by space */
    auto lit = m_CNNLayers.begin() + 1;
    for (int i = 1; i < n_layers; i++)
    {
        if ((*lit)->m_property != 'c')
        {
            lit++;
            continue;
        }

        getline(fr, str);
        assert((*lit)->l_conv == stoi(str));
        for (int k = 0; k < (*lit)->n_units; k++)
        {
            for (int r = 0; r < (*lit)->l_conv; r++)
            {
                getline(fr, str);
                int c = 0;
                size_t firSpace = str.find_first_not_of(" ", 0);
                size_t secSpace = str.find_first_of(" ", firSpace);
                (*lit)->m_conv_weight[k](r, c++) = atof(str.substr(firSpace, secSpace - firSpace).c_str());
                
                while (c < (*lit)->l_conv)
                {
                    firSpace = str.find_first_not_of(" ", secSpace);
                    secSpace = str.find_first_of(" ", firSpace);
                    (*lit)->m_conv_weight[k](r, c++) = atof(str.substr(firSpace, secSpace - firSpace).c_str());
                }
            }
        }
        lit++;
    }
    
    /* Download the MLP weight matrix */
    getline(fr, str);
    assert(stoi(str) == MLPWeightMatrix.rows());
    getline(fr, str);
    assert(stoi(str) == MLPWeightMatrix.cols());
    for (int r = 0; r < MLPWeightMatrix.rows(); r++)
    {
        int c = 0;
        getline(fr, str);
        
        size_t firSpace = str.find_first_not_of(" ", 0);
        size_t secSpace = str.find_first_of(" ", firSpace);
        MLPWeightMatrix(r, c++) = atof(str.substr(firSpace, secSpace - firSpace).c_str());
        
        while (firSpace < str.size() && c < MLPWeightMatrix.cols())
        {
            firSpace = str.find_first_not_of(" ", secSpace);
            secSpace = str.find_first_of(" ", firSpace);
            MLPWeightMatrix(r, c++) = atof(str.substr(firSpace, secSpace - firSpace).c_str());
        }
    }
    
    std::cout << "Successfully download weight matrix" << std::endl;
}

void ConvolutionalNeuronNetwork::DownloadTrainBias(std::string src)
{
    /*
     *  Description:
     *  Download the trained bias matrix(the same content as shown above)
     */
    
    std::ifstream fr(src);
    
    std::string str;
    getline(fr, str);
    
    /* Group size */
    assert(stoi(str) == int(n_layers));
    
    /* Download the bias matrix */
    auto lit = m_CNNLayers.begin() + 1;
    for (int i = 1; i < n_layers; i++)
    {
        if ((*lit)->m_property != 'c')
        {
            lit++;
            continue;
        }
        
        getline(fr, str);
        int bias_rows = stoi(str);
        getline(fr, str);
        int bias_cols = stoi(str);
        
        assert(bias_rows == (*lit)->m_conv_bias[0].rows());
        assert(bias_cols == (*lit)->m_conv_bias[0].cols());
        for (int k = 0; k < (*lit)->n_units; k++)
        {
            for (int r = 0; r < bias_rows; r++)
            {
                int c = 0;
                getline(fr, str);

                size_t firSpace = str.find_first_not_of(" ", 0);
                size_t secSpace = str.find_first_of(" ", firSpace);
                
                (*lit)->m_conv_bias[k](r, c++) = atof(str.substr(firSpace, secSpace - firSpace).c_str());
                
                while (firSpace < str.size() && c < bias_cols)
                {
                    firSpace = str.find_first_not_of(" ", secSpace);
                    secSpace = str.find_first_of(" ", firSpace);
                    
                    (*lit)->m_conv_bias[k](r, c++) = atof(str.substr(firSpace, secSpace - firSpace).c_str());
                }
            }
        }
        lit++;
    }
    
    /* Download bias vector */
    getline(fr, str);
    assert(MLPBiasVector.size() == stoi(str));
    for (int i = 0; i < MLPBiasVector.size(); i++)
    {
        getline(fr, str);
        MLPBiasVector(i) = atof(str.c_str());
    }
    
    std::cout << "Successfully download bias matrice and vectors\n";
}

    /******************************************************************
     End Upload and download module
     ******************************************************************/


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
    LoadMatrix(m_train_compressed_images, "/Users/liuyang/Desktop/Class/ML/DL/ANN/MNIST/train_x", n_image_cols * n_image_rows, n_train_samples);
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
    LoadMatrix(m_test_compressed_images, "/Users/liuyang/Desktop/Class/ML/DL/ANN/MNIST/test_x", n_image_cols * n_image_rows, n_test_samples);
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
    
    LoadMatrix(m_train_label, "/Users/liuyang/Desktop/Class/ML/DL/ANN/MNIST/train_y", n_output_dim, n_train_samples);
}

void ConvolutionalNeuronNetwork::LoadTestLabel()
{
    /*
     *  Description:
     *  Load test label
     */
    
    LoadMatrix(m_test_label, "/Users/liuyang/Desktop/Class/ML/DL/ANN/MNIST/test_y", n_output_dim, n_test_samples);
}


void ConvolutionalNeuronNetwork::LoadTrainImage(Matrices train_image)
{
    n_train_samples = (int)train_image.size();
    m_train_Images = train_image;
}

void ConvolutionalNeuronNetwork::LoadTestImage(Matrices test_image)
{
    n_test_samples = (int)test_image.size();
    m_test_Images = test_image;
}

void ConvolutionalNeuronNetwork::LoadTrainLabel(MatrixXf train_label)
{
    m_train_label = train_label;
}

void ConvolutionalNeuronNetwork::LoadTestLabel(MatrixXf test_label)
{
    m_test_label = test_label;
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
        
        d_conv_bias.resize(n_units);
        d_conv_weight.resize(n_units);
        
        double init_weight = sqrt(6./(1 + l_output_rows * l_output_cols + input_cols * input_rows));
        
        for (int i = 0; i < n_units; i++)
        {
            m_Neuron_Value[i] = MatrixXf::Zero(l_output_rows, l_output_cols);
            
            m_conv_weight[i] = init_weight * MatrixXf::Random(l_conv, l_conv);
            m_conv_bias[i] = MatrixXf::Zero(l_output_rows, l_output_cols);
            
            d_conv_weight[i] = MatrixXf::Zero(l_conv, l_conv);
            d_conv_bias[i] = MatrixXf::Zero(l_output_rows, l_output_cols);
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
     *
     */
    
    if (m_property == 'c')      /* If current layer is convolutional */
    {
        for (int i = 0; i < n_units; i++)
        {
            /* Clear the previous sample's result */
            m_Neuron_Value[i] = MatrixXf::Zero(m_conv_bias[0].rows(), m_conv_bias[0].cols());
            
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

void CNNLayer::BackPropagate(Matrices &Err_dXn, Matrices &Err_dXnm1)
{
    /*
     *  Description:
     *  Back propagation in the current layer
     *  
     *  @param Err_dXn: the error of current layer(Known)
     *  @param Err_dXnm1: the error of previous layer(Unkown)
     *
     */
    
    if (m_property == 'c')
    {
        
        for (int i = 0; i < n_units; i++)
        {
            /* The convolutioanl layer have used sigmoid as activation */
            for (int j = 0; j < Err_dXn[i].rows(); j++)
            {
                for (int k = 0; k < Err_dXn[i].cols(); k++)
                {
                    Err_dXn[i](j, k) *= DSIGMOID(m_Neuron_Value[i](j, k));
                }
            }

            MatrixXf err = Err_dXn[i];
            for (int j = 0; j < prev_CNN->n_units; j++)
            {
                /* Find if j-th error of previous links the i-th of current's*/
                if (m_link_matrix(i, j) != 0)
                {
                    /* The formulation can be seen in .md file */
                    Err_dXnm1[j] += Convolution(err, Rot180(m_conv_weight[i]), 'f');
                    d_conv_weight[i] += Convolution(prev_CNN->m_Neuron_Value[j], err, 'v');
                }
            }
            d_conv_bias[i] += err;
        }
    }
    else if (m_property == 'p')
    {
        for (int i = 0; i < n_units; i++)
        {
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
    
    MatrixXf rot90 = input.transpose().colwise().reverse();
    return rot90.transpose().colwise().reverse();
    
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

int max_label_index(VectorXf label)
{
    /*
     *  Description:
     *  Return the index of the maximum element in the vector
     */
    
    int index_max = 0;
    for (int i = 1; i < label.size(); i++)
    {
        if (label(i) > label(index_max))
        {
            index_max = i;
        }
    }
    return index_max;
}

std::vector<int> get_shuffled_index(int n)
{
    std::vector<int> v(n, 0);
    int k = 0;
    std::for_each(v.begin(), v.end(), [&k](int &vvalue){vvalue += (k++);});
    
    static std::random_device rd;
    std::mt19937 g(rd());
    
    std::shuffle(v.begin(), v.end(), g);
    return v;
}