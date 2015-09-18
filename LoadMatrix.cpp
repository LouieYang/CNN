#include "LoadMatrix.h"

void LoadMatrix(Eigen::MatrixXf& matrix, std::string address,
                const int cols, const int rows)
{
    matrix = Eigen::MatrixXf(rows, cols);
    
    std::ifstream inf;
    
    inf.open(address, std::ifstream::in);
    
    std::string line;
    
    size_t firComma = 0;
    size_t secComma = 0;
    
    int col = 0;
    int row = 0;
    
    while (!inf.eof())
    {
        getline(inf,line);
        
        firComma = line.find(',', 0);
        
        if (row == rows)
        {
            break;
        }
        
        matrix(row, col++) = atof(line.substr(0,firComma).c_str());
        
        while (firComma < line.size() && col <= cols - 1)
        {
            secComma = line.find(',',firComma + 1);
            matrix(row, col++) = atof(line.substr(firComma + 1,secComma-firComma-1).c_str());
            firComma = secComma;
        }
        if (col == cols)
        {
            col = 0;
            ++row;
        }
    }
    inf.close();
}