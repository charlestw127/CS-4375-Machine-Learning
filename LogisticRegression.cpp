//Charles Wallis
//Machine Learning
//Oct 2, 2022

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

using namespace std;
const int MAX_VALUE = 1046;

//DF
class dataField {
    public:
    vector<double> X;
    vector<double> survived;
    vector<double> pclass;
    vector<double> sex;
    vector<double> age;
    int num_instances;
};
    
// Store values in matrix
class Matrix {
    private:
        int rows = 1;
        int columns = 1;
    
    public:
        Matrix(int x,int y): rows(x), columns(y), matrix_values(rows * columns) {}
        void setValue(int x, int y, double value) {matrix_values[x * columns + y] = value;}
        double getValue(int x, int y) {return matrix_values[x * columns + y];}
        int getRows()   {return rows;}
        int getColumns(){return columns;}
        int getSize()   {return rows * columns;}
        vector<double> matrix_values;
    
    
    void fillMatrix(vector<double> x){
        if(x.size() == rows * columns){
            for(int i = 0; i < rows; i++){
                for(int j = 0; j < columns; j++){
                    matrix_values[i * columns + j] = x[i * columns + j];
                }
            }
        }
    }  //fill matrix with vector
    
    void printMatrix(){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++){
                cout << matrix_values[i * columns + j] << "   ";
            }
        }
    } //print matrix rows
    
    vector<double> getColumnV(int col){
        vector<double> col_vector;
        col_vector.resize(rows);
        for(int i = 0; i < rows; i++){
            col_vector[i] = matrix_values[i * columns + col];
        }
        return col_vector;
    }   //get a vector containing the column
    
    // takes a vector and makes the matrix column equal that vector
    void setColumnV(vector<double> vec, int col){
        for(int i = 0; i < rows; i++){
            matrix_values[i * columns + col] = vec[i];
        }
    }
    
    // takes a vector and makes the matrix row equal that vector
    void setRowV(vector<double> vec, int row){
        for(int i = 0; i < columns; i++){
            matrix_values[row * columns + i] = vec[i];
        }
    }
    
    // takes an integer and fills the matrix column with that integer
    void setColumnV(int num, int col){
        for(int i = 0; i < rows; i++){
            matrix_values[i * columns + col] = num;
        }
    }
};

Matrix matrixMultiply(Matrix A, Matrix B){
    int A_rows = A.getRows();
    int A_columns = A.getColumns();
    int B_rows = B.getRows();
    int B_columns = B.getColumns();
    Matrix C(A_rows, B_columns);
    double dotSum = 0;

    for(int i = 0; i < A_rows; i++){
        // get all dot products for one row of A
        for(int j = 0; j < B_columns; j++){
            // get 1 row column dot product sum
            for(int k = 0; k < A_columns; k++){
                dotSum += A.getValue(i, k) * B.getValue(k, j);
            }
            C.setValue(i, j, dotSum);
            dotSum = 0;
        }
    }
    return C;
}    //Matrix * Matrix

Matrix matrixTranspose(Matrix A){
    Matrix T(A.getColumns(),A.getRows());
    for(int i = 0; i < A.getColumns(); i++){
        T.setRowV(A.getColumnV(i), i);
    }
    return T;
}  //Matrix Transpose

Matrix matrixMultiply(Matrix A, double num){
    for(int i = 0; i < A.getRows(); i++){
        for(int j = 0; j < A.getColumns(); j++){
            A.setValue(i, j, (A.getValue(i, j) * num));
        }
    }
    return A;
}  //Matrix * number

Matrix matrixAdd(Matrix A, Matrix B){
    Matrix C(A.getRows(), A.getColumns());
    for(int i = 0; i < A.getRows(); i++){
        for(int j = 0; j < A.getColumns(); j++){
            C.setValue(i, j, (A.getValue(i, j) + B.getValue(i, j)));
        }
    }
    return C;
}  //Matrix addition

Matrix matrixSubtract(Matrix A, Matrix B){
    Matrix C(A.getRows(), A.getColumns());
    for(int i = 0; i < A.getRows(); i++){
        for(int j = 0; j < A.getColumns(); j++){
            C.setValue(i, j, (A.getValue(i, j) - B.getValue(i, j)));
        }
    }
    return C;
}   //Matrix subtraction

Matrix sigmoid(Matrix A){
    const double e = 2.71828182845904;
    vector<double> sig_values;
    sig_values.resize(A.getRows());
    Matrix C(A.getRows(),1);
    for(int i =0; i < A.getRows(); i++){
        double tempValue = 1/(1+pow(e, (A.getValue(i, 0) * -1)));
        sig_values.at(i) = tempValue;
    }
    
    C.setColumnV(sig_values, 0);
    return C;
}   //calc Sigmoid value

Matrix probabilities(Matrix A){
    const double e = 2.71828;
    Matrix P(A.getRows(), 1);
    double temp_value = 0;
    for(int i = 0; i < A.getRows(); i++){
        temp_value = pow(e, A.getValue(i, 0))/(1 + A.getValue(i, 0));
        P.setValue(i, 0, temp_value);
    }
    return P;
} //calc Probability

Matrix predictions(Matrix A){
    Matrix P(A.getRows(), 1);
    for(int i = 0; i < A.getRows(); i++){
        if(A.getValue(i, 0) > 0.5){
            P.setValue(i, 0, 1);
        }
        else {P.setValue(i, 0, 0);}
    }
    return P;
}   //calc Predictions

double accuracy(Matrix pred, Matrix test){
    double mean;
    double correct = 0;
    for(int i = 0; i < pred.getRows(); i++){
        if(pred.getValue(i, 0) == test.getValue(i, 0)) {
            correct++;
        }
    }
    return correct/pred.getRows();
}  //calc Accuracy

dataField train_df(dataField df, int numRows){
    dataField train;
    
    train.X.resize(numRows);
    train.pclass.resize(numRows);
    train.survived.resize(numRows);
    train.sex.resize(numRows);
    train.age.resize(numRows);
    train.num_instances = numRows;
    
    for(int i = 0; i < numRows; i++){
        train.X.at(i) = df.X.at(i);
        train.age.at(i) = df.age.at(i);
        train.pclass.at(i) = df.pclass.at(i);
        train.sex.at(i) = df.sex.at(i);
        train.survived.at(i) = df.survived.at(i);
    }
    return train;
}    //fill Train DF

dataField test_df(dataField df, int numRows){
    dataField test;
    int num_instances = df.num_instances;
    int count =0;
    
    test.X.resize(numRows);
    test.pclass.resize(numRows);
    test.survived.resize(numRows);
    test.sex.resize(numRows);
    test.age.resize(numRows);
    test.num_instances = numRows;
    
    for(int i = num_instances-numRows; i < num_instances; i++){
        test.X[count] = df.X[i];
        test.age[count] = df.age[i];
        test.pclass[count] = df.pclass[i];
        test.sex[count] = df.sex[i];
        test.survived[count] = df.survived[i];
        count++;
    }
    return test;
} //fill Test DF

double countTP(Matrix survived, Matrix predict){
    double TP = 0;
    for(int i = 0; i < survived.getRows(); i++){
        if(survived.getValue(i, 0) == predict.getValue(i, 0) && survived.getValue(i, 0) == 1)
            TP++;
    }
    return TP;
}  //calc TP

double countTN(Matrix survived, Matrix predict){
    double TN = 0;
    for(int i = 0; i < survived.getRows(); i++){
        if(survived.getValue(i, 0) == predict.getValue(i, 0) && survived.getValue(i, 0) == 0)
            TN++;
    }
    return TN;
}  //calc TN

void LogisticRegression(dataField train, dataField test){
    Matrix weight_sex(2, 1);
    weight_sex.setColumnV(1, 0);

    Matrix labels(train.num_instances, 1);
    labels.setColumnV(train.survived, 0);
    Matrix data_matrix_sex(train.num_instances, 2);
    
    data_matrix_sex.setColumnV(1, 0);
    data_matrix_sex.setColumnV(train.sex, 1);
    
    Matrix error_sex(train.num_instances, 1);
    Matrix prob_vector_sex(train.num_instances, 1);

    double learn_rate = 0.001;

    auto start = std::chrono::system_clock::now();
    //training model
    for(int i = 0; i < 10000; i++){
        prob_vector_sex = sigmoid(matrixMultiply(data_matrix_sex, weight_sex));
        error_sex = matrixSubtract(labels, prob_vector_sex);
        weight_sex = matrixAdd(weight_sex, matrixMultiply(matrixMultiply(matrixTranspose(data_matrix_sex), error_sex), learn_rate));
    }
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    //predict data
    Matrix test_matrix_sex(test.num_instances, 2);
    test_matrix_sex.setColumnV(1, 0);
    test_matrix_sex.setColumnV(test.sex, 1);
    
    //compare to test data
    Matrix test_labels(test.num_instances,1);
    test_labels.setColumnV(test.survived, 0);
    
    Matrix predicted_sex = matrixMultiply(test_matrix_sex, weight_sex);
    Matrix probabilitiy_sex = probabilities(predicted_sex);
    Matrix predictions_sex = predictions(probabilitiy_sex);
    double accuracy_sex = accuracy(predictions_sex, test_labels);
    
    //predict deaths
    double pred_false = 0;
    for(int i = 0; i < predictions_sex.getRows(); i++){
        if(predictions_sex.getValue(i, 0) == 0){
            pred_false++;
        }
    }
    //predict survives
    double pred_true = predictions_sex.getRows() - pred_false;
    
    double TP = countTP(predictions_sex, test_labels);
    double TN = countTN(predictions_sex, test_labels);
    double FP = pred_true - TP;
    double FN = pred_false - TN;
    
    cout << "|==================== Logistic Regression ====================|\n";
    cout << "|=== Coefficient of survivability using Sex as a Predictor ===|\n                 ";
    weight_sex.printMatrix();
    cout << "\n|==============Test Metrics using all Predictors==============|\n";
    cout << "                 Accuracy:   " << accuracy_sex << endl;
    cout << "              Sensitivity:   " << TP/(TP+FN) << endl;
    cout << "              Specificity:   " << TN/(TN+FP) << endl;
    cout << "            Training Time:   " << elapsed.count() << "ms" << endl;
}   //Linear Regression

int main() {
    ifstream inFile;
    string line;
    string X_in, survived_in, pclass_in, sex_in, age_in, trash;
    dataField titanic;
    titanic.X.resize(MAX_VALUE);
    titanic.pclass.resize(MAX_VALUE);
    titanic.survived.resize(MAX_VALUE);
    titanic.sex.resize(MAX_VALUE);
    titanic.age.resize(MAX_VALUE);
    string filename = "titanic_project.csv";
    
    inFile.open(filename);
    if(!inFile.is_open()){cout<<"Could not open input file "<<filename<<endl;}
    
    getline(inFile, line);
    
    int numObservations = 0;
    
    while(inFile.good()){
        inFile.get();
        getline(inFile, X_in, '"');
        inFile.get();
        getline(inFile, pclass_in, ',');
        getline(inFile, survived_in, ',');
        getline(inFile, sex_in, ',');
        getline(inFile, age_in, '\n');
        //read data and split it
        //store them
        titanic.X.at(numObservations) = stof(X_in);
        titanic.pclass.at(numObservations) = stof(pclass_in);
        titanic.survived.at(numObservations) = stof(survived_in);
        titanic.sex.at(numObservations) = stof(sex_in);
        titanic.age.at(numObservations) = stof(age_in);

        numObservations++;
    }
    
    titanic.X.resize(numObservations);
    titanic.pclass.resize(numObservations);
    titanic.survived.resize(numObservations);
    titanic.sex.resize(numObservations);
    titanic.age.resize(numObservations);

    titanic.num_instances = int(titanic.X.size());
    inFile.close();
    
    //split to train and test, run logistic regression model
    dataField train = train_df(titanic, 800);
    dataField test = test_df(titanic, titanic.num_instances - 800);
    LogisticRegression(train, test);
}   //main