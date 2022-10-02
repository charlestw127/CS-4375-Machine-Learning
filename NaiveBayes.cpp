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

double meanAgeByClass(Matrix train, int survived){
    double sum = 0;
    double count = 0;
    for(int i = 0; i < train.getRows(); i++){
        if(train.getValue(i, 0) == survived){
            sum+= train.getValue(i, 3);
            count++;
        }
    }
    return sum/count;
}   //avg age by class

double ageVarianceByClass(Matrix train, int survived, double mean){
    double sum = 0;
    double count = 0;
    double temp;
    for(int i = 0; i < train.getRows(); i++){
        temp = train.getValue(i, 0);
        if(temp == survived){
            sum+= pow(temp-mean,2);
            count++;
        }
    }
    return (1/(count - 1)) * sum;
}   //variance of age by class

double SurvivabilityAge(double age, double AvgAge, double AgeVariance){
    double pi = 3.14159;
    double e = 2.71828;
    
    double first = 1/sqrt(2*pi*AgeVariance);
    double second = pow(e, -(pow(age-AvgAge, 2)/(2*AgeVariance)));
    
    return first * second;
} //survivability by age

Matrix modelProbability(double pclass, double sex, double age, Matrix apriori, Matrix lh_pclass, Matrix lh_sex, Matrix AvgAge, Matrix AgeVariance){
    Matrix raw(2,1);
    double num_s;
    double num_p;
    double denominator;
    
    num_s = lh_pclass.getValue(1, pclass - 1) * lh_sex.getValue(1, sex) * 
    apriori.getValue(1, 0)*SurvivabilityAge(age, AvgAge.getValue(1, 0), AgeVariance.getValue(1, 0));
    num_p = lh_pclass.getValue(0, pclass - 1) * lh_sex.getValue(0, sex) *
    apriori.getValue(0, 0)*SurvivabilityAge(age, AvgAge.getValue(0, 0), AgeVariance.getValue(0, 0));
    denominator = num_s + num_p; 
    raw.setValue(1, 0, num_s/denominator);
    raw.setValue(0, 0, num_p/denominator);
    
    return raw;
}   //model probability

void naiveBayes(dataField train, dataField test){
    //calc probability using train/test sets
    Matrix data_train(train.num_instances, 4);
    Matrix data_test(test.num_instances, 4);
    
    data_train.setColumnV(train.survived, 0);
    data_train.setColumnV(train.pclass, 1);
    data_train.setColumnV(train.sex, 2);
    data_train.setColumnV(train.age, 3);
    data_test.setColumnV(test.survived, 0);
    data_test.setColumnV(test.pclass, 1);
    data_test.setColumnV(test.sex, 2);
    data_test.setColumnV(test.age, 3);
    
    auto start = std::chrono::system_clock::now();

    //Apriori data
    Matrix apriori(2,1);
    Matrix count_survived(2,1);
    double survivors = 0;
    double deaths = 0;
    for(int i = 0; i < data_train.getRows(); i++){
        if(data_train.getValue(i, 0) == 1){
            survivors++;
        }
        if(data_train.getValue(i, 0) == 0){
            deaths++;
        }
    }
    
    //set matrix with apriori
    apriori.setValue(1, 0, survivors/data_train.getRows());
    apriori.setValue(0, 0, deaths/data_train.getRows());
    count_survived.setValue(1, 0, survivors);
    count_survived.setValue(0, 0, deaths);
    
    
    cout << "|============= Naïve Bayes =============|\n";
    cout << " Apriori: "  << "   Survived: " << survivors <<  "   Died: " << deaths <<  endl;
    Matrix lh_pclass(2, 3);
    
    //survivability by class
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 3; j++){
            double value = 0;
            double num_rows = 0;
            
            for(int k = 0; k < data_train.getRows(); k++){
                if(data_train.getValue(k, 1) == j+1 && data_train.getValue(k, 0) == i){
                    num_rows++;
                }
            }
            value = num_rows/count_survived.getValue(i, 0);
            lh_pclass.setValue(i, j, value);
        }   //for each class
    }   //loop through different class
    
    
    //survivability by sex
    Matrix lh_sex(2,2);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            double value = 0;
            double num_rows = 0;
            
            for(int k = 0; k < data_train.getRows(); k++){
                if(data_train.getValue(k, 2) == j && data_train.getValue(k, 0) == i)
                    num_rows++;
            }
            value = num_rows/count_survived.getValue(i, 0);
            lh_sex.setValue(i, j, value);
        }
    }   //for each sex
    
    Matrix AvgAge(2, 1); //matrix for 
    Matrix AgeVariance(2, 1);
    
    for(int i = 0; i < 2; i++){
        AvgAge.setValue(i, 0, meanAgeByClass(data_train, i));
        AgeVariance.setValue(i, 0, ageVarianceByClass(data_train, i, AvgAge.getValue(i, 0)));
    }
    
    Matrix probability(data_test.getRows(), 1);
    
    for(int i = 0; i < data_test.getRows(); i++){
        Matrix raw = modelProbability(data_test.getValue(i, 1), data_test.getValue(i,2), data_test.getValue(i, 3), apriori, lh_pclass, lh_sex, AvgAge, AgeVariance);
        probability.setValue(i, 0, raw.getValue(1, 0));
    }
    
    //predict on probabilities
    Matrix pred = predictions(probability);
    
    //finish training timer
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    Matrix test_labels(data_test.getRows(), 1);
    test_labels.setColumnV(data_test.getColumnV(0), 0);
    
    double falsePredictions = 0;
    for(int i = 0; i < pred.getRows(); i++){
        if(pred.getValue(i, 0) == 0){
            falsePredictions++;
        }
    }
    double truePredictions = pred.getRows() - falsePredictions;
    
    double TP = countTP(pred, test_labels);
    double TN = countTN(pred, test_labels);
    double FP = truePredictions - TP;
    double FN = falsePredictions - TN;
    
    cout << "\nSurvivability for each class\n";
    lh_pclass.printMatrix();
    
    cout << "\nSurvivability for each Sex\n";
    lh_sex.printMatrix();
    
    cout << "\n\n|===Test Metrics using all Predictors===|\n";
    cout << "       Accuracy:   " << accuracy(pred, test_labels) << endl;
    cout << "    Sensitivity:   " << TP/(TP+FN) << endl;
    cout << "    Specificity:   " << TN/(TN+FP) << endl;
    cout << "  Training Time:   " << elapsed.count() << "ns" << endl;
}    //Naive Bayes

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
    
    //split to train and test, run naive bayes model
    dataField train = train_df(titanic, 800);
    dataField test = test_df(titanic, titanic.num_instances - 800);
    naiveBayes(train, test);
}   //main