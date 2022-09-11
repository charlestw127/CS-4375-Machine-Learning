//Charles Wallis, University of Texas at Dallas
//Data Exploration 
//September 11, 2022

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

int readFile(string filename, vector<double> &rm, vector<double> &medv){
    //initialize
    string line;
    string rmRead;
    string medvRead;
    ifstream inFile(filename);
    int numObservations = 0;

    //validate file open
    if(!inFile.is_open()){cout<<"Could not open input file "<<filename<<endl;}

    //read first line, Header
    getline(inFile, line);
    cout << " Header: " << line << endl;

    //read lines
    while(getline(inFile, line)){
        stringstream ss(line);
        
        //split the two values
        getline(ss, rmRead, ',');
        getline(ss, medvRead);
        
        //enter values in vector
        rm.push_back(stof(rmRead));
        medv.push_back(stof(medvRead));
        numObservations++;      //increment number of records
    }//split rm and medv read data in two vectors
    
    //close file
    inFile.close();
    
    return numObservations;
}//read file, input rm and medv


double calcSum(vector<double> v){
    double sum = 0;
    for (int i = 0; i < v.size(); i++){
        sum += v[i];
    }
    return sum;
}//add up to sum


double calcMean(vector<double> v){
    double sum = calcSum(v);
    double size = v.size();
    double mean = sum / size;
    return mean;
}//calculate average


double calcMedian(vector<double> v){
    int size = v.size();
    sort(v.begin(), v.end()); //sort the numbers to find middle

    //size is odd, median is middle
    if(size % 2 == 1){return v[size/2];}
    else{//size is even, median is avg of middle 2
        double middleLeft,middleRight;
        middleLeft = v[size/2-1]; 
        middleRight = v[size/2];
        return (middleLeft+middleRight)/2;
    }//median if even
}//calculate median


double calcMax(vector<double> v){
    double max = 0;
    //loop vector
    for(int i=0; i < v.size(); i++){
        //find max
        if(v[i] > max){max = v[i];}
    }//set max
    return max;
}//calculate range

double calcMin(vector<double> v){
    double min = 10000000;
    //loop vector
    for(int i=0; i < v.size(); i++){
        //find min
        if(v[i] < min){min = v[i];}
    }//set min
    return min;
}//calculate min


double calcCovariance(vector<double> rm, vector<double> medv){
    double rmMean = calcMean(rm);
    double medvMean = calcMean(medv);

    //numerator
    double numer = 0;
    for(int i=0; i < rm.size(); i++){
        numer += (rm[i] - rmMean) * (medv[i] - medvMean);
    }
    //denominator
    double denom = rm.size()-1;
    //covariance
    double covariance = numer / denom;
    return covariance;
}//calculate covariance for the vectors


double calcCorrelation(vector<double> rm, vector<double> medv){
    //find rm sigma
    double rmVariance = calcCovariance(rm, rm);
    double rmSig = sqrt(rmVariance);
    //find medv sigma
    double medvVariance = calcCovariance(medv, medv);
    double medvSig = sqrt(medvVariance);

    //find covariance for rm and medv
    double covar = calcCovariance(rm, medv);
    
    double correlation = covar / (rmSig * medvSig);
    return correlation;
}//calculate correlation for the vectors


int main() {
    //initialize rm, medv, and set filename
    vector<double> rm;
    vector<double> medv;
    string filename = "Boston.csv";

    //store rm and medv data from the file into vectors, while counting num of records
    int numObservations = readFile(filename, rm, medv);
    
    cout << " Number of Records: " << numObservations << endl;
    //stats for rm
    cout << "\n| === RM Stats === |\n";
    cout << "    Sum: " << calcSum(rm) << endl;
    cout << "   Mean: " << calcMean(rm) << endl;
    cout << " Median: " << calcMedian(rm) << endl;
    cout << "  Range: [" << calcMin(rm) << " - " << calcMax(rm) << "]" << endl;
    
    //stats for medv
    cout << "\n| == MEDV Stats == |\n";
    cout << "    Sum: " << calcSum(medv) << endl;
    cout << "   Mean: " << calcMean(medv) << endl;
    cout << " Median: " << calcMedian(medv) << endl;
    cout << "  Range: [" << calcMin(medv) << " - " << calcMax(medv) << "]" << endl;
    
    //covariance and correlation
    cout << "\n| === RM and MEDV === |\n";
    cout << "  Covariance: " << calcCovariance(rm, medv) << endl;
    cout << " Correlation: " << calcCorrelation(rm, medv) << endl;
}   //main driver
