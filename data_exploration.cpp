//Charles Wallis, University of Texas at Dallas
//Data Exploration 
//September 10, 2022

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>

using namespace std;

int readFile(string filename, vector<double> &rm, vector<double> &medv){
    string line;
    string rmRead;
    string medvRead;
    ifstream inFile(filename);
    int numObservations = 0;

    //validate file open
    if(!inFile.is_open()){cout<<"Could not open input file "<<filename<<endl;}

    //read first line
    getline(inFile, line);

    //read lines
    while(getline(inFile, line)){
        stringstream ss(line);
        getline(ss, rmRead, ',');
        getline(ss, medvRead);
        rm.push_back(stof(rmRead));
        medv.push_back(stof(medvRead));
        numObservations++;
    }//split rm and medv read data in two vectors
    
    return numObservations;
    
    //close file
    inFile.close();
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


double calcRange(vector<double> v){
    double max = 0;
    double min = 10000000;
    //loop vector
    for(int i=0; i < v.size(); i++){
        //find max
        if(v[i] > max){max = v[i];}
        //find min
        if(v[i] < min){min = v[i];}
    }//set min and max
    return max - min;
}//calculate range


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
    string filename = "Boston.txt";

    //store rm and medv data from the file into vectors, while counting num of records
    int numObservations = readFile(filename, rm, medv);

    cout << "Number of Records: " << numObservations << endl;
    //stats for rm
    cout << "|=====RM Stats=====|\n";
    cout << "|Sum    : " << calcSum(rm) << endl;
    cout << "|Mean   : " << calcMean(rm) << endl;
    cout << "|Median : " << calcMedian(rm) << endl;
    cout << "|Range  : " << calcRange(rm) << endl;
    
    //stats for medv
    cout << "|====MEDV Stats====|\n";
    cout << "|Sum    : " << calcSum(medv) << endl;
    cout << "|Mean   : " << calcMean(medv) << endl;
    cout << "|Median : " << calcMedian(medv) << endl;
    cout << "|Range  : " << calcRange(medv) << endl;

    //covariance and correlation
    cout << "|=====RM and MEDV====|\n";
    cout << "|Covariance  : " << calcCovariance(rm, medv) << endl;
    cout << "|Correlation : " << calcCorrelation(rm, medv) << endl;
}