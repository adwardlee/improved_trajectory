#pragma once
#pragma comment(lib,"vl.lib")
extern "C" {
#include <vl/generic.h>
#include <vl/fisher.h>
#include <vl/gmm.h>
#include <vl/vlad.h>
#include <vl/mathop.h>
#include <vl/kmeans.h>

}

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

// normalization option for Bag of Visual Words
#define L1_NORM 0
#define L2_NORM 1

//----------------------------------------------------------------------------------
// Fisher Vector encoding
//----------------------------------------------------------------------------------
class FisherVector {
private:
	vl_size fv_dimension; // the dimension of fisher vecrtor
	VlGMM* gmm;           // gaussian mixture model object
	float* means;         // gmm means
	float* covariances;   // gmm covariances
	float* priors;        // gmm priors
	vl_size num_clusters; // the number of gaussians
	vl_type data_type;    // type of data
	vl_size dimension;    // data dimension

//	VlKMeans* kmeans;       // kmeans object
//	VlVectorComparisonType distance_type;
//	vl_size num_centers; // the number of centers
public:


	FisherVector();
	FisherVector(const char* file_dir);
	~FisherVector();
	void GmmCluster(cv::Mat& data, int num_visualwords);
	cv::Mat FvEncode(cv::Mat& dataToEncode);

	void SaveGMM(const char* file_dir);
	vl_size GetDimension() { return this->fv_dimension; }
	float* GetMeans() { return this->means; }
	float* GetCovarances() { return this->covariances; }
	float* GetPriors() { return this->priors; }
};

//----------------------------------------------------------------------------------
// VLAD encoding
//----------------------------------------------------------------------------------
class VLAD {
private:
	vl_size vlad_dimension; // the dimension of vlad
	//VlKMeans* kmeans;       // kmeans object
	float* means;           // kmeans centers
	VlVectorComparisonType distance_type;
	vl_size num_centers; // the number of centers
	vl_type data_type;   // type of data
	vl_size dimension;   // data dimension
public:
	VlKMeans* kmeans;       // kmeans object
	VLAD();
	VLAD(const char* file_dir);
	~VLAD();
	void KmeansCluster(cv::Mat& data, int num_visualwords);
	cv::Mat VladEncode(cv::Mat& dataToEncode);

	void SaveKmeans(const char* file_dir);
	vl_size GetDimension() { return this->vlad_dimension; }
	float* GetMeans() { return this->means; }
};

//----------------------------------------------------------------------------------
// Bag of Visual Words
//----------------------------------------------------------------------------------
class BoVW {
private:
	vl_size bovw_dimension; // the dimension of bag of visual words
	VlKMeans* kmeans;       // kmeans object
	float* means;           // kmeans centers
	VlVectorComparisonType distance_type;
	vl_size num_centers; // the number of centers
	vl_type data_type;   // type of data
	vl_size dimension;   // data dimension
public:
	BoVW();
	BoVW(const char* file_dir);
	~BoVW();
	void KmeansCluster(cv::Mat& data, int  num_visualwords);
	cv::Mat BuidHistogram(cv::Mat& dataToBuild);

	void SaveKmeans(const char* file_dir);
	vl_size GetDimension() { return this->bovw_dimension; }
	float* GetMeans() { return this->means; }
};

//#endif /* _VECTOR_ENCODING_H */