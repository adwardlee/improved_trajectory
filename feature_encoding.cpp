#include "feature_encoding.h"

FisherVector::FisherVector()
{
  data_type = VL_TYPE_FLOAT;
 // distance_type = VlDistanceL2;
}

void FisherVector::GmmCluster(cv::Mat &data, int num_visualwords)
{
  this->num_clusters = num_visualwords;
  this->dimension = data.cols;
  this->fv_dimension = 2 * dimension * num_clusters;
  this->gmm = vl_gmm_new(data_type, dimension, num_clusters);
  
  //  [5/20/2015 Administrator]
  //this->num_centers = num_visualwords;
  //this->kmeans = vl_kmeans_new(data_type, distance_type);
  //vl_kmeans_cluster(kmeans, data.data, dimension, data.rows, num_centers);
  //vl_gmm_init_with_kmeans(gmm,data.data,data.rows,kmeans);
  //  [5/20/2015 Administrator]
  
  // to get means, covariances and priors of the estimated mixture
  vl_gmm_cluster(gmm, data.data, data.rows);
  this->means = (float*)vl_gmm_get_means(gmm);
  this->covariances = (float*)vl_gmm_get_covariances(gmm);
  this->priors = (float*)vl_gmm_get_priors(gmm);
}

// create a new FisherVector instance
// with a gmm loading external params
//----------------------------------------------------------------------------------
FisherVector::FisherVector(const char* file_dir)
{
  std::ifstream ifs(file_dir, std::ios_base::binary);
  if (!ifs)
  {
    std::cerr<<"Failed to open the file";
    abort();
  }

  ifs.read((char*)&data_type, sizeof(vl_type));
  ifs.read((char*)&num_clusters, sizeof(vl_size));
  ifs.read((char*)&dimension, sizeof(vl_size));

  this->fv_dimension = 2 * dimension * num_clusters;

  this->means = (float*)vl_malloc(sizeof(float) * num_clusters * dimension);
  this->covariances = (float*)vl_malloc(sizeof(float) * num_clusters * dimension);
  this->priors = (float*)vl_malloc(sizeof(float) * num_clusters);

  for (unsigned int i = 0; i < num_clusters; i++)
  {
    for (unsigned int j = 0; j < dimension; j++)
    {
      ifs.read((char*)&means[i * dimension + j], sizeof(float));
      ifs.read((char*)&covariances[i * dimension + j], sizeof(float));
    }
    ifs.read((char*)&priors[i], sizeof(float));
  }

  ifs.close();

  // intialize the gmm object and set the params
  this->gmm = vl_gmm_new(data_type, dimension, num_clusters);
  vl_gmm_set_covariances(gmm, covariances);
  vl_gmm_set_priors(gmm, priors);
  vl_gmm_set_means(gmm, means);
}

// fisher vector encoding
//----------------------------------------------------------------------------------
cv::Mat FisherVector::FvEncode(cv::Mat &data)
{
  cv::Mat encoded_vec = cv::Mat_<float>(1, (int)fv_dimension);

  /**
   * VL_FISHER_FLAG_IMPROVED
   *  - Square root
   *  - L2 normalization
   */
  vl_fisher_encode(encoded_vec.data, data_type, means, dimension, num_clusters, covariances, priors, data.data,
                   data.rows,
                   VL_FISHER_FLAG_IMPROVED);

  //for (int i = 0 ; i<data.cols; i = i+2)
  //{
	 // if (encoded_vec.at<float>(0,i)||encoded_vec.at<float>(0,i))
	 // {
		//  encoded_vec.at<float>(0,i) = encoded_vec.at<float>(0,i)/sqrt(pow(encoded_vec.at<float>(0,i),2)+pow(encoded_vec.at<float>(0,i+1),2));
		//  encoded_vec.at<float>(0,i+1) = encoded_vec.at<float>(0,i+1)/sqrt(pow(encoded_vec.at<float>(0,i),2)+pow(encoded_vec.at<float>(0,i+1),2));
	 // }

  //}

  //encoded_vec = encoded_vec/cv::norm(encoded_vec,cv::NORM_L2);

  return encoded_vec;
}

// save the gmm parameters into external file
//----------------------------------------------------------------------------------
void FisherVector::SaveGMM(const char* file_dir)
{
  std::ofstream ofs(file_dir, std::ios_base::out | std::ios_base::binary);
  if (!ofs)
  {
    std::cerr<<"Failed to open the file";
    abort();
  }

  ofs.write((char*)&data_type, sizeof(vl_type));
  ofs.write((char*)&num_clusters, sizeof(vl_size));
  ofs.write((char*)&dimension, sizeof(vl_size));

  for (unsigned int i = 0; i < num_clusters; i++)
  {
    for (unsigned int j = 0; j < dimension; j++)
    {
      ofs.write((char*)&means[i * dimension + j], sizeof(float));
      ofs.write((char*)&covariances[i * dimension + j], sizeof(float));
    }
    ofs.write((char*)&priors[i], sizeof(float));
  }

  ofs.close();
}

// destructor
//----------------------------------------------------------------------------------
FisherVector::~FisherVector()
{
  vl_gmm_delete(this->gmm);

//  vl_kmeans_delete(this->kmeans);
}

//----------------------------------------------------------------------------------
// VLAD encoding
//----------------------------------------------------------------------------------

// create a new VLAD instance
// with a k-means that clustering given data
//----------------------------------------------------------------------------------
VLAD::VLAD()
{
  this->data_type = VL_TYPE_FLOAT;
  this->distance_type = VlDistanceL2;
}

void VLAD::KmeansCluster(cv::Mat& data, int num_visualwords)
{
  this->kmeans = vl_kmeans_new(data_type, distance_type);
  this->num_centers = num_visualwords;
  this->dimension = data.cols;
  this->vlad_dimension = dimension * num_centers;

  vl_kmeans_cluster(kmeans, data.data, dimension, data.rows, num_centers);

  this->means = (float*)vl_kmeans_get_centers(kmeans);
}

// create a new VLAD instance
// with a k-means loading external params
//----------------------------------------------------------------------------------
VLAD::VLAD(const char* file_dir)
{
  std::ifstream ifs(file_dir, std::ios_base::binary);
  if (!ifs)
  {
    std::cerr<<"Failed to open the file";
    abort();
  }

  ifs.read((char*)&data_type, sizeof(vl_type));
  ifs.read((char*)&distance_type, sizeof(VlVectorComparisonType));
  ifs.read((char*)&dimension, sizeof(vl_size));
  ifs.read((char*)&num_centers, sizeof(vl_size));

  this->vlad_dimension = dimension * num_centers;

  this->means = (float*)vl_malloc(sizeof(float) * num_centers * dimension);
  for (unsigned int i = 0; i < dimension * num_centers; i++)
  {
    ifs.read((char*)&means[i], sizeof(float));
  }

  ifs.close();

  // intialize the kmeans object and set the params
  this->kmeans = vl_kmeans_new(data_type, distance_type);
  vl_kmeans_set_centers(kmeans, means, dimension, num_centers);
}

// vlad encoding
//----------------------------------------------------------------------------------
cv::Mat VLAD::VladEncode(cv::Mat& data)
{
  // stores the indice of k-means center that given data is assigned
  cv::Mat indexes = cv::Mat_<unsigned int>(1, data.rows);
  vl_kmeans_quantize(kmeans, (vl_uint32*)indexes.data, NULL, data.data, data.rows);

  // hard assignment
  cv::Mat assignments = cv::Mat::zeros(cv::Size(data.rows * num_centers, 1), CV_32F);
  for (int i = 0; i < data.rows; i++)
    assignments.at<float>(0, i * num_centers + indexes.at<int>(0, i)) = 1.;

  /**
   * VL_VLAD_FLAG_SQUARE_ROOT
   * VL_VLAD_FLAG_NORMALIZE_COMPONENTS
   */
  cv::Mat encoded_vec = cv::Mat_<float>(1, (int)vlad_dimension);
  vl_vlad_encode(encoded_vec.data, data_type, means, dimension, num_centers, data.data, data.rows, assignments.data,
  VL_VLAD_FLAG_SQUARE_ROOT | VL_VLAD_FLAG_NORMALIZE_MASS);

  return encoded_vec;
}

// save the kmeans parameters into external file
//----------------------------------------------------------------------------------
void VLAD::SaveKmeans(const char* file_dir)
{
  std::ofstream ofs(file_dir, std::ios_base::out | std::ios_base::binary);
  if (!ofs)
  {
    std::cerr<<"Failed to open the file";
    abort();
  }

  ofs.write((char*)&data_type, sizeof(vl_type));
  ofs.write((char*)&distance_type, sizeof(VlVectorComparisonType));
  ofs.write((char*)&dimension, sizeof(vl_size));
  ofs.write((char*)&num_centers, sizeof(vl_size));

  for (unsigned int i = 0; i < dimension * num_centers; i++)
  {
    ofs.write((char*)&means[i], sizeof(float));
  }

  ofs.close();
}

// destructor
//----------------------------------------------------------------------------------
VLAD::~VLAD()
{
  vl_kmeans_delete(this->kmeans);
}

//----------------------------------------------------------------------------------
// Bag of Visual Words
//----------------------------------------------------------------------------------

// create a new BoVW instance
// with a k-means that clustering given data
//----------------------------------------------------------------------------------
BoVW::BoVW()
{
  this->data_type = VL_TYPE_FLOAT;
  this->distance_type = VlDistanceL2;
}

void BoVW::KmeansCluster(cv::Mat& data, int num_visualwords)
{
  this->kmeans = vl_kmeans_new(data_type, distance_type);
  this->num_centers = num_visualwords;
  this->dimension = data.cols;
  this->bovw_dimension = num_centers;

  // k-means
  vl_kmeans_cluster(kmeans, data.data, dimension, data.rows, num_centers);

  this->means = (float*)vl_kmeans_get_centers(kmeans);
}

// create a new BoVW instance
// with a k-means loading external params
//----------------------------------------------------------------------------------
BoVW::BoVW(const char* file_dir)
{
  std::ifstream ifs(file_dir, std::ios_base::binary);
  if (!ifs)
    std::cout << "Failed to open the file" << std::endl;

  ifs.read((char*)&data_type, sizeof(vl_type));
  ifs.read((char*)&distance_type, sizeof(VlVectorComparisonType));
  ifs.read((char*)&dimension, sizeof(vl_size));
  ifs.read((char*)&num_centers, sizeof(vl_size));

  this->bovw_dimension = num_centers;

  this->means = (float*)vl_malloc(sizeof(float) * num_centers * dimension);
  for (unsigned int i = 0; i < dimension * num_centers; i++)
  {
    ifs.read((char*)&means[i], sizeof(float));
  }

  ifs.close();

  // intialize the kmeans object and set the params
  kmeans = vl_kmeans_new(data_type, distance_type);
  vl_kmeans_set_centers(kmeans, means, dimension, num_centers);
}

// build histogram
//----------------------------------------------------------------------------------
cv::Mat BoVW::BuidHistogram(cv::Mat& data)
{
  cv::Mat indexes = cv::Mat_<int>(1, data.rows);

  vl_kmeans_quantize(kmeans, (vl_uint32*)indexes.data, NULL, data.data, data.rows);

  float l2_sum = 0.;

  float tmp;

  cv::Mat builtHist = cv::Mat::zeros(1, (int)bovw_dimension, CV_32F);

  for (int i = 0; i < data.rows; i++)
  {
    tmp = ++(builtHist.at<float>(0, indexes.at<unsigned int>(0, i)));
    l2_sum += tmp * tmp;
  }

  l2_sum = vl_sqrt_f(l2_sum);
  l2_sum = VL_MAX(l2_sum, 1e-12);

  for (unsigned int i = 0; i < bovw_dimension; i++)
  {
    builtHist.at<float>(0, i) /= l2_sum;
  }

  return builtHist;
}

// save the kmeans parameters into external file
//----------------------------------------------------------------------------------
void BoVW::SaveKmeans(const char* file_dir)
{
  std::ofstream ofs(file_dir, std::ios_base::out | std::ios_base::binary);
  if (!ofs)
  {
    std::perror("Failed to open the file");
    abort();
  }

  ofs.write((char*)&data_type, sizeof(vl_type));
  ofs.write((char*)&distance_type, sizeof(VlVectorComparisonType));
  ofs.write((char*)&dimension, sizeof(vl_size));
  ofs.write((char*)&num_centers, sizeof(vl_size));

  for (unsigned int i = 0; i < dimension * num_centers; i++)
  {
    ofs.write((char*)&means[i], sizeof(float));
  }

  ofs.close();
}

// destructor
//----------------------------------------------------------------------------------
BoVW::~BoVW()
{
  vl_kmeans_delete(this->kmeans);
}