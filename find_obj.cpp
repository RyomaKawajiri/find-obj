#include <opencv2/core/core.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;

class finder{
public:
  finder(//search params
	 int _checks,
	 //surf params
	 double _hessianThreshold, 
	 int _nOctaves=4,
         int _nOctaveLayers=2, 
	 bool _extended=false,
	 //flann params
	 float _target_precision = 0.9,
	 float _build_weight = 0.01,
	 float _memory_weight = 0,
	 float _sample_fraction = 0.1):
    surf(_hessianThreshold, _nOctaves, _nOctaveLayers, _extended),
    flann_params(_target_precision, _build_weight, _memory_weight, _sample_fraction),
    search_params(_checks)
  {
  };

  ~finder(){
  };

  void set(Mat& image){
    obj obj;
    obj.image = image;
    //vector<KeyPoint> key;
    vector<float> desc;
    surf(image, Mat(), obj.key, desc);
    obj.feats = Mat::Mat(desc, true).
      reshape(1, desc.size() / surf.descriptorSize());

    obj.indices = Mat::Mat(obj.feats.rows, 2, CV_32S);
    obj.dists = Mat::Mat(obj.feats.rows, 2, CV_32F);
    cerr << "number of keypoints: " << obj.feats.rows << endl;

    objs.push_back(obj);
  }

  vector<int> operator ()(const Mat& image, Mat& output)
  {
    //output image
    int num_obj = objs.size();
#ifdef DEBUG
    int objs_width = 0, objs_height = 0;
    {
      for(int i=0; i<num_obj; ++i){
	objs_width += objs[i].image.cols;
	if(objs[i].image.rows > objs_height)
	  objs_height = objs[i].image.rows;
      }
      output = Mat::Mat(image.rows + objs_height,
			image.cols > objs_width ? image.cols : objs_width,
			CV_8UC3);
      int start_x =0;
      for(int i=0; i<num_obj; ++i){
	Mat temp = Mat::Mat(output, Rect(start_x, 0, 
					 objs[i].image.cols, 
					 objs[i].image.rows));
	cvtColor(objs[i].image, temp, CV_GRAY2BGR);
	start_x += objs[i].image.cols;
      }
      Mat temp = Mat::Mat(output, Rect(0, objs_height, image.cols, image.rows));
      cvtColor(image, temp, CV_GRAY2BGR);
    }
#endif

    //surf extractor
    vector<KeyPoint> key;
    vector<float> desc;
    surf(image, Mat(), key, desc);
    if(desc.empty()) return vector<int>();
    Mat scene_feats = Mat::Mat(desc, true).
      reshape(1, desc.size() / surf.descriptorSize());

    //flann search
    flann::Index_<float> flann(scene_feats, flann::KDTreeIndexParams(3));
    //flann::Index_<float> flann(scene_feats, flann_params);
    for(int i=0; i<num_obj; ++i){
      flann.knnSearch(objs[i].feats, objs[i].indices, objs[i].dists, 
		      2, search_params);
    }

    //voting and drawing lines
    vector<vector<int> > box(objs.size(), vector<int>(scene_feats.rows));
    int start_x = 0;
    for(int j=0; j<num_obj; ++j){
      int* indices_ptr = objs[j].indices.ptr<int>(0);
      float* dists_ptr = objs[j].dists.ptr<float>(0);
      for(int i=0; i<objs[j].indices.rows; ++i)
	if(dists_ptr[2*i] < 0.3*dists_ptr[2*i+1]){
	  //box[indices_ptr[2*i]] ++;
	  box[j][indices_ptr[2*i]] ++;
#ifdef DEBUG
	  Point obj_pt = Point( start_x + objs[j].key[i].pt.x, 
				objs[j].key[i].pt.y);
	  Point scene_pt = Point( key[indices_ptr[2*i]].pt.x,
				  key[indices_ptr[2*i]].pt.y + objs_height);
	  
	  circle(output, obj_pt, 3, CV_RGB(255, 0, 0));
	  circle(output, scene_pt, 3, CV_RGB(0, 255, 0));
	  line(output, obj_pt, scene_pt, CV_RGB(255, 255, 255));
#endif
	}
      start_x += objs[j].image.cols;
    }

    vector<int> ret(objs.size());
    for(int i=0; i<num_obj; ++i)
      for(int j=0; j<scene_feats.rows; j++)
	if(box[i][j] == 1)
	  ret[i] ++;

    return ret;
  }

  vector<int> check(Mat& image, Mat& output){
    return (*this)(image, output);
  }

private:
  struct obj{
    Mat image;
    Mat feats;
    vector<KeyPoint> key;
    Mat indices;
    Mat dists;
  };
  vector<obj> objs;
  SURF surf;
  flann::AutotunedIndexParams flann_params;
  flann::SearchParams search_params;
  //flann::Index_<float> *flann;
};

int main(int argc, char **argv){
  if( argc < 2 ){
    cerr << "no param" << endl;
    return 1;
  }

  //constructor
  cerr << "preparing" << endl;
  finder fnd(64, 100);

  //object set
  for(int i=1; i<argc; i++){
    Mat image = imread(argv[i]), obj;
    cvtColor(image, obj, CV_BGR2GRAY);
    fnd.set(obj);
  }

  //camera set
  VideoCapture cap(0);
  Mat frame, scene, output, resized;  
  
  namedWindow( "multi find obj", 
	       CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED );
  //loop
  for(;;){
    cap >> frame;
    cvtColor(frame, scene, CV_BGR2GRAY);
    resize(scene, resized, Size(640, 480));

    vector<int> ret = fnd(resized, output);
    stringstream ss;
    //ss << "number of point: ";
    int n = ret.size();
    for(int i=0; i<n; i++){
      if(ret[i] > 2){
	ss << argv[i+1] << " " ;
      }
    }
#ifdef DEBUG
    imshow("multi find obj", output);
#else
    imshow("multi find obj", resized);
#endif    

    displayOverlay("multi find obj", ss.str(), 60);

    char c = (char)waitKey(1);
    if(c > 0) break;
  }

  return 0;
}
