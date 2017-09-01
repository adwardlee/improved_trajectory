#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"
//#include "feature_encoding.h"

#include <time.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

int main(int argc, char** argv)
{
	struct timeval starttime, onevideo_starttime, onevideo_endtime, totaltime;
	gettimeofday(&starttime, NULL);

	char *name[] = {"brush_hair","cartwheel","catch", "chew", "clap", "climb", "climb_stairs", "dive", "draw_sword", "dribble", "drink","eat","fall_floor","fencing","flic_flac",
			"golf", "handstand", "hit", "hug","jump","kick","kick_ball","kiss","laugh","pick","pour","pullup","punch","push","pushup","ride_bike","ride_horse","run","shake_hands",
				"shoot_ball","shoot_bow","shoot_gun","sit","situp","smile","smoke","somersault","stand","swing_baseball","sword","sword_exercise","talk","throw","turn","walk","wave"};
	vector<string> constnames(name, name + classnum);
	vector<string> txtfilename(0);
	std::map<int,std::vector<string> > videonames;
	
	string videoname_prefix("/home/lijun/action_dataset/HMDB51/");
	string save_videoname_prefix("/home/lijun/tvcj/hmdb51_features/");
	for (int i = 0; i<classnum; i++)
	{	
		getFiles(videoname_prefix + constnames[i],"avi",videonames[i]);
	}
	
	for(int myclass = 48 ; myclass<51; myclass++)
{
//////////////// 每隔X个读视频////
	//int getnum[8] = {0,0,0,0,0,0,0,0};
	//generate_random_nonrepeating_num(getnum);
	//vector<string> filename;

	//for (int i = 0 ; i<required_num; i++)
	//{
	//	filename.push_back(videonames[myclass][getnum[i]]);
	//}
///////////////////////////	

	for (int num = 0;num<videonames[myclass].size();num++) //循环所有视频
	{
		gettimeofday(&onevideo_starttime, NULL);
		string video = videonames[myclass][num];
		string name = videonames[myclass][num];
		std::cout << "video name is:  "<<name<<std::endl;
		name.erase(name.begin(),name.begin() + name.rfind("/") + 1);
		name.erase(name.end()-4,name.end());
		name = save_videoname_prefix + constnames[myclass] + "/"+ name + ".txt";
		std::ofstream file(name.c_str(),std::ios::out|std::ios::binary|std::ios::ate|std::ios::trunc);

	VideoCapture capture;
	//char* video = argv[1];
	//int flag = arg_parse(argc, argv);
	int flag = 0;
	capture.open(video);

	if(!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);

	std::vector<Frame> bb_list;

	
	
	if (bb_file.length())
	{
		bb_file.clear();
		bb_file.assign("/home/lijun/tvcj/improved_trajectory_release/bb_file/HMDB51");                   //////////////////////////////boundingbox location
		//bb_file.assign("H:\\improved_trajectory_release\\bb_file\\Olympic_Sports");

		bb_file.append(video,video.rfind('/'),video.length()-4-video.rfind('/'));                         /////video file location

		bb_file.append(".bb");
		std::cout<<"bb file is : "<<bb_file<<std::endl;
	}
	
	

	if(bb_file.length()) {
		LoadBoundBox(bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}

	if(flag)
		seqInfo.length = end_frame - start_frame + 1;

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track == 1)
		namedWindow("DenseTrackStab", 0);

	SurfFeatureDetector detector_surf(200);
	SurfDescriptorExtractor extractor_surf(true, true);

	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;

	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf;
	Mat flow, human_mask;

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
	while(true) {
		Mat frame;
		int i, j, c;

		// get a new frame
		capture >> frame;
		if(frame.empty())
			break;

		if(frame_num < start_frame || frame_num > end_frame) {
			frame_num++;
			continue;
		}

		if(frame_num == start_frame) {
			image.create(frame.size(), CV_8UC3);                   // build previse grey pyramid 
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);
			BuildPry(sizes, CV_32FC2, flow_warp_pyr);          // warp flow

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_warp_pyr);            //warp poly

			xyScaleTracks.resize(scale_num);

			frame.copyTo(image);
			cvtColor(image, prev_grey, CV_BGR2GRAY);

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);         //从灰度图上稠密采样特征点 5*5之内只有1个点

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(i = 0; i < points.size(); i++)
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
			}

			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, 7, 1.5);

			human_mask = Mat::ones(frame.size(), CV_8UC1);
			if(bb_file.length())
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);                           //human detector

			detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
			extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);                  //extract surf descriptor

			frame_num++;
			continue;
		}

		init_counter++;
		frame.copyTo(image);
		cvtColor(image, grey, CV_BGR2GRAY);

		// match surf features
		if(bb_file.length())
			InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
		detector_surf.detect(grey, kpts_surf, human_mask);
		extractor_surf.compute(grey, kpts_surf, desc_surf);
		ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);   //corresponding SURF point

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2);

		MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);                  //corresponding tracking point from flow
		MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);           //combine SURF point and tracking point 

		Mat H = Mat::eye(3, 3, CV_64FC1);
		if(pts_all.size() > 50) {
			std::vector<unsigned char> match_mask;
			Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
			if(countNonZero(Mat(match_mask)) > 25)
				H = temp;
		}

		Mat H_inv = H.inv();
		Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
		MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, 7, 1.5);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2);         //更新WARP后的flow

		for(int iScale = 0; iScale < scale_num; iScale++) {
			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(flow_warp_pyr[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
			MbhComp(flow_warp_pyr[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];




			//对金字塔iscale层的图像上的所有特征点进行跟踪，算出下一帧该点所在位置，并更新每一点的HOG,HOF,MBH，index为该点已跟踪帧数
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);           //如不图像范围内 则抹去该点
					continue;
				}

				iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
				iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];

				// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);

				// draw the trajectories at the first scale
				if(show_track == 1 && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

				// if the trajectory achieves the maximal length
				if(iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];
				
					std::vector<Point2f> displacement(trackInfo.length);
					for (int i = 0; i < trackInfo.length; ++i)
						displacement[i] = iTrack->disp[i]*fscales[iScale];
	
					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length) && IsCameraMotion(displacement)) {
						// output the trajectory
				//		printf("%d\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);

						//输入总帧数
						file.precision(7);
						file.setf(std::ios::fixed);
						file<<num+1<<"\t";
						//输入当前帧数，X平均值等
						file<<frame_num<<"\t"<<mean_x<<"\t"<<mean_y<<"\t"<<var_x<<"\t"<<var_y<<"\t"<<length<<"\t"<<fscales[iScale]<<"\t";


						// for spatio-temporal pyramid
					    
						file<< std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999)<<"\t"<<std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999)<<"\t"<<std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0 - start_frame)/float(seqInfo.length), 0), 0.999)<<"\t";

						// output the trajectory
						for (int i = 0; i < trackInfo.length; ++i)
						{
							file<<trajectory[i].x<<"\t"<<trajectory[i].y<<"\t";
						}
						PrintDesc(iTrack->hog, hogInfo, trackInfo,file);
						PrintDesc(iTrack->hof, hofInfo, trackInfo,file);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo,file);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo,file);

				//		printf("\n");
						file<<std::endl;
					}

					iTrack = tracks.erase(iTrack);        //特征点跟踪15帧后，无论是否是静止的，运动范围大的，还是正常运动的点都需要除去
					continue;
				}
				++iTrack;
			}//one trajectory end

			
			////////////////////////////////////////////////////////////
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every gap frames
			std::vector<Point2f> points(0);

			//储存已有特征点，使新增的特征点不与已有特征点重复
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for(i = 0; i < points.size(); i++)
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for(i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

		prev_kpts_surf = kpts_surf;
		desc_surf.copyTo(prev_desc_surf);

		frame_num++;


		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			c = cvWaitKey(3);
			if((char)c == 27) break;
		}

	}//one frame end


	std::cout<<"the class"<<myclass<<" the "<<num<<"  th video end"<<std::endl;
	//file1<<num<<std::endl;
	gettimeofday(&onevideo_endtime, NULL);
	std::cout<<"the "<<num<<" th video end time is "<<(onevideo_endtime.tv_sec-onevideo_starttime.tv_sec)<<" s"<<std::endl;

	file.close();
	} //for end   one video

	//file1<<"total time is "<<totaltime/1000<<" s "<<std::endl;
	//file.close();
	//file1.close();
	//file2.close();
	//file3.close();
	//file4.close();
	//file5.close();


}// for end 1 class video	

	if( show_track == 1 )
		destroyWindow("DenseTrackStab");

	gettimeofday(&totaltime, NULL);
	std::cout<<"time =   "<<(totaltime.tv_sec-starttime.tv_sec)/60<<" m"<<std::endl;

	return 0;
}
