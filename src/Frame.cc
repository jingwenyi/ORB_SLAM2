/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Ra煤l Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

//双目图像的Frame 对象构造
Frame::Frame(const cv::Mat &imLeft, 			//左摄像头灰度图
				const cv::Mat &imRight, 		//右摄像头灰度图
				const double &timeStamp, 		//图像对象的时间戳
				ORBextractor* extractorLeft, 	//左摄像头orb 提取器对象指针
				ORBextractor* extractorRight,   //右摄像头orb 提取器对象指针
				ORBVocabulary* voc, 			//orb 词典ORBvoc.txt
				cv::Mat &K, 					//相机内参矩阵
				cv::Mat &distCoef,				//相机畸变矩阵
				const float &bf, 				//双目相机基线b 和fx 的乘积
				const float &thDepth)			//相机深度阈值
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), 
    mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    //设置Frame id
    mnId=nNextId++;

    // Scale Level Info
    //读取orb 缩放级别信息，在文件TUM1.yaml
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //开一个线程去做左摄像头orb 关键点提取工作
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
	//开一个线程去做右摄像头orb 关键点提取工作
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
	//等待 左摄像头关键点提取完成
    threadLeft.join();
	//等待右摄像头关键点提取完成
    threadRight.join();

	//获取左摄像头关键点个数
    N = mvKeys.size();

	//如果关键点向量为空，表示提取失败，返回
    if(mvKeys.empty())
        return;

	//使用相机内参和畸变参数校正做摄像头的关键点
    UndistortKeyPoints();

	//左右摄像头关键点进行匹配，匹配成功后计算深度
	//保存与左关键点关联的右坐标
    ComputeStereoMatches();

	//为关键点关联的地图点申请空间
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
    	//计算图像边界
        ComputeImageBounds(imLeft);
		//代码中设置了woindsize = 100 , 就是10*10, 就是宽10和长10
		//这里网格的行设置的是64， 网格的宽设计的是48
		//因为图像的分辨率是640x480
		//计算网格宽度的倒数
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

		//保存相机内参
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		//求相机焦距的倒数
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

	//计算双目的基线b
    mb = mbf/fx;

	//为网格分配关键点以加速匹配
    AssignFeaturesToGrid();
}


//rgbd 图像 Frame 构造
Frame::Frame(const cv::Mat &imGray, 			//图像灰度图
				const cv::Mat &imDepth, 		//图像对应的深度
				const double &timeStamp, 		//图像对应的时间戳
				ORBextractor* extractor,		//orb提取器对象指针
				ORBVocabulary* voc, 			//orb 词典ORBvoc.txt
				cv::Mat &K, 					//相机内参矩阵
				cv::Mat &distCoef, 				//相机畸变矩阵
				const float &bf,				// 双目相机基线b 和fx 的乘积
				const float &thDepth)			//相机深度阈值
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    //设置frame id
    mnId=nNextId++;

    // Scale Level Info
    //读取orb 缩放级别信息，在文件TUM1.yaml
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //提取图像的orb 特征
    ExtractORB(0,imGray);

	//记录orb 提取到关键点的个数
    N = mvKeys.size();

	//如果关键点向量为空，表示提取失败，返回
    if(mvKeys.empty())
        return;

	//用相机内参和畸变参数，校正关键点
    UndistortKeyPoints();

	//求关键点的立体坐标和深度信息
    ComputeStereoFromRGBD(imDepth);

	//为关键点关联的地图点申请空间
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
    	//计算图像边界
        ComputeImageBounds(imGray);
		//代码中设置了woindsize = 100 , 就是10*10, 就是宽10和长10
		//这里网格的行设置的是64， 网格的宽设计的是48
		//因为图像的分辨率是640x480
		//计算网格宽度的倒数
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

		//保存相机内参
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		//求相机焦距的倒数
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

	//计算双目的基线b
    mb = mbf/fx;

	//为网格分配关键点以加速匹配
    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray,  			//图像的灰度图
				const double &timeStamp, 		//图像对应的时间戳
				ORBextractor* extractor, 		//orb提取器对象指针
				ORBVocabulary* voc, 			//orb 词典ORBvoc.txt
				cv::Mat &K, 					//相机内参矩阵TUM1.yaml文件中
				cv::Mat &distCoef, 				//相机畸变矩阵TUM1.yaml文件中
				const float &bf, 				//双目相机基线b 和fx 的乘积
				const float &thDepth)			//相机深度阈值
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    //设置frame id
    mnId=nNextId++;

    // Scale Level Info
    //读取orb 缩放级别信息，在文件TUM1.yaml
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //提取图像的orb 特征
    ExtractORB(0,imGray);

	//记录orb 提取到的关键点个数
    N = mvKeys.size();

	//如果关键点向量为空，表示提取失败，返回
    if(mvKeys.empty())
        return;

	//用相机内参和畸变参数，校正关键点
    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

	//为关键点关联的地图点申请空间
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
    	//计算图像的边界
        ComputeImageBounds(imGray);

		//代码中设置了woindsize = 100 , 就是10*10, 就是宽10和长10
		//这里网格的行设置的是64， 网格的宽设计的是48
		//因为图像的分辨率是640x480
		//计算网格宽度的倒数
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
		//计算网格长度的倒数
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

		//保存相机内参
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		//求相机焦距的倒数
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

	//计算双目摄像机的基线 b
    mb = mbf/fx;

	//为网格分配关键点以加速特征匹配
    AssignFeaturesToGrid();
}

//为网格分配关键点以加速特征匹配
void Frame::AssignFeaturesToGrid()
{
	//求出每个网格平均有多少个关键点，为啥去1/2?
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
	//为每个网格向量申请预留空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
		//对每个关键点进行缩放，push到对应的网格中
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

//在构造frame 进行的orb 特征提取
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
	//这里调用了ORBextractor::operator() 重载函数
	//把提取到的关键点放到mvKeys 关键点向量中
	//把orb 描述符保存到mDescriptors 中
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}


//把参考帧的关键点放到当前帧的区域中去匹配
vector<size_t> Frame::GetFeaturesInArea(const float &x, //参考帧关键点的坐标x 值
											const float  &y, //参考帧关键点的坐标y 值
											const float  &r,  //窗口大小
											const int minLevel,  //指定的图像金字塔开始层
											const int maxLevel) const //指定的图像金字塔结束层
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

	//把参考帧x 坐标窗口向上滑动一个窗口缩放，检查是否在网格内
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

	//把参考帧x 坐标窗口向下滑动一个窗口缩放，检查是否在网格内
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

	//把参考帧y 坐标窗口向左滑动一个窗口缩放，检查是否在网格内
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

	//把参考帧y 坐标窗口向右滑动一个窗口缩放，检查是否在网格内
    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

	//检查图像金字塔层数是否大于0
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

	//遍历所属区域的点
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
        	//获取网格点保存关键点的坐标
            const vector<size_t> vCell = mGrid[ix][iy];
			//如果该网格点没有关键点返回
            if(vCell.empty())
                continue;
			//遍历每一个关键点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
            	//获取关键点
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
				//检查关键点对于的图像金字塔层是否正确
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

				//计算当前帧关键点坐标和参考帧关键点坐标距离差
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;
				//检查距离差是否在一个窗口内
                if(fabs(distx)<r && fabs(disty)<r)
					//把找到的关键点放到返回向量中
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
	//对关键点进行缩放到对应的网格中，网格是64x48
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

//把Frame的关键点和orb描述符，转化为bow视觉词袋
void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}


//用相机内参和畸变参数校正关键点
void Frame::UndistortKeyPoints()
{
	//如果没有相机畸变参数则不处理，直接相等
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    //获取所有关键点的坐标，填充Nx2矩阵
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    //改变通道，还是Nx2矩阵
    mat=mat.reshape(2);
	//用相机内参和畸变参数校正关键点
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
	//改变通道，还是Nx2矩阵
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    //用校正后的关键点填充mvkeysUn关键点向量
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}


//计算图像的边界
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
    	//将图像的4 个角的坐标作观察点矩阵
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        //修改矩阵通道为2，还是4x2矩阵
        mat=mat.reshape(2);
		//通过相机的内参和畸变参数把图像
		//四个角的坐标旋转平移到图像理想点坐标
		//mk:是相机内参矩阵
		//mDistCoef: 是相机畸变矩阵
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
		//修改矩阵通道为1，还是4x2矩阵
        mat=mat.reshape(1);

		//获取图像边界x 的最小值和最大值
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
		//获取图像边界y 的最小值和最大值
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
    	//如果没有提供相机内参和畸变参数
    	//图像x  的范围就是0-cols
    	//图像y  的范围就是0-row
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

//左右摄像头关键点进行匹配
//匹配成功计算深度，记录与左关键点关联的右坐标
void Frame::ComputeStereoMatches()
{
	//为关键点立体坐标申请向量空间
    mvuRight = vector<float>(N,-1.0f);
	//为关键点深度信息申请向量空间
    mvDepth = vector<float>(N,-1.0f);

	//获取像素点相差最佳像素点阈值的 中值
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

	//获取左摄像头orb 提取器图像金字塔的0 层的行数
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    //申请一个vRowIndices 向量来管理每行，为每行申请向量
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());
	
	//为每行的向量保留200个空间
    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

	//获取右摄像头的关键点个数
    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
    	//依次获取右摄像头关键点
        const cv::KeyPoint &kp = mvKeysRight[iR];
		//获取关键点的y 坐标，即行坐标
        const float &kpY = kp.pt.y;
		//获取图像金字塔的该层的比例因子
		//搜索窗口等于比例因子x 2
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];

		//获取该关键点搜索窗口的范围( min r  - maxr)
		const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

		//把该搜索窗口的所有点加入到对应的行向量中
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    //设置搜索限制
    const float minZ = mb;  //双目的基线b, mb = mbf /fx(mb 在这里还没有初始化,bug)
    const float minD = 0;  //最小视差，如果p点在双目的正中，d=xl - xr = 0 
    const float maxD = mbf/minZ;  // 最大视差 fx

    // For each left keypoint search a match in the right image
    //保存左关键点和对于的右关键点的SAD偏差
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

	//遍历左关键点, 在右关键点总去匹配
    for(int iL=0; iL<N; iL++)
    {
    	//获取左关键点
        const cv::KeyPoint &kpL = mvKeys[iL];
		//左关键点的金字塔层
        const int &levelL = kpL.octave;
		//左关键点的x,y 坐标
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

		//通过左关键点的行坐标去获取右关键点对于的行
		//左右摄像机在一个平面上，获取的关键点x 坐标可以不一样，
		//y坐标在的行号相同,而且右关键点行的搜索范围有所扩大(minr-maxr)
        const vector<size_t> &vCandidates = vRowIndices[vL];

		//如果没有匹配到右关键点的行号就跳过该关键点
        if(vCandidates.empty())
            continue;

		//根据视差范围，求出该关键点x 坐标对应的右关键点x 坐标的范围
        const float minU = uL-maxD; //最小匹配范围
        const float maxU = uL-minD; //最大匹配范围

        if(maxU<0)
            continue;

		//初始化最佳距离像素点
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

		//获取左关键点的orb 描述子
        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        //匹配所有在该范围内的右关键点的特征描述符，距离越小越好
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
        	//获取匹配到的右关键点
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

			//如果该右关键点的金字塔层不在左关键点 l-1到l+1直接就舍弃
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

			//获取该点的x 坐标
            const float &uR = kpR.pt.x;

			//获取的右关键点x 坐标在我们计算范围内
            if(uR>=minU && uR<=maxU)
            {
            	//获取该右关键点的特征描述子
                const cv::Mat &dR = mDescriptorsRight.row(iR);
				//获取特征描述子的距离
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                	//记录特征描述子 距离最小的点和距离大小
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        //亚像素匹配
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            //获取描述子匹配到的右关键点的x 坐标
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
			//获取左关键点对应的图像金字塔的缩放比例
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
			//计算左关键点坐标在该层图像金字塔对应的坐标
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
			//计算右关键点在该层对应的x 坐标
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;//滑动窗口大小 2*w +1, 保证关键点为中心点
            //在对应图像金字塔层上获取左关键点对应的滑动窗口
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
			//把IL 矩阵转换成32位浮点型矩阵
            IL.convertTo(IL,CV_32F);
			//窗口中的每个元素减去正中心的那个元素
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
			//为SAD算法申请差的向量
            vector<float> vDists;
            vDists.resize(2*L+1);

			//计算右关键点滑动窗口范围，这里的范围xr -L-w  到xr+L+w+1
            const float iniu = scaleduR0+L-w;  //这里应该错了，bug
            const float endu = scaleduR0+L+w+1;
			//提前检查滑动窗口是否会越界
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
            	//row 不变，横向移动窗口
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
				//窗口中的每个元素减去正中心的那个元素
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

				//矩阵范数，求矩阵列向量绝对值之和的最大值
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist; //SAD匹配目前做小的偏差
                    bestincR = incR; //SAD 匹配目前最佳的修正量
                }

                vDists[L+incR] = dist; //保存每一个SAD 匹配偏差，由于颜色梯度的问题，这里数据应该是抛物线变化
            }

			//整个滑动窗口中，SAD最小值不是以抛物线形式出现，匹配失败
            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            //获取抛物线顶点的前一个点
            const float dist1 = vDists[L+bestincR-1];
			//获取抛物线顶点
            const float dist2 = vDists[L+bestincR];
			//获取抛物线顶点的后一个点
            const float dist3 = vDists[L+bestincR+1];

			//bestincR+deltaR就是抛物线谷底的位置，相对SAD匹配出的最小值bestincR的修正量为deltaR
            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

			//抛物线拟合修正量不能超过一个像素
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // 通过描述子匹配得到匹配点位置为scaleduR0
            // 通过SAD匹配找到修正量bestincR
            // 通过抛物线拟合找到亚像素修正量deltaR
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

			//计算左右关键x 坐标的视差
            float disparity = (uL-bestuR);

			//确保视差在设定的阈值内
            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
				//计算关键点的深度
                mvDepth[iL]=mbf/disparity;
				//保存匹配的右关键点x 坐标
                mvuRight[iL] = bestuR;
				//关联左关键点和对应的右关键点SAD 偏差
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

	//对关键点按照SAD偏差进行排序
    sort(vDistIdx.begin(),vDistIdx.end());
	//获取SAD 偏差的中值
    const float median = vDistIdx[vDistIdx.size()/2].first;
	//设置偏差阈值
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
        	//删点大于阈值的点
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

//求关键点的深度信息和立体坐标
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
	//为关键点对于的立体坐标申请向量空间
    mvuRight = vector<float>(N,-1);
	//为关键点深度信息申请向量空间
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
    	//获取校正前的关键点坐标
        const cv::KeyPoint &kp = mvKeys[i];
		//获取校正后的关键点坐标
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;
		//获取关键点对于的深度信息，
		//以校正前的关键点为准
        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
        	//保存关键点的深度值
            mvDepth[i] = d;
			//计算保存关键点x 轴对于的立体坐标
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
