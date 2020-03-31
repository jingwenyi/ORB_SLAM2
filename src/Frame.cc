/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
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

//˫Ŀͼ���Frame ������
Frame::Frame(const cv::Mat &imLeft, 			//������ͷ�Ҷ�ͼ
				const cv::Mat &imRight, 		//������ͷ�Ҷ�ͼ
				const double &timeStamp, 		//ͼ������ʱ���
				ORBextractor* extractorLeft, 	//������ͷorb ��ȡ������ָ��
				ORBextractor* extractorRight,   //������ͷorb ��ȡ������ָ��
				ORBVocabulary* voc, 			//orb �ʵ�ORBvoc.txt
				cv::Mat &K, 					//����ڲξ���
				cv::Mat &distCoef,				//����������
				const float &bf, 				//˫Ŀ�������b ��fx �ĳ˻�
				const float &thDepth)			//��������ֵ
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), 
    mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    //����Frame id
    mnId=nNextId++;

    // Scale Level Info
    //��ȡorb ���ż�����Ϣ�����ļ�TUM1.yaml
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //��һ���߳�ȥ��������ͷorb �ؼ�����ȡ����
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
	//��һ���߳�ȥ��������ͷorb �ؼ�����ȡ����
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
	//�ȴ� ������ͷ�ؼ�����ȡ���
    threadLeft.join();
	//�ȴ�������ͷ�ؼ�����ȡ���
    threadRight.join();

	//��ȡ������ͷ�ؼ������
    N = mvKeys.size();

	//����ؼ�������Ϊ�գ���ʾ��ȡʧ�ܣ�����
    if(mvKeys.empty())
        return;

	//ʹ������ڲκͻ������У��������ͷ�Ĺؼ���
    UndistortKeyPoints();

	//��������ͷ�ؼ������ƥ�䣬ƥ��ɹ���������
	//��������ؼ��������������
    ComputeStereoMatches();

	//Ϊ�ؼ�������ĵ�ͼ������ռ�
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
    	//����ͼ��߽�
        ComputeImageBounds(imLeft);
		//������������woindsize = 100 , ����10*10, ���ǿ�10�ͳ�10
		//��������������õ���64�� ����Ŀ���Ƶ���48
		//��Ϊͼ��ķֱ�����640x480
		//���������ȵĵ���
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

		//��������ڲ�
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		//���������ĵ���
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

	//����˫Ŀ�Ļ���b
    mb = mbf/fx;

	//Ϊ�������ؼ����Լ���ƥ��
    AssignFeaturesToGrid();
}


//rgbd ͼ�� Frame ����
Frame::Frame(const cv::Mat &imGray, 			//ͼ��Ҷ�ͼ
				const cv::Mat &imDepth, 		//ͼ���Ӧ�����
				const double &timeStamp, 		//ͼ���Ӧ��ʱ���
				ORBextractor* extractor,		//orb��ȡ������ָ��
				ORBVocabulary* voc, 			//orb �ʵ�ORBvoc.txt
				cv::Mat &K, 					//����ڲξ���
				cv::Mat &distCoef, 				//����������
				const float &bf,				// ˫Ŀ�������b ��fx �ĳ˻�
				const float &thDepth)			//��������ֵ
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    //����frame id
    mnId=nNextId++;

    // Scale Level Info
    //��ȡorb ���ż�����Ϣ�����ļ�TUM1.yaml
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //��ȡͼ���orb ����
    ExtractORB(0,imGray);

	//��¼orb ��ȡ���ؼ���ĸ���
    N = mvKeys.size();

	//����ؼ�������Ϊ�գ���ʾ��ȡʧ�ܣ�����
    if(mvKeys.empty())
        return;

	//������ڲκͻ��������У���ؼ���
    UndistortKeyPoints();

	//��ؼ������������������Ϣ
    ComputeStereoFromRGBD(imDepth);

	//Ϊ�ؼ�������ĵ�ͼ������ռ�
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
    	//����ͼ��߽�
        ComputeImageBounds(imGray);
		//������������woindsize = 100 , ����10*10, ���ǿ�10�ͳ�10
		//��������������õ���64�� ����Ŀ���Ƶ���48
		//��Ϊͼ��ķֱ�����640x480
		//���������ȵĵ���
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

		//��������ڲ�
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		//���������ĵ���
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

	//����˫Ŀ�Ļ���b
    mb = mbf/fx;

	//Ϊ�������ؼ����Լ���ƥ��
    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray,  			//ͼ��ĻҶ�ͼ
				const double &timeStamp, 		//ͼ���Ӧ��ʱ���
				ORBextractor* extractor, 		//orb��ȡ������ָ��
				ORBVocabulary* voc, 			//orb �ʵ�ORBvoc.txt
				cv::Mat &K, 					//����ڲξ���TUM1.yaml�ļ���
				cv::Mat &distCoef, 				//����������TUM1.yaml�ļ���
				const float &bf, 				//˫Ŀ�������b ��fx �ĳ˻�
				const float &thDepth)			//��������ֵ
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    //����frame id
    mnId=nNextId++;

    // Scale Level Info
    //��ȡorb ���ż�����Ϣ�����ļ�TUM1.yaml
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    //��ȡͼ���orb ����
    ExtractORB(0,imGray);

	//��¼orb ��ȡ���Ĺؼ������
    N = mvKeys.size();

	//����ؼ�������Ϊ�գ���ʾ��ȡʧ�ܣ�����
    if(mvKeys.empty())
        return;

	//������ڲκͻ��������У���ؼ���
    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

	//Ϊ�ؼ�������ĵ�ͼ������ռ�
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
    	//����ͼ��ı߽�
        ComputeImageBounds(imGray);

		//������������woindsize = 100 , ����10*10, ���ǿ�10�ͳ�10
		//��������������õ���64�� ����Ŀ���Ƶ���48
		//��Ϊͼ��ķֱ�����640x480
		//���������ȵĵ���
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
		//�������񳤶ȵĵ���
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

		//��������ڲ�
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		//���������ĵ���
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

	//����˫Ŀ������Ļ��� b
    mb = mbf/fx;

	//Ϊ�������ؼ����Լ�������ƥ��
    AssignFeaturesToGrid();
}

//Ϊ�������ؼ����Լ�������ƥ��
void Frame::AssignFeaturesToGrid()
{
	//���ÿ������ƽ���ж��ٸ��ؼ��㣬Ϊɶȥ1/2?
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
	//Ϊÿ��������������Ԥ���ռ�
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
		//��ÿ���ؼ���������ţ�push����Ӧ��������
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

//�ڹ���frame ���е�orb ������ȡ
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
	//���������ORBextractor::operator() ���غ���
	//����ȡ���Ĺؼ���ŵ�mvKeys �ؼ���������
	//��orb ���������浽mDescriptors ��
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


//�Ѳο�֡�Ĺؼ���ŵ���ǰ֡��������ȥƥ��
vector<size_t> Frame::GetFeaturesInArea(const float &x, //�ο�֡�ؼ��������x ֵ
											const float  &y, //�ο�֡�ؼ��������y ֵ
											const float  &r,  //���ڴ�С
											const int minLevel,  //ָ����ͼ���������ʼ��
											const int maxLevel) const //ָ����ͼ�������������
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

	//�Ѳο�֡x ���괰�����ϻ���һ���������ţ�����Ƿ���������
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

	//�Ѳο�֡x ���괰�����»���һ���������ţ�����Ƿ���������
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

	//�Ѳο�֡y ���괰�����󻬶�һ���������ţ�����Ƿ���������
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

	//�Ѳο�֡y ���괰�����һ���һ���������ţ�����Ƿ���������
    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

	//���ͼ������������Ƿ����0
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

	//������������ĵ�
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
        	//��ȡ����㱣��ؼ��������
            const vector<size_t> vCell = mGrid[ix][iy];
			//����������û�йؼ��㷵��
            if(vCell.empty())
                continue;
			//����ÿһ���ؼ���
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
            	//��ȡ�ؼ���
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
				//���ؼ�����ڵ�ͼ����������Ƿ���ȷ
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

				//���㵱ǰ֡�ؼ�������Ͳο�֡�ؼ�����������
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;
				//��������Ƿ���һ��������
                if(fabs(distx)<r && fabs(disty)<r)
					//���ҵ��Ĺؼ���ŵ�����������
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
	//�Թؼ���������ŵ���Ӧ�������У�������64x48
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

//��Frame�Ĺؼ����orb��������ת��Ϊbow�Ӿ��ʴ�
void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}


//������ڲκͻ������У���ؼ���
void Frame::UndistortKeyPoints()
{
	//���û�������������򲻴���ֱ�����
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    //��ȡ���йؼ�������꣬���Nx2����
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    //�ı�ͨ��������Nx2����
    mat=mat.reshape(2);
	//������ڲκͻ������У���ؼ���
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
	//�ı�ͨ��������Nx2����
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    //��У����Ĺؼ������mvkeysUn�ؼ�������
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}


//����ͼ��ı߽�
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
    	//��ͼ���4 ���ǵ��������۲�����
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        //�޸ľ���ͨ��Ϊ2������4x2����
        mat=mat.reshape(2);
		//ͨ��������ڲκͻ��������ͼ��
		//�ĸ��ǵ�������תƽ�Ƶ�ͼ�����������
		//mk:������ڲξ���
		//mDistCoef: ������������
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
		//�޸ľ���ͨ��Ϊ1������4x2����
        mat=mat.reshape(1);

		//��ȡͼ��߽�x ����Сֵ�����ֵ
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
		//��ȡͼ��߽�y ����Сֵ�����ֵ
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
    	//���û���ṩ����ڲκͻ������
    	//ͼ��x  �ķ�Χ����0-cols
    	//ͼ��y  �ķ�Χ����0-row
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

//��������ͷ�ؼ������ƥ��
//ƥ��ɹ�������ȣ���¼����ؼ��������������
void Frame::ComputeStereoMatches()
{
	//Ϊ�ؼ��������������������ռ�
    mvuRight = vector<float>(N,-1.0f);
	//Ϊ�ؼ��������Ϣ���������ռ�
    mvDepth = vector<float>(N,-1.0f);

	//��ȡ���ص����������ص���ֵ�� ��ֵ
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

	//��ȡ������ͷorb ��ȡ��ͼ���������0 �������
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    //����һ��vRowIndices ����������ÿ�У�Ϊÿ����������
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());
	
	//Ϊÿ�е���������200���ռ�
    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

	//��ȡ������ͷ�Ĺؼ������
    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
    	//���λ�ȡ������ͷ�ؼ���
        const cv::KeyPoint &kp = mvKeysRight[iR];
		//��ȡ�ؼ����y ���꣬��������
        const float &kpY = kp.pt.y;
		//��ȡͼ��������ĸò�ı�������
		//�������ڵ��ڱ�������x 2
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];

		//��ȡ�ùؼ����������ڵķ�Χ( min r  - maxr)
		const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

		//�Ѹ��������ڵ����е���뵽��Ӧ����������
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    //������������
    const float minZ = mb;  //˫Ŀ�Ļ���b, mb = mbf /fx(mb �����ﻹû�г�ʼ��,bug)
    const float minD = 0;  //��С�Ӳ���p����˫Ŀ�����У�d=xl - xr = 0 
    const float maxD = mbf/minZ;  // ����Ӳ� fx

    // For each left keypoint search a match in the right image
    //������ؼ���Ͷ��ڵ��ҹؼ����SADƫ��
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

	//������ؼ���, ���ҹؼ�����ȥƥ��
    for(int iL=0; iL<N; iL++)
    {
    	//��ȡ��ؼ���
        const cv::KeyPoint &kpL = mvKeys[iL];
		//��ؼ���Ľ�������
        const int &levelL = kpL.octave;
		//��ؼ����x,y ����
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

		//ͨ����ؼ����������ȥ��ȡ�ҹؼ�����ڵ���
		//�����������һ��ƽ���ϣ���ȡ�Ĺؼ���x ������Բ�һ����
		//y�����ڵ��к���ͬ,�����ҹؼ����е�������Χ��������(minr-maxr)
        const vector<size_t> &vCandidates = vRowIndices[vL];

		//���û��ƥ�䵽�ҹؼ�����кž������ùؼ���
        if(vCandidates.empty())
            continue;

		//�����ӲΧ������ùؼ���x �����Ӧ���ҹؼ���x ����ķ�Χ
        const float minU = uL-maxD; //��Сƥ�䷶Χ
        const float maxU = uL-minD; //���ƥ�䷶Χ

        if(maxU<0)
            continue;

		//��ʼ����Ѿ������ص�
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

		//��ȡ��ؼ����orb ������
        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        //ƥ�������ڸ÷�Χ�ڵ��ҹؼ��������������������ԽСԽ��
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
        	//��ȡƥ�䵽���ҹؼ���
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

			//������ҹؼ���Ľ������㲻����ؼ��� l-1��l+1ֱ�Ӿ�����
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

			//��ȡ�õ��x ����
            const float &uR = kpR.pt.x;

			//��ȡ���ҹؼ���x ���������Ǽ��㷶Χ��
            if(uR>=minU && uR<=maxU)
            {
            	//��ȡ���ҹؼ��������������
                const cv::Mat &dR = mDescriptorsRight.row(iR);
				//��ȡ���������ӵľ���
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                	//��¼���������� ������С�ĵ�;����С
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        //������ƥ��
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            //��ȡ������ƥ�䵽���ҹؼ����x ����
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
			//��ȡ��ؼ����Ӧ��ͼ������������ű���
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
			//������ؼ��������ڸò�ͼ���������Ӧ������
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
			//�����ҹؼ����ڸò��Ӧ��x ����
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;//�������ڴ�С 2*w +1, ��֤�ؼ���Ϊ���ĵ�
            //�ڶ�Ӧͼ����������ϻ�ȡ��ؼ����Ӧ�Ļ�������
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
			//��IL ����ת����32λ�����;���
            IL.convertTo(IL,CV_32F);
			//�����е�ÿ��Ԫ�ؼ�ȥ�����ĵ��Ǹ�Ԫ��
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
			//ΪSAD�㷨����������
            vector<float> vDists;
            vDists.resize(2*L+1);

			//�����ҹؼ��㻬�����ڷ�Χ������ķ�Χxr -L-w  ��xr+L+w+1
            const float iniu = scaleduR0+L-w;  //����Ӧ�ô��ˣ�bug
            const float endu = scaleduR0+L+w+1;
			//��ǰ��黬�������Ƿ��Խ��
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
            	//row ���䣬�����ƶ�����
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
				//�����е�ÿ��Ԫ�ؼ�ȥ�����ĵ��Ǹ�Ԫ��
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

				//�����������������������ֵ֮�͵����ֵ
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist; //SADƥ��Ŀǰ��С��ƫ��
                    bestincR = incR; //SAD ƥ��Ŀǰ��ѵ�������
                }

                vDists[L+incR] = dist; //����ÿһ��SAD ƥ��ƫ�������ɫ�ݶȵ����⣬��������Ӧ���������߱仯
            }

			//�������������У�SAD��Сֵ��������������ʽ���֣�ƥ��ʧ��
            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            //��ȡ�����߶����ǰһ����
            const float dist1 = vDists[L+bestincR-1];
			//��ȡ�����߶���
            const float dist2 = vDists[L+bestincR];
			//��ȡ�����߶���ĺ�һ����
            const float dist3 = vDists[L+bestincR+1];

			//bestincR+deltaR���������߹ȵ׵�λ�ã����SADƥ�������СֵbestincR��������ΪdeltaR
            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

			//������������������ܳ���һ������
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // ͨ��������ƥ��õ�ƥ���λ��ΪscaleduR0
            // ͨ��SADƥ���ҵ�������bestincR
            // ͨ������������ҵ�������������deltaR
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

			//�������ҹؼ�x ������Ӳ�
            float disparity = (uL-bestuR);

			//ȷ���Ӳ����趨����ֵ��
            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
				//����ؼ�������
                mvDepth[iL]=mbf/disparity;
				//����ƥ����ҹؼ���x ����
                mvuRight[iL] = bestuR;
				//������ؼ���Ͷ�Ӧ���ҹؼ���SAD ƫ��
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

	//�Թؼ��㰴��SADƫ���������
    sort(vDistIdx.begin(),vDistIdx.end());
	//��ȡSAD ƫ�����ֵ
    const float median = vDistIdx[vDistIdx.size()/2].first;
	//����ƫ����ֵ
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
        	//ɾ�������ֵ�ĵ�
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

//��ؼ���������Ϣ����������
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
	//Ϊ�ؼ�����ڵ������������������ռ�
    mvuRight = vector<float>(N,-1);
	//Ϊ�ؼ��������Ϣ���������ռ�
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
    	//��ȡУ��ǰ�Ĺؼ�������
        const cv::KeyPoint &kp = mvKeys[i];
		//��ȡУ����Ĺؼ�������
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;
		//��ȡ�ؼ�����ڵ������Ϣ��
		//��У��ǰ�Ĺؼ���Ϊ׼
        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
        	//����ؼ�������ֵ
            mvDepth[i] = d;
			//���㱣��ؼ���x ����ڵ���������
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
