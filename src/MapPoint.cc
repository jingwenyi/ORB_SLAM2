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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const cv::Mat &Pos, 			//关键点的世界坐标点
								KeyFrame *pRefKF, 	//当前关键帧
								Map* pMap):			//局部地图
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector/cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos);
    return mWorldPos.clone();
}

cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

//为地图点添加观测帧和对于的特征点序号
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
	//count 函数是查找pkF 关键帧的个数
    if(mObservations.count(pKF))
        return;
	//如果没有找到，就添加
    mObservations[pKF]=idx;

	//更新能够观测到该地图点帧的个数
    if(pKF->mvuRight[idx]>=0)
        nObs+=2;
    else
        nObs++;
}

//删掉该地图点的关键帧
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
		//观察到该地图点所有帧中查找是否有该帧
        if(mObservations.count(pKF))
        {
        	//获取该帧的idx
            int idx = mObservations[pKF];
			//如果是双目，帧的个数减2 , 单目减1
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

			//删除该关键帧
            mObservations.erase(pKF);

			//如果该地图的参考关键帧为当前帧
            if(mpRefKF==pKF)
				//把该地图点的参考关键帧放到第一帧上
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            //如果该地图的观测帧少于2个帧
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
		//设置地图点为bad 标志
        SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return nObs;
}

void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;  //设置该地图点的bad 标志
        //获取能够观测到该地图点的关键帧的map 对象
        obs = mObservations;
		//清除map 对象,先clear , 是为了防止别的线程使用，内存还没有释放
        mObservations.clear();
    }
	//遍历map 对象
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
    	//获取关键帧
        KeyFrame* pKF = mit->first;
		//删掉该关键帧匹配的地图点
        pKF->EraseMapPointMatch(mit->second);
    }

	//在地图中删掉该地图点
    mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(MapPoint* pMP)
{
	//地图点序号，确保不是同一个地图点
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
		//获取地图点观测帧向量
        obs=mObservations;
		//清空向量
        mObservations.clear();
		//设置该地图点坏点标志
        mbBad=true;
		//获取地图点计数器
        nvisible = mnVisible;
        nfound = mnFound;
		//用传入的地图点代替
        mpReplaced = pMP;
    }

	//遍历所有观察到该地图点的关键帧向量
	//替所有能观测到该地图点的关键帧中的地图点都要替换
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        //获取关键帧
        KeyFrame* pKF = mit->first;

		//查找地图点是否被观测帧观测到
        if(!pMP->IsInKeyFrame(pKF))
        {
        	//该关键帧没有观测到该地图点
        	//为该帧插入地图点
            pKF->ReplaceMapPointMatch(mit->second, pMP);
			//为地图点添加关键帧
            pMP->AddObservation(pKF,mit->second);
        }
        else
        {	
        	//如果该地图点已经被该帧观测到
        	//就会产生矛盾就是一个地图点对应两个关键点
        	//所以这里没有替换，而是直接删掉
            pKF->EraseMapPointMatch(mit->second);
        }
    }
	//更新观测计数器
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
	//计算具有代表性的描述子
    pMP->ComputeDistinctiveDescriptors();

	//从地图中删除该地图点
    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}


//计算具有代表的描述子
//由于一个地图点会被很多关键帧观测到，因此在插入关键帧后
//需要判断是否更新当前地图点的最合适描述子
//先获得当前地图点的所有描述子，然后计算描述子之间的两两距离
//最好的描述子和其他的描述子应该有最小的距离中值
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors;

    map<KeyFrame*,size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        if(mbBad)
            return;
        observations=mObservations;
    }

    if(observations.empty())
        return;

    vDescriptors.reserve(observations.size());

	//遍历观测到的3d 点的所有关键帧，获取orb 描述子
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
			//把orb 描述子插入到vDescriptors
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

	//计算每个描述子两两直接的汉明距离
    float Distances[N][N];
	//vector<vector<float>>  Distances;
	//Distances.resize(N,vector<float>(N,0));
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
        	//计算两个描述子直接的汉明距离
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
	//获取中间距离最小的描述子
    for(size_t i=0;i<N;i++)
    {
    	//vector<int> vDists(Distances[i].begin(),Distances[i].end());
        vector<int> vDists(Distances[i],Distances[i]+N);
        sort(vDists.begin(),vDists.end());
        int median = vDists[0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
		//最好的描述子就是和其他描述子的距离最小
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

//查找地图点是否被该关键帧观测到
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

//更新地图点平均观测方向和观测距离的范围
//
void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if(mbBad)
            return;
		//获取观测到该3D 点的所有观察帧
        observations=mObservations;
		//观测到该点的参考关键帧
        pRefKF=mpRefKF; 
		//3D 点在世界坐标系中的位置
        Pos = mWorldPos.clone();
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first; //获取关键帧
        cv::Mat Owi = pKF->GetCameraCenter(); //获取关键帧的中心位置
        cv::Mat normali = mWorldPos - Owi; //获取该3D 点相对于相机中心位置的坐标
        //对所有观测帧对该点的观测方向归一化为为单位向量进行求和
        normal = normal + normali/cv::norm(normali);
        n++;
    }

	//参考关键帧相机指向3D 点的方向
    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
	//该点到参考关键帧相机的距离
    const float dist = cv::norm(PC);
	//获取参考帧关键点在图像金字塔的第几层
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
	//计算缩放比例
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
	//获取图像金字塔层数
    const int nLevels = pRefKF->mnScaleLevels;

    {
        unique_lock<mutex> lock3(mMutexPos);
		//观测到该点距离最大值
        mfMaxDistance = dist*levelScaleFactor;
		//观测到该点距离最小值
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
		//获取平均观测方向
        mNormalVector = normal/n;
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

//通过距离，预测在地图点在关键帧的图像金字塔的那一层
int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
		//计算
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
