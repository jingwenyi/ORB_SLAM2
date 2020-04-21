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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const cv::Mat &Pos, 			//�ؼ�������������
								KeyFrame *pRefKF, 	//��ǰ�ؼ�֡
								Map* pMap):			//�ֲ���ͼ
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

//Ϊ��ͼ����ӹ۲�֡�Ͷ��ڵ����������
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
	//count �����ǲ���pkF �ؼ�֡�ĸ���
    if(mObservations.count(pKF))
        return;
	//���û���ҵ��������
    mObservations[pKF]=idx;

	//�����ܹ��۲⵽�õ�ͼ��֡�ĸ���
    if(pKF->mvuRight[idx]>=0)
        nObs+=2;
    else
        nObs++;
}

//ɾ���õ�ͼ��Ĺؼ�֡
void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
		//�۲쵽�õ�ͼ������֡�в����Ƿ��и�֡
        if(mObservations.count(pKF))
        {
        	//��ȡ��֡��idx
            int idx = mObservations[pKF];
			//�����˫Ŀ��֡�ĸ�����2 , ��Ŀ��1
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

			//ɾ���ùؼ�֡
            mObservations.erase(pKF);

			//����õ�ͼ�Ĳο��ؼ�֡Ϊ��ǰ֡
            if(mpRefKF==pKF)
				//�Ѹõ�ͼ��Ĳο��ؼ�֡�ŵ���һ֡��
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            //����õ�ͼ�Ĺ۲�֡����2��֡
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
		//���õ�ͼ��Ϊbad ��־
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
        mbBad=true;  //���øõ�ͼ���bad ��־
        //��ȡ�ܹ��۲⵽�õ�ͼ��Ĺؼ�֡��map ����
        obs = mObservations;
		//���map ����,��clear , ��Ϊ�˷�ֹ����߳�ʹ�ã��ڴ滹û���ͷ�
        mObservations.clear();
    }
	//����map ����
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
    	//��ȡ�ؼ�֡
        KeyFrame* pKF = mit->first;
		//ɾ���ùؼ�֡ƥ��ĵ�ͼ��
        pKF->EraseMapPointMatch(mit->second);
    }

	//�ڵ�ͼ��ɾ���õ�ͼ��
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
	//��ͼ����ţ�ȷ������ͬһ����ͼ��
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
		//��ȡ��ͼ��۲�֡����
        obs=mObservations;
		//�������
        mObservations.clear();
		//���øõ�ͼ�㻵���־
        mbBad=true;
		//��ȡ��ͼ�������
        nvisible = mnVisible;
        nfound = mnFound;
		//�ô���ĵ�ͼ�����
        mpReplaced = pMP;
    }

	//�������й۲쵽�õ�ͼ��Ĺؼ�֡����
	//�������ܹ۲⵽�õ�ͼ��Ĺؼ�֡�еĵ�ͼ�㶼Ҫ�滻
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        //��ȡ�ؼ�֡
        KeyFrame* pKF = mit->first;

		//���ҵ�ͼ���Ƿ񱻹۲�֡�۲⵽
        if(!pMP->IsInKeyFrame(pKF))
        {
        	//�ùؼ�֡û�й۲⵽�õ�ͼ��
        	//Ϊ��֡�����ͼ��
            pKF->ReplaceMapPointMatch(mit->second, pMP);
			//Ϊ��ͼ����ӹؼ�֡
            pMP->AddObservation(pKF,mit->second);
        }
        else
        {	
        	//����õ�ͼ���Ѿ�����֡�۲⵽
        	//�ͻ����ì�ܾ���һ����ͼ���Ӧ�����ؼ���
        	//��������û���滻������ֱ��ɾ��
            pKF->EraseMapPointMatch(mit->second);
        }
    }
	//���¹۲������
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
	//������д����Ե�������
    pMP->ComputeDistinctiveDescriptors();

	//�ӵ�ͼ��ɾ���õ�ͼ��
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


//������д����������
//����һ����ͼ��ᱻ�ܶ�ؼ�֡�۲⵽������ڲ���ؼ�֡��
//��Ҫ�ж��Ƿ���µ�ǰ��ͼ��������������
//�Ȼ�õ�ǰ��ͼ������������ӣ�Ȼ�����������֮�����������
//��õ������Ӻ�������������Ӧ������С�ľ�����ֵ
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

	//�����۲⵽��3d ������йؼ�֡����ȡorb ������
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;

        if(!pKF->isBad())
			//��orb �����Ӳ��뵽vDescriptors
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second));
    }

    if(vDescriptors.empty())
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size();

	//����ÿ������������ֱ�ӵĺ�������
    float Distances[N][N];
	//vector<vector<float>>  Distances;
	//Distances.resize(N,vector<float>(N,0));
    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
        	//��������������ֱ�ӵĺ�������
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]);
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    int BestIdx = 0;
	//��ȡ�м������С��������
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
		//��õ������Ӿ��Ǻ����������ӵľ�����С
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

//���ҵ�ͼ���Ƿ񱻸ùؼ�֡�۲⵽
bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

//���µ�ͼ��ƽ���۲ⷽ��͹۲����ķ�Χ
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
		//��ȡ�۲⵽��3D ������й۲�֡
        observations=mObservations;
		//�۲⵽�õ�Ĳο��ؼ�֡
        pRefKF=mpRefKF; 
		//3D ������������ϵ�е�λ��
        Pos = mWorldPos.clone();
    }

    if(observations.empty())
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first; //��ȡ�ؼ�֡
        cv::Mat Owi = pKF->GetCameraCenter(); //��ȡ�ؼ�֡������λ��
        cv::Mat normali = mWorldPos - Owi; //��ȡ��3D ��������������λ�õ�����
        //�����й۲�֡�Ըõ�Ĺ۲ⷽ���һ��ΪΪ��λ�����������
        normal = normal + normali/cv::norm(normali);
        n++;
    }

	//�ο��ؼ�֡���ָ��3D ��ķ���
    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
	//�õ㵽�ο��ؼ�֡����ľ���
    const float dist = cv::norm(PC);
	//��ȡ�ο�֡�ؼ�����ͼ��������ĵڼ���
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
	//�������ű���
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
	//��ȡͼ�����������
    const int nLevels = pRefKF->mnScaleLevels;

    {
        unique_lock<mutex> lock3(mMutexPos);
		//�۲⵽�õ�������ֵ
        mfMaxDistance = dist*levelScaleFactor;
		//�۲⵽�õ������Сֵ
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
		//��ȡƽ���۲ⷽ��
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

//ͨ�����룬Ԥ���ڵ�ͼ���ڹؼ�֡��ͼ�����������һ��
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
		//����
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
