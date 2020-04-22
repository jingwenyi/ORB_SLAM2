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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame(Frame &F, 					//���뵱ǰ֡
						Map *pMap, 					//�ؼ�֡��ͼ
						KeyFrameDatabase *pKFDB):	//�ؼ�֡���ݿ�
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{
    mnId=nNextId++;

	//���������С
    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

	//����λ��
    SetPose(F.mTcw);    
}

void KeyFrame::ComputeBoW()
{
    if(mBowVec.empty() || mFeatVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw;

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

//��ȡ�������
cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}


//��ȡ��Ⱥ����ת����
cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

//��ȡ��Ⱥ��ƽ�ƾ���
cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(!mConnectedKeyFrameWeights.count(pKF))
            mConnectedKeyFrameWeights[pKF]=weight;
        else if(mConnectedKeyFrameWeights[pKF]!=weight)
            mConnectedKeyFrameWeights[pKF]=weight;
        else
            return;
    }

    UpdateBestCovisibles();
}

//���¹���֡
void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections);
	//����������Ŀ����Ϊ������
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
	//�Ѹùؼ�֡���ӵ�ͼȫ���ŵ�������
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first));

	//����������С����
    sort(vPairs.begin(),vPairs.end());
	//��������ֵ������У��Ӵ�С
    list<KeyFrame*> lKFs;
    list<int> lWs;
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
    	//��С�ķŵ�ǰ�� push_front
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

	//��list ֵ�ŵ�vector ��
    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());    
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

//��ȡ�ؼ�֡�Ĺ��ӹؼ�֡
vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

//��ȡ�������֡
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);
    if((int)mvpOrderedConnectedKeyFrames.size()<N)
        return mvpOrderedConnectedKeyFrames;
    else
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

//��ȡ�����ؼ�֡������Ȩ��
int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
	//�����Ƿ���pKF �ؼ�֡
    if(mConnectedKeyFrameWeights.count(pKF))
		//��ȡ����Ȩ��
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

//Ϊ�ؼ�֡��ӵ�ͼ�ϺͶ������������Ҫ
void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
	//��ӵ�ͼ������������
    mvpMapPoints[idx]=pMP;
}

//ɾ��ƥ��ĵ�ͼ��
void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


//Ϊ֡�����ͼ��
void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures);

    int nPoints=0;
    const bool bCheckObs = minObs>0;
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(bCheckObs)
                {
                    if(mvpMapPoints[i]->Observations()>=minObs)
                        nPoints++;
                }
                else
                    nPoints++;
            }
        }
    }

    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

//����ͼ������
//��ǰ֡�Ĺ���֡�����չ���Ȩ�ؽ�������
void KeyFrame::UpdateConnections()
{
    map<KeyFrame*,int> KFcounter;

    vector<MapPoint*> vpMP;

    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
		//��ȡ�� �ؼ�֡�ؼ�������ĵ�ͼ��
        vpMP = mvpMapPoints;
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    //�������еĵ�ͼ��
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit; //��ȡ��ͼ��

        if(!pMP)//��ͼ�㲻����
            continue;

		//��Ч��ͼ��
        if(pMP->isBad())
            continue;

		//��ȡ�۲⵽�õ�ͼ������йؼ�֡
        map<KeyFrame*,size_t> observations = pMP->GetObservations();

		//
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
        	//��ȥ�����Լ����Լ����㹲��
            if(mit->first->mnId==mnId)
                continue;
			//ͳ��ÿһ���ؼ��������ĵ�ͼ�����
            KFcounter[mit->first]++;
        }
    }

    // This should not happen
    if(KFcounter.empty())
        return;

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15; //��ֵ

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
	//����ÿ���ؼ�֡�۲�ĵĵ�ͼ���������
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax)
        {
            nmax=mit->second;
			//�ҵ���ӦȨ�����Ĺؼ�֡
            pKFmax=mit->first;
        }
		//��ӦȨ�ش�����ֵ
        if(mit->second>=th)
        {
        	//�Թؼ�֡��Ȩ�ؽ�������
            vPairs.push_back(make_pair(mit->second,mit->first));
			//���������ؼ�֡�͵�ǰ֡������Ȩ��
            (mit->first)->AddConnection(this,mit->second);
        }
    }

	//���û�г���Ȩ�أ����Ȩ�����Ĺؼ�֡��������
    if(vPairs.empty())
    {
        vPairs.push_back(make_pair(nmax,pKFmax));
        pKFmax->AddConnection(this,nmax);
    }

	//����Ĺؼ��������Ӧ��Ȩ�ؽ�������
	//�ɴ�С����
    sort(vPairs.begin(),vPairs.end());
    list<KeyFrame*> lKFs;
    list<int> lWs;
	//��Ȩ�غͶ��ڵ�֡�ֱ�����б�
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        //���µ�ǰ֡��������֡������Ȩ��
        mConnectedKeyFrameWeights = KFcounter;
		//��������֡������˳��
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
		//��������֡Ȩ��˳��
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

		//����������������
        if(mbFirstConnection && mnId!=0)
        {
        	//��ʼ���ؼ�֡�ĸ��ڵ�ΪΪ���ӳ̶���ߵ��Ǹ��ؼ�֡
            mpParent = mvpOrderedConnectedKeyFrames.front();
			//�ѵ�ǰ֡����Ϊ���ӽڵ㣬����˫���ϵ
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
		//mbNotErase  ����Ϊtrue�� ��ʾ�ùؼ�֡�Ѿ�ɾ������ʵ���ﻹû��ɾ��
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

	//�������Ĺؼ�֡ɾ�����Լ�������
    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

 	//�������еĵ�ͼ��
    for(size_t i=0; i<mvpMapPoints.size(); i++)
		//������ڵ�ͼ��
        if(mvpMapPoints[i])
			//ɾ��õ�ͼ��Ĺ۲�֡
            mvpMapPoints[i]->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);
		//����Լ��������ؼ�֡����ϵ
        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        //����һ������
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        //�������ؼ�֡�к��ӹؼ�֡�����������Լ����У��Ͻ��Ҹ��ؼ�֡
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

			//����ÿһ���ӹؼ�֡�������Ǹ���ָ��ĸ��ؼ�֡
			//���ӽڵ�Ĺ���֡�͸��ڵ����Ƚϣ��ҳ��ӽڵ�͸��ڵ����������֡����Ϊ���ӽڵ�ĸ��ڵ�
			//ɾ�����ӽڵ㣬�Ѹ��ӽڵ���ӵ������ӽڵ�ı�ѡ���ڵ���,
			//����Ϊ�����ӽڵ�����ͬ�ķ�ʽ�ڱ�ѡ���ڵ��в������ƥ��
			//ֱ�������ӽڵ�ȫ���ҵ����ڵ㣬����һ���ֽڵ��Ҳ������ڵ�Ϊֹ
			//����Ҫ�ǵ�һ�θ��ڵ�ı�ѡֻ��һ����ʱ�򣬲���ʧ���ˣ������ӽڵ㣬
			//ֱ�Ӽ̳и��ڵ�ĸ��ڵ�
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
            	//��ȡ�ؼ�֡
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                //��ȡ�ؼ�֡�Ĺ��ӹؼ�֡
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
				//�ӹؼ�֡����ÿһ�����乲�ӵĹؼ�֡
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                	//����Ҫɾ���ؼ�֡��Ӧ�ĸ��ڵ�
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                    	//�ӹؼ�֡�Ĺ���֡���븸�ؼ�֡���
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                        	//��ȡ�ӹؼ�֡�͸��ؼ�֡������Ȩ��
                            int w = pKF->GetWeight(vpConnected[i]);
							//�ҳ�����Ȩ�����Ĺؼ�֡
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

			//����ҵ�
            if(bContinue)
            {
            	//�����ӹؼ�֡�ĸ��ؼ�֡
                pC->ChangeParent(pP);
				//���ӽڵ��ҵ����µĸ��ڵ㣬�������˸��ڵ㣬
				//��ô���ӽڵ���������Ϊ�����ӽڵ�ı�ѡ���ڵ�
                sParentCandidates.insert(pC);
				//�ڵ�ǰ�ؼ���ɾ�����ӹؼ�֡
                mspChildrens.erase(pC);
            }
            else
				
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        //��������ӽڵ�û���ҵ��µĸ��ڵ�
        if(!mspChildrens.empty())
			//��������û���ҵ����ڵ���ӽڵ�
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
            	//ֱ�ӰѸ��ڵ�ĸ��ڵ���Ϊ�Լ��ĸ��ڵ�
                (*sit)->ChangeParent(mpParent);
            }

		//�ø��ڵ�ɾ���ӽڵ�
        mpParent->EraseChild(this);
		//���㵱ǰ֡����ڸ��ڵ��λ��
		//RR 
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


	//�Ӿֲ���ͼɾ���ùؼ�֡
    mpMap->EraseKeyFrame(this);
	//�ؼ�֡���ݿ���ɾ���ùؼ�֡
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
}

//��ǰ֡ɾ���봫��ֱ֡�ӵ�����
void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
		//�����Ƿ�����ùؼ�֡
        if(mConnectedKeyFrameWeights.count(pKF))
        {
        	//ɾ���ùؼ�֡
            mConnectedKeyFrameWeights.erase(pKF);
			//���¹���֡��־
            bUpdate=true;
        }
    }

	//���¹���֡
    if(bUpdate)
        UpdateBestCovisibles();
}

//���ҳ��ùؼ�֡��(x,y) Ϊ���ģ�r Ϊ2r  Ϊ�ߵĴ����еĹؼ���
vector<size_t> KeyFrame::GetFeaturesInArea(const float &x,   //��ͶӰ�����x ����
												const float &y,  //��ͶӰ�����y ����
												const float &r) const   //�����뾶
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

	//���������������Χ
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

	//����������������Χ
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

	//���ҷ�Χ�ڵĹؼ���
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
			//���������йؼ��������
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

				//������
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

//������ǰ�ؼ�֡�ĳ�����ȣ�q=2 ��ʾ��ֵ
float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
		//��ȡ��ͼ������
        vpMapPoints = mvpMapPoints;
		//��ȡ��ǰ֡λ��
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
	//��ȡ�ڶ��е�0 ��3 Ԫ�� 1x3 ����
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
	//����ת����3x1����
    Rcw2 = Rcw2.t();
	
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
        	//��ȡ��ͼ��
            MapPoint* pMP = mvpMapPoints[i];
			//��ȡ��ͼ���3 ά����
            cv::Mat x3Dw = pMP->GetWorldPos();
			//�����
            float z = Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

} //namespace ORB_SLAM
