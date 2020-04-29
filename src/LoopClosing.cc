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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        //�ջ�������mlpLoopKeyFrameQueue ��Ϊ��
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            //���loop ��ѡ֡����鹲��֡��������
            if(DetectLoop())
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               //���㵱ǰ֡�ͱջ�֡��sim3 �任
               if(ComputeSim3())
               {
                   // Perform loop fusion and pose graph optimization
                   //���л�·�ںϺ�λ��ͼ�Ż�
                   CorrectLoop();
               }
            }
        }       

		//���ܵ�tracking �̷߳�����������Ϣ
        ResetIfRequested();

		//slam ϵͳ�ر�����
        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

//��loopClosing �̲߳����µĹؼ�֡
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

//���ջ��������Ƿ�Ϊ��
bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
		//�Ӷ�����ȡ��һ���ؼ�֡
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
		//ɾ�������еĹؼ�֡
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        //���ùؼ�֡��ɾ����־����ֹ�ڴ�������б�locamapping �߳�ɾ��
        mpCurrentKF->SetNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    //��ǰ�ؼ�֡�����ϴιؼ�֡�ļ������10 ���ؼ�֡�������лػ����
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
    	//�Ѹ�ֱ֡�Ӽ���ؼ�֡���ݿ�
        mpKeyFrameDB->add(mpCurrentKF);
		//���õ�ǰ֡ɾ��״̬
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    //���㵱ǰ�ؼ�֡���乲�ӹؼ�֡����͵÷֣�
    //���������͵÷ֿ����ڱջ�����ѡ֡�д���ɸѡ�������뵱ǰ֡�γɱջ���֡
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    //�������й��ӹؼ�֡�����㵱ǰ�ؼ�֡��ÿ�����ӹؼ�֡���Ӿ��ʴ����ƶȵ÷�
    //���õ���ͷ���
    //��ȡ��ǰ�ؼ�֡�Ĺ��ӹؼ�֡
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
	//��ȡ��ǰ֡���Ӿ��ʴ���������
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
	//�������еĹ���֡
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
    	//��ȡ�ù���֡
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
		//��ȡ�ù���֡���Ӿ��ʴ�
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

		//�����Ӿ��ʴ������ƶȵ÷�
        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

		//��ȡ��͵÷�
        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    //�ڱջ�������ҵ���ùؼ�֡���ܱջ��Ĺؼ�֡
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    //���û���ҵ���ѡ֡
    if(vpCandidateKFs.empty())
    {
    	//�ѵ�ǰ֡����ؼ�֡���ݿ�
    	//Ϊÿ���Ӿ��ʴ���word ��ӹؼ�֡
        mpKeyFrameDB->add(mpCurrentKF);
		//�����������
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
	//�ں�ѡ֡�м����������Եĺ�ѡ֡
	//1��ÿ����ѡ֡�����Լ������Ĺؼ�֡����һ���Ӻ�ѡ��spCandidateGroup
	//2������Ӻ�ѡ����ÿһ���ؼ�֡�Ƿ���������ԣ�������ڣ��ͰѸ��Ӻ�ѡ����뵱ǰ������vCurrentConsistentGroups


	//����ɸѡ��õ��ջ�֡
    mvpEnoughConsistentCandidates.clear();

	//��ǰ������
    vector<ConsistentGroup> vCurrentConsistentGroups;
	//ͳ�Ƶ�ǰ��ѡ���֮ǰ�����������Ƿ������ı�־����
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
	//������ǰ֡�ջ��ĺ�ѡ�ؼ�֡
	//���ܺ�ѡ֡���ɵĺ�ѡ�飬��֮ǰ���������Ƿ���������������������飬ֻ��Ȩ�ز�һ��
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
    	//��ȡ�ؼ�֡
        KeyFrame* pCandidateKF = vpCandidateKFs[i];
		//���Լ����Լ�������֡����һ���Ӻ�ѡ��
		//��ȡ��ؼ�֡������֡
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
		//�Ѹùؼ�֡���������ؼ�֡����
        spCandidateGroup.insert(pCandidateKF);

		//�����Ա�־
        bool bEnoughConsistent = false;
		//
        bool bConsistentForSomeGroup = false;
		//���ùؼ�֡��֮ǰ�����������������
		//������������
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
        	//ȡ��һ��֮ǰ����������
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

			//����ÿ���Ӻ�ѡ�飬����ѡ����ÿһ���ؼ�֡�������������Ƿ����
			//�����һ֡��ͬ�������Ӻ�ѡ����֮ǰ�����������У���ô�Ӻ�ѡ�����������������
            bool bConsistent = false;
			//�����Ӻ�ѡ��
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
            	//��֮ǰ�����������в��ң����������Ƿ���ͬһ���ؼ�֡
                if(sPreviousGroup.count(*sit))
                {
                	//�ҵ����������Ա�־
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

			//�Ƿ�����
            if(bConsistent)
            {
            	//��ȡ֮ǰ�������� ��������Ȩ��
                int nPreviousConsistency = mvConsistentGroups[iG].second;
				//��ǰ��ѡ��������Ȩ�ؼ�1
                int nCurrentConsistency = nPreviousConsistency + 1;
				//��ǰ��ѡ��͸������������Ƿ��Ѿ��������
                if(!vbConsistentGroup[iG])
                {
                	//û�б������
                	//����������
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
					//���뵽��ǰ������
                    vCurrentConsistentGroups.push_back(cg);
					//����������־����ֹ�ظ�����
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
				//mnCovisibilityConsistencyTh  = 3 ,��������ֵ
				//����Ȩ�ش���3 �� �㹻�����Ա�־λ��û����
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                	//�Ѹú�ѡ�ؼ�֡�����������㹻�ĺ�ѡ�ؼ�֡������
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
					//���������㹻��־
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        //���֮ǰ�������� ����û���ҵ���ǰ��ѡ���е�֡����ʾ������
        if(!bConsistentForSomeGroup)
        {
        	//���������
        	//���ø�֡���ɵĺ�ѡ��������Ϊ0
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
			//���뵽��ǰ������
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    //����������
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    //Ϊÿһ���Ӿ��ʴ���word ��ӹؼ�֡
    mpKeyFrameDB->add(mpCurrentKF);

	//���û���ҵ������Դ���2 �ĺ�ѡ֡
    if(mvpEnoughConsistentCandidates.empty())
    {
    	//���õ�ǰ֡ɾ����־
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

	//���������ô�����ߵ���Ŷ
    mpCurrentKF->SetErase();
    return false;
}

//���㵱ǰ֡�ͱջ�֡��sim3 �任
//1��ͨ���Ӿ��ʴ����������ӵ�ƥ�䣬��������������Լ������ǰ֡��ջ�֡��sim3 
//2�����ݹ��Ƶ�sim3, ��3D�����ͶӰ�ҵ������ƥ�䣬ͨ���Ż��ķ����������ȷ��sim3
//3�����ջ�֡�Լ��ջ�֡�����Ĺؼ�֡�ĵ�ͼ���뵱ǰ֡�ĵ����ƥ��
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
	//��ȡ��ǰ֡�ıջ���ѡ֡
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    //����ORBmatcher ����
    ORBmatcher matcher(0.75,true);

	//ÿһ����ѡ֡����һ��Sim3Solvers
    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

	//��ͼ������
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

	//�������к�ѡ֡
    for(int i=0; i<nInitialCandidates; i++)
    {
    	//��ȡ��ѡ֡
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        //���ò�ɾ����־�������ڴ����ʱ���֡��ɾ�������¿�ָ��
        pKF->SetNotErase();

		//��֡�ǲ��ǻ���
        if(pKF->isBad())
        {
        	//ֱ�ӽ���֡����
            vbDiscarded[i] = true;
            continue;
        }

		//����ǰ֡�ͱջ���ѡ֡����ƥ��
		//ͨ���Ӿ��ʴ��ļ��٣����㵱ǰ֡�ͺ�ѡ֡��ƥ��������
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

		//ƥ�䵽������ĸ���
        if(nmatches<20)
        {
        	//С��20 ���������ú�ѡ֡
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
        	//����sim3 �����
        	//mbFixScale Ϊtrue ����6DoF �Ż�( ˫Ŀrgbd ) , �����false ����7DoF �Ż�( ��Ŀ )
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
			//����������, ����20 ���ڵ㣬300 �ε���
            pSolver->SetRansacParameters(0.99,20,300);
			//�Ѹú�ѡ֡��sim3 ����������vpSim3Solvers ������
            vpSim3Solvers[i] = pSolver;
        }

		//�㹻ƥ��ĺ�ѡ֡++
        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    //һֱѭ�����еĺ�ѡ֡��ÿ����ѡ֡����5 �Σ����5�ε�����ò���������ͻ���һ����ѡ֡
    //ֱ����һ����ѡ֡�״ε����ɹ��� ����ĳ����ѡ֡�ܵĵ��������������ƣ�ֱ�ӽ����޳�
    while(nCandidates>0 && !bMatch)
    {
    	//�������йؼ���ѡ֡
        for(int i=0; i<nInitialCandidates; i++)
        {
        	//���ùؼ���ѡ֡�Ƿ��Ǳ��޳���
            if(vbDiscarded[i])
                continue;

			
			//��ȡ�ؼ���ѡ֡
            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

			//��ȡ��֡�͵�ǰ֡��sim3 ����������
            Sim3Solver* pSolver = vpSim3Solvers[i];
			//������5 �Σ� ����sim3  �任��t12
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            //û������ϸ��sim3 �任�� �ú�ѡ֡�ߵ�
            if(bNoMore)
            {
            	//�޳���֡
                vbDiscarded[i]=true;
				//��ѡ֡��1
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            //sim3 ���ɹ�
            if(!Scm.empty())
            {
            	//Ϊ�ú�ѡ֡�ĵ�ͼ�����ռ�
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
				//��������sim3 �������ڵ�
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                	//�����ڵ�ĵ�ͼ��
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

				//[sR  t; 0 1]
				//��ȡ��ѡ֡����ǰ֡��R12
                cv::Mat R = pSolver->GetEstimatedRotation();
				//��ѡ֡����ǰ֡��t12 , �����к�ѡָ֡��ǰ֡
                cv::Mat t = pSolver->GetEstimatedTranslation();
				//��ѡ֡����ǰ֡�任�߶�s12
                const float s = pSolver->GetEstimatedScale();
				//���Ҹ����ƥ�䣬�ɹ��ıջ�ƥ����Ҫ�����㹻���ƥ����������
				//֮ǰʹ��SearchByBow ����������ƥ��ʱ����©ƥ��
				//ͨ��sim3 �任��ȷ��pkf1����������pkf2�еĴ�������ͬ��
				//ȷ��pkf2 ����������pkf1 �еĴ�������
				//�ڸ�������ͨ�������ӽ���ƥ�䲶��pkf1 �� pkf2 ֮ǰ©ƥ��������㣬����vpMapPointMatches
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

				//sim3 �Ż���ֻҪ��һ����ѡ֡ͨ��sim3 ��������Ż�����������ֹͣ��������ѡ֡���ж�
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
				//�Ż���ǰ֡�ͺ�ѡ֡��Ӧ��ͼ���sim3, �õ��Ż������gScm
				//����chi2 ������ֵ
				const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                //�õ�ƥ����ڵ��������20
                if(nInliers>=20)
                {
                    bMatch = true;
					//�ҵ����뵱ǰ֡ƥ��ıջ�֡
                    mpMatchedKF = pKF;
					//�õ�����������ϵ���ú�ѡ֡��sim3 �任��Scale = 1
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
					//�õ�g2o�Ż������������ϵ����ǰ֡��sim3�任
					mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
					//ֻҪ��һ����ѡ֡ͨ��sim3 ��������Ż���������ֹͣ��������ѡ֡���ж�
                    break;
                }
            }
        }
    }

	//û��һ���ջ�ƥ���ѡ֡ͨ��sim3 ��������Ż�
    if(!bMatch)
    {
    	//���ú�ѡ֡�Ŀ���ɾ������
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
		//���õ�ǰ֡�Ŀ���ɾ������
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    //��ȡƥ��ɹ��ջ�֡�Ĺ���֡
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();

	//��ǰ֡����Ϊ�ջ�֡�Ĺ���֡
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
	//�����ջ�֡�����й���֡����ӵ�ͼ�㵽mvpLoopMapPoints
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
    	//��ȡ�ؼ�֡
        KeyFrame* pKF = *vit;
		//��ȡ�ؼ�֡�ĵ�ͼ��
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
		//������ͼ��
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                	//�ѵ�ͼ�����mvpLoopMapPoints
                    mvpLoopMapPoints.push_back(pMP);
					//��ֹ�ظ����
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    //���ջ�ƥ���Ϲؼ�֡�Լ������ؼ�֡�ĵ�ͼ��ͶӰ����ǰ�ؼ�֡����ͶӰƥ��
    //����ͶӰ���Ҹ����ƥ�䣬�ɹ��ıջ�ƥ����Ҫ�����㹻���ƥ����������
    //����sim3�任����ÿ����ͼ��ͶӰ����ǰ֡�ϣ������ݳ߶�ȷ��һ����������
    //���ݸõ�ͼ�����������������ڵ����������ƥ�䣬���ƥ�����С��50��
    //��ƥ��ɹ������µ�ǰ֡ƥ��ĵ�ͼ��
    //��ǰ֡ƥ��ĵ�ͼ���뵱ǰ��ͼ����ܳ��ֳ�ͻ�������ں�
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
	//����ƥ������ͳ��ƥ��ɹ�����
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

	//ƥ��ɹ���������40 ƥ��ɹ�
    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
			//����ƥ��ɹ��ĺ�ѡ֡�͵�ǰ֡�������ѡ֡����ɾ����־
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
    	//ƥ��ʧ��
        for(int i=0; i<nInitialCandidates; i++)
			//���ú�ѡ֡��ɾ����־
            mvpEnoughConsistentCandidates[i]->SetErase();
		//���õ�ǰ֡ɾ����־
        mpCurrentKF->SetErase();
        return false;
    }

}

//�ջ�
// 1��ͨ�����sim3 �Լ����λ�˹�ϵ��
//�����뵱ǰ֡�����Ĺؼ�֡λ���Լ��ؼ�֡�۲⵽�ĵ�ͼ��λ��
//2�����ջ�֡�Լ��ջ�֡�����Ĺؼ�֡�ĵ�ͼ����뵱ǰ֡�����Ĺؼ�֡�ĵ�ͼ�����ƥ��
//3��ͨ����ͼ���ƥ���ϵ������Щ֮֡������ӹ�ϵ��������֡�ĸ���
//4����λ��ͼ�����Ż�����ͼ���λ��������Ż����λ������Ӧ�ĵ���
//5�������߳̽���ȫ��ba �Ż�
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    //����ֲ���ͼֹͣ����ֹ
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    //�����������ȫ���Ż�
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
		//ֹͣȫ��BA �Ż�
        mbStopGBA = true;

		//ȫ���Ż�idx
        mnFullBAIdx++;
		//ֹͣȫ���Ż��߳�
        if(mpThreadGBA)
        {
        	//�����̣߳����ڵײ������Դ
            mpThreadGBA->detach();
			//ɾ���߳�ָ��
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    //�ȴ����ؽ�ͼֹͣ
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    //���µ�ǰ֡�Ĺ���֡Ȩ��
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    //��ȡ��ǰ֡�Ĺ��ӹؼ�֡
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
	//���뵱ǰ֡
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

	//CorrectedSim3 ����sim3 �ջ��Ż����λ��
	//NonCorrectedSim3 ����û�бջ��Ż����λ��
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
	//���浱ǰ֡��sim3 �任
    CorrectedSim3[mpCurrentKF]=mg2oScw;
	//se3 ��Ⱥ����ת����
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
		//ͨ��λ�˴������õ�sim3 �����������뵱ǰ֡�����ؼ�֡��λ��

		//�������еĹ���֡
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
        	//��ȡ�ؼ�֡
            KeyFrame* pKFi = *vit;

			//��ȡ�ؼ�֡��λ��
            cv::Mat Tiw = pKFi->GetPose();

			//�����ڵ�ǰ֡����ǰ֡��ǰ���Ѿ�����
            if(pKFi!=mpCurrentKF)
            {
            	//�õ���ǰ֡��������֡����Ա任
                cv::Mat Tic = Tiw*Twc;
				//��ȡ��R
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
				//��ȡ��t
                cv::Mat tic = Tic.rowRange(0,3).col(3);

				//���������֡��sim3 �任����
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
				//��ǰ֡��λ�˹̶������������Ĺؼ�֡������Թ�ϵ�õ�sim3 ������λ��
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                //�õ��ջ��Ż�������ؼ�֡��λ��
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

			//��λ������ȡR
            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
			//��λ������ȡt
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
			//����sim3 �任����
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            //��Ӹ�����֡û�н��бջ��Ż���λ��
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        //�õ���������֡��λ�˺�������Щ�ؼ�֡�ĵ�ͼ��
        //�����������λ��
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
        	//��ȡ���֡
            KeyFrame* pKFi = mit->first;
			//��ȡsim3 λ��
            g2o::Sim3 g2oCorrectedSiw = mit->second;
			//��ȡsim3 λ�˵������
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

			//��ȡ��Ӧ��û���Ż���λ��
            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

			//��ȡ�ùؼ�֡�ĵ�ͼ��
            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
			//������ͼ��
			//for(vector<MapPoint*>::iterator imp = vpMPsi.begin(), endMPi = vpMPsi.end(); iMP != endMPi, iMP++)
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
            	//MapPoint* pMPi = *vpMPsi;
            	//��ȡ��ͼ��
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
				//��ֹ�ظ�����
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                //��ȡ��ͼ��3d ����
                cv::Mat P3Dw = pMPi->GetWorldPos();
				//�ѵ�ͼ�������ת����sim3 ʹ�õľ���
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
				//g2oSiw.map(eigP3Dw)  ��eigP3Dw ӳ�䵽��֡û���Ż����������ϵ��
				//g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw))  Ȼ���ڷ�ӳ�䵽У׼�����������ϵ
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

				//ת��Ϊ������������3D  ��
                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
				//����������ĵ�ͼ��
                pMPi->SetWorldPos(cvCorrectedP3Dw);
				//�����Ѿ�������־
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
				//�����ο�֡Ϊ������֡
                pMPi->mnCorrectedReference = pKFi->mnId;
				//���µ�ͼ���ƽ������͹۲ⷶΧ
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            //��sim3 ת����se3  , ���ݸ��µ�sim3, ���¹ؼ�֡��λ��
            //����R
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
			//����t
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
			//��ȡs
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]


			//��sim3 ת����se3, ����λ��
            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

			//�����������λ��
            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            //���¸�֡�Ĺ���֡����
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        //��鵱ǰ֡�ĵ�ͼ����ջ�ƥ��֡�ĵ�ͼ���Ƿ���ڳ�ͻ��
        //�Գ�ͻ�ĵ�ͼ�����������
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
            	//��ȡ�ջ��ǵ�ǰ֡ƥ�䵽�ĵ�ͼ��
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
				//��ȡ��ǰ֡��Ӧ�ĵ�ͼ��
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
				//�����ǰ�������ͼ�����
                if(pCurMP)
					//�ñջ��ĵ�ͼ���滻��ͼ��
                    pCurMP->Replace(pLoopMP);
                else
                {
                	//��ǰ�����㲻���ڵ�ͼ��
                	//�ѱջ��ĵ�ͼ�����λ��������ĵ�ͼ��
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
					//���¸õ�ͼ��Ĺ۲�֡����Ӧ��������
                    pLoopMP->AddObservation(mpCurrentKF,i);
					//���¼����ͼ������������
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //ͨ�����ջ�ʱ�����ؼ�֡�ıջ���ͼ��ͶӰ����Щ�ؼ�֡�У����е�ͼ������滻
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    //���µ�ǰ�ؼ�֮֡��Ĺ���������ϵ���õ���ջ�ʱ��ͼ���ں϶��µõ������ӹ�ϵ
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

	//������ǰ֡�����ؼ�֡��һ������
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
    	//��ȡ�ؼ�֡
        KeyFrame* pKFi = *vit;
		//��ȡ�ùؼ�֡�Ĺ���֡����������
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        //���µ�ǰ֡����֡������
        pKFi->UpdateConnections();
		//ȡ����֡���º�����ӹ�ϵ
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
		//�����ӹ�ϵ��ȥ���ջ�֮ǰ���� �����ӹ�ϵ��ʣ�µ����Ӿ����ɱջ��õ������ӹ�ϵ
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
        	//ɾ��֮ǰ�����ӹ�ϵ
            LoopConnections[pKFi].erase(*vit_prev);
        }
		//�����ӹ�ϵ��ȥ��֮ǰ��һ�����ӹ�ϵ��ʣ�µ����Ӿ����ɱջ��õ������ӹ�ϵ
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    //��ͼ�Ż�
    //����λ���Ż���LoopConnections ���γɱջ��������ɵ����ӹ�ϵ������������7 �е�ǰ֡��ջ�ƥ��֮֡������ӹ�ϵ
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    //��ӵ�ǰ֡��ջ�ƥ��֮֡��ıߣ�������ӹ�ϵ���Ż�
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    //�½�һ���߳�����ȫ��BA �Ż�
    //OptimizeEssentialGraph ֻ���Ż���һЩ��Ҫ�ؼ�֡��λ�ˣ��������ȫ��BA ����ȫ���Ż�����λ�˺͵�ͼ��
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
	//�½�һ���߳̽���ȫ���Ż�
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

	//�������һ�αջ�֡
    mLastLoopKFid = mpCurrentKF->mnId;   
}

//ͨ�����ջ�ʱ�����ؼ�֡��ͼ��ͶӰ����Щ�ؼ�֡�У����е�ͼ��ļ����滻
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
	//����ORBmatcher ����
    ORBmatcher matcher(0.8);

	//�����ջ������Ĺؼ�֡
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
    	//��ȡ�ؼ�֡
        KeyFrame* pKF = mit->first;

		//��ȡ��֡sim3 λ��
        g2o::Sim3 g2oScw = mit->second;
		//ת��Ϊcv �ľ�����ʽ
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

		//���ջ�����֡�ĵ�ͼ������任����֡����ϵ��Ȼ��ͶӰ������ͻ���ں�
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
		//�������бջ���ͼ��
        for(int i=0; i<nLP;i++)
        {
        	//��ȡ��Ҫ�滻�ĵ�ͼ��
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
            	//�ñջ��ĵ�ͼ���滻��֡�ĵ�ͼ��
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
    	//��ջػ����
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
