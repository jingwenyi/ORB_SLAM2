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
        //闭环检测队列mlpLoopKeyFrameQueue 不为空
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            //检测loop 候选帧，检查共视帧的连续性
            if(DetectLoop())
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               //计算当前帧和闭环帧的sim3 变换
               if(ComputeSim3())
               {
                   // Perform loop fusion and pose graph optimization
                   //进行回路融合和位姿图优化
                   CorrectLoop();
               }
            }
        }       

		//接受到tracking 线程发来的重置消息
        ResetIfRequested();

		//slam 系统关闭请求
        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

//给loopClosing 线程插入新的关键帧
void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

//检查闭环检测队列是否为空
bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
		//从队列中取出一个关键帧
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
		//删除队列中的关键帧
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        //设置关键帧不删除标志，防止在处理过程中被locamapping 线程删除
        mpCurrentKF->SetNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    //当前关键帧距离上次关键帧的间隔不到10 个关键帧，不进行回环检测
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
    	//把该帧直接加入关键帧数据库
        mpKeyFrameDB->add(mpCurrentKF);
		//设置当前帧删除状态
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    //计算当前关键帧和其共视关键帧的最低得分，
    //利用这个最低得分可以在闭环检测候选帧中大致筛选出可能与当前帧形成闭环的帧
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    //遍历所有共视关键帧，计算当前关键帧与每个共视关键帧的视觉词袋相似度得分
    //并得到最低分数
    //获取当前关键帧的共视关键帧
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
	//获取当前帧的视觉词袋特征向量
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
	//遍历所有的共视帧
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
    	//获取该共视帧
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
		//获取该共视帧的视觉词袋
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

		//计算视觉词袋的相似度得分
        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

		//获取最低得分
        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    //在闭环检测中找到与该关键帧可能闭环的关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    //如果没有找到备选帧
    if(vpCandidateKFs.empty())
    {
    	//把当前帧加入关键帧数据库
    	//为每个视觉词袋的word 添加关键帧
        mpKeyFrameDB->add(mpCurrentKF);
		//清空子连续组
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
	//在候选帧中检测具有连续性的候选帧
	//1、每个候选帧将与自己相连的关键帧构成一个子候选组spCandidateGroup
	//2、检测子候选组中每一个关键帧是否存在连续性，如果存在，就把该子候选组放入当前连续组vCurrentConsistentGroups


	//最终筛选后得到闭环帧
    mvpEnoughConsistentCandidates.clear();

	//当前连续组
    vector<ConsistentGroup> vCurrentConsistentGroups;
	//统计当前候选组合之前的子连续组是否连续的标志向量
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
	//遍历当前帧闭环的后选关键帧
	//不管候选帧构成的候选组，跟之前子连续组是否连续，都会加入子连续组，只是权重不一样
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
    	//获取关键帧
        KeyFrame* pCandidateKF = vpCandidateKFs[i];
		//将自己和自己相连的帧构成一个子候选组
		//获取与关键帧相连的帧
        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
		//把该关键帧插入相连关键帧集合
        spCandidateGroup.insert(pCandidateKF);

		//连续性标志
        bool bEnoughConsistent = false;
		//
        bool bConsistentForSomeGroup = false;
		//检测该关键帧跟之前的子连续组的连续性
		//遍历子连续组
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
        	//取出一个之前的子连续组
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

			//遍历每个子候选组，检测候选组中每一个关键帧在子连续组中是否存在
			//如果有一帧共同存在于子候选组与之前的子连续组中，那么子候选组与该子连续组连续
            bool bConsistent = false;
			//遍历子候选组
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
            	//在之前的子连续组中查找，看两个组是否有同一个关键帧
                if(sPreviousGroup.count(*sit))
                {
                	//找到设置连续性标志
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

			//是否连续
            if(bConsistent)
            {
            	//获取之前子连续组 连续的性权重
                int nPreviousConsistency = mvConsistentGroups[iG].second;
				//当前候选组连续性权重加1
                int nCurrentConsistency = nPreviousConsistency + 1;
				//当前候选组和该子连续性组是否已经标记连接
                if(!vbConsistentGroup[iG])
                {
                	//没有标记连接
                	//制作连续组
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
					//插入到当前连续组
                    vCurrentConsistentGroups.push_back(cg);
					//设置连续标志，防止重复连接
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
				//mnCovisibilityConsistencyTh  = 3 ,连续性阈值
				//连续权重大于3 ， 足够连续性标志位还没设置
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                	//把该候选关键帧放入连续性足够的候选关键帧向量中
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
					//设置连续足够标志
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        //如果之前的子连续 组中没有找到当前候选组中的帧，表示不连续
        if(!bConsistentForSomeGroup)
        {
        	//如果不连续
        	//设置该帧构成的候选组连续性为0
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
			//插入到当前连续组
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    //更新连续组
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    //为每一个视觉词袋的word 添加关键帧
    mpKeyFrameDB->add(mpCurrentKF);

	//如果没有找到连续性大于2 的候选帧
    if(mvpEnoughConsistentCandidates.empty())
    {
    	//设置当前帧删除标志
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

	//这里代码怎么可能走的来哦
    mpCurrentKF->SetErase();
    return false;
}

//计算当前帧和闭环帧的sim3 变换
//1、通过视觉词袋加速描述子的匹配，利用随机采样粗略计算出当前帧与闭环帧的sim3 
//2、根据估计的sim3, 对3D点进行投影找到更多的匹配，通过优化的方法计算更精确的sim3
//3、将闭环帧以及闭环帧相连的关键帧的地图点与当前帧的点进行匹配
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
	//获取当前帧的闭环候选帧
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    //创建ORBmatcher 对象
    ORBmatcher matcher(0.75,true);

	//每一个候选帧都有一个Sim3Solvers
    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

	//地图点数组
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

	//遍历所有候选帧
    for(int i=0; i<nInitialCandidates; i++)
    {
    	//获取候选帧
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        //设置不删除标志，避免在处理的时候该帧被删除，导致空指针
        pKF->SetNotErase();

		//该帧是不是坏的
        if(pKF->isBad())
        {
        	//直接将该帧舍弃
            vbDiscarded[i] = true;
            continue;
        }

		//将当前帧和闭环候选帧进行匹配
		//通过视觉词袋的加速，计算当前帧和候选帧的匹配特征点
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

		//匹配到特征点的个数
        if(nmatches<20)
        {
        	//小于20 个，舍弃该候选帧
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
        	//构造sim3 求解器
        	//mbFixScale 为true 则用6DoF 优化( 双目rgbd ) , 如果是false 则用7DoF 优化( 单目 )
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
			//设计随机采样, 至少20 个内点，300 次迭代
            pSolver->SetRansacParameters(0.99,20,300);
			//把该候选帧的sim3 解析器放入vpSim3Solvers 向量中
            vpSim3Solvers[i] = pSolver;
        }

		//足够匹配的候选帧++
        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    //一直循环所有的候选帧，每个候选帧迭代5 次，如果5次迭代后得不到结果，就换下一个候选帧
    //直到有一个候选帧首次迭代成功， 或者某个候选帧总的迭代次数超过限制，直接将它剔除
    while(nCandidates>0 && !bMatch)
    {
    	//遍历所有关键候选帧
        for(int i=0; i<nInitialCandidates; i++)
        {
        	//检查该关键候选帧是否是被剔除的
            if(vbDiscarded[i])
                continue;

			
			//获取关键候选帧
            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

			//获取该帧和当前帧的sim3 解析器对象
            Sim3Solver* pSolver = vpSim3Solvers[i];
			//最多迭代5 次， 返航sim3  变换的t12
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            //没有求出合格的sim3 变换， 该候选帧踢掉
            if(bNoMore)
            {
            	//剔除该帧
                vbDiscarded[i]=true;
				//候选帧减1
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            //sim3 求解成功
            if(!Scm.empty())
            {
            	//为该候选帧的地图点分配空间
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
				//遍历所有sim3 求解出的内点
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
                	//保存内点的地图点
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

				//[sR  t; 0 1]
				//获取候选帧到当前帧的R12
                cv::Mat R = pSolver->GetEstimatedRotation();
				//候选帧到当前帧的t12 , 方向有候选帧指向当前帧
                cv::Mat t = pSolver->GetEstimatedTranslation();
				//候选帧到当前帧变换尺度s12
                const float s = pSolver->GetEstimatedScale();
				//查找更多的匹配，成功的闭环匹配需要满足足够多的匹配特征点数
				//之前使用SearchByBow 进行特征点匹配时会有漏匹配
				//通过sim3 变换，确定pkf1的特征点在pkf2中的大致区域，同理，
				//确定pkf2 的特征点在pkf1 中的大致区域
				//在该区域内通过描述子进行匹配捕获pkf1 和 pkf2 之前漏匹配的特征点，更新vpMapPointMatches
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

				//sim3 优化，只要有一个候选帧通过sim3 的求解与优化，就跳出来停止对其他候选帧的判读
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
				//优化当前帧和候选帧对应地图点的sim3, 得到优化后的量gScm
				//卡方chi2 检验阈值
				const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                // If optimization is succesful stop ransacs and continue
                //得到匹配的内点个数大于20
                if(nInliers>=20)
                {
                    bMatch = true;
					//找到了与当前帧匹配的闭环帧
                    mpMatchedKF = pKF;
					//得到从世界坐标系到该候选帧的sim3 变换，Scale = 1
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
					//得到g2o优化后从世界坐标系到当前帧的sim3变换
					mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
					//只要有一个候选帧通过sim3 的求解与优化，就跳出停止对其他候选帧的判断
                    break;
                }
            }
        }
    }

	//没有一个闭环匹配候选帧通过sim3 的求解与优化
    if(!bMatch)
    {
    	//设置候选帧的可以删除变量
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
		//设置当前帧的可以删除变量
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    //获取匹配成功闭环帧的共视帧
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();

	//当前帧设置为闭环帧的共视帧
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();
	//变量闭环帧的所有共视帧，添加地图点到mvpLoopMapPoints
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
    	//获取关键帧
        KeyFrame* pKF = *vit;
		//获取关键帧的地图点
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
		//遍历地图点
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                	//把地图点放入mvpLoopMapPoints
                    mvpLoopMapPoints.push_back(pMP);
					//防止重复添加
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    //将闭环匹配上关键帧以及相连关键帧的地图点投影到当前关键帧进行投影匹配
    //根据投影查找更多的匹配，成功的闭环匹配需要满足足够多的匹配特征点数
    //根据sim3变换，将每个地图点投影到当前帧上，并根据尺度确定一个搜索区域
    //根据该地图点的描述子与该区域内的特征点进行匹配，如果匹配误差小于50，
    //即匹配成功，更新当前帧匹配的地图点
    //当前帧匹配的地图点与当前地图点可能出现冲突，进行融合
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
	//遍历匹配结果，统计匹配成功个数
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

	//匹配成功个数大于40 匹配成功
    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
			//除了匹配成功的候选帧和当前帧，其余候选帧设置删除标志
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
    	//匹配失败
        for(int i=0; i<nInitialCandidates; i++)
			//设置候选帧的删除标志
            mvpEnoughConsistentCandidates[i]->SetErase();
		//设置当前帧删除标志
        mpCurrentKF->SetErase();
        return false;
    }

}

//闭环
// 1、通过求解sim3 以及相对位姿关系，
//调整与当前帧相连的关键帧位姿以及关键帧观测到的地图点位置
//2、将闭环帧以及闭环帧相连的关键帧的地图点和与当前帧相连的关键帧的地图点进行匹配
//3、通过地图点的匹配关系更新这些帧之间的连接关系，即共视帧的更新
//4、对位姿图进行优化，地图点的位置则根据优化后的位姿做相应的调整
//5、创建线程进行全局ba 优化
void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    //请求局部地图停止，防止
    mpLocalMapper->RequestStop();

    // If a Global Bundle Adjustment is running, abort it
    //如果真正进行全局优化
    if(isRunningGBA())
    {
        unique_lock<mutex> lock(mMutexGBA);
		//停止全局BA 优化
        mbStopGBA = true;

		//全局优化idx
        mnFullBAIdx++;
		//停止全局优化线程
        if(mpThreadGBA)
        {
        	//分离线程，利于底层回收资源
            mpThreadGBA->detach();
			//删除线程指针
            delete mpThreadGBA;
        }
    }

    // Wait until Local Mapping has effectively stopped
    //等待本地建图停止
    while(!mpLocalMapper->isStopped())
    {
        usleep(1000);
    }

    // Ensure current keyframe is updated
    //更新当前帧的共视帧权重
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    //获取当前帧的共视关键帧
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
	//插入当前帧
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

	//CorrectedSim3 保存sim3 闭环优化后的位姿
	//NonCorrectedSim3 保存没有闭环优化后的位姿
    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
	//保存当前帧的sim3 变换
    CorrectedSim3[mpCurrentKF]=mg2oScw;
	//se3 李群的旋转矩阵
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();


    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
		//通过位姿传播，得到sim3 调整后其它与当前帧相连关键帧的位姿

		//遍历所有的关联帧
        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
        	//获取关键帧
            KeyFrame* pKFi = *vit;

			//获取关键帧的位姿
            cv::Mat Tiw = pKFi->GetPose();

			//不等于当前帧，当前帧的前面已经加入
            if(pKFi!=mpCurrentKF)
            {
            	//得到当前帧到该相连帧的相对变换
                cv::Mat Tic = Tiw*Twc;
				//提取出R
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
				//提取出t
                cv::Mat tic = Tic.rowRange(0,3).col(3);

				//构造该相连帧的sim3 变换矩阵
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
				//当前帧的位姿固定不动，其它的关键帧根据相对关系得到sim3 调整的位姿
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                //得到闭环优化后各个关键帧的位姿
                CorrectedSim3[pKFi]=g2oCorrectedSiw;
            }

			//从位姿中提取R
            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
			//从位置中提取t
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
			//构造sim3 变换矩阵
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            //添加该相连帧没有进行闭环优化的位姿
            NonCorrectedSim3[pKFi]=g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        //得到调整相连帧的位姿后，修正这些关键帧的地图点
        //遍历修正后的位姿
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
        	//获取相关帧
            KeyFrame* pKFi = mit->first;
			//获取sim3 位姿
            g2o::Sim3 g2oCorrectedSiw = mit->second;
			//获取sim3 位姿的逆矩阵
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

			//获取对应的没有优化的位姿
            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

			//获取该关键帧的地图点
            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
			//遍历地图点
			//for(vector<MapPoint*>::iterator imp = vpMPsi.begin(), endMPi = vpMPsi.end(); iMP != endMPi, iMP++)
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
            {
            	//MapPoint* pMPi = *vpMPsi;
            	//获取地图点
                MapPoint* pMPi = vpMPsi[iMP];
                if(!pMPi)
                    continue;
                if(pMPi->isBad())
                    continue;
				//防止重复修正
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                //获取地图的3d 坐标
                cv::Mat P3Dw = pMPi->GetWorldPos();
				//把地图坐标矩阵转换成sim3 使用的矩阵
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
				//g2oSiw.map(eigP3Dw)  将eigP3Dw 映射到该帧没有优化的相机坐标系里
				//g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw))  然后在反映射到校准后的世界坐标系
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

				//转换为世界坐标矩阵的3D  点
                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
				//设置修正后的地图点
                pMPi->SetWorldPos(cvCorrectedP3Dw);
				//设置已经修正标志
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
				//修正参考帧为该相连帧
                pMPi->mnCorrectedReference = pKFi->mnId;
				//更新地图点的平均方向和观测范围
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            //将sim3 转换成se3  , 根据更新的sim3, 更新关键帧的位姿
            //计算R
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
			//计算t
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
			//获取s
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]


			//将sim3 转换成se3, 更新位姿
            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

			//设置修正后的位姿
            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            //更新该帧的共视帧连接
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        //检查当前帧的地图点与闭环匹配帧的地图点是否存在冲突，
        //对冲突的地图点进行替代或填补
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
            {
            	//获取闭环是当前帧匹配到的地图点
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
				//获取当前帧对应的地图点
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
				//如果当前特征点地图点存在
                if(pCurMP)
					//用闭环的地图点替换地图点
                    pCurMP->Replace(pLoopMP);
                else
                {
                	//当前特征点不存在地图点
                	//把闭环的地图点添加位该特征点的地图点
                    mpCurrentKF->AddMapPoint(pLoopMP,i);
					//更新该地图点的观测帧和相应的特征点
                    pLoopMP->AddObservation(mpCurrentKF,i);
					//重新计算地图点的最佳描述子
                    pLoopMP->ComputeDistinctiveDescriptors();
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    //通过将闭环时相连关键帧的闭环地图点投影到这些关键帧中，进行地图点检查和替换
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    //更新当前关键帧之间的共视相连关系，得到因闭环时地图点融合而新得到的连接关系
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

	//遍历当前帧相连关键帧，一级相连
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
    	//获取关键帧
        KeyFrame* pKFi = *vit;
		//获取该关键帧的共视帧，二级相连
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        //更新当前帧共视帧的连接
        pKFi->UpdateConnections();
		//取出该帧更新后的连接关系
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
		//从连接关系中去除闭环之前二级 的连接关系，剩下的连接就是由闭环得到的连接关系
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
        	//删除之前的连接关系
            LoopConnections[pKFi].erase(*vit_prev);
        }
		//从连接关系中去除之前的一级连接关系，剩下的连接就是由闭环得到的连接关系
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2);
        }
    }

    // Optimize graph
    //地图优化
    //进行位姿优化，LoopConnections 是形成闭环后新生成的连接关系，不包括步骤7 中当前帧与闭环匹配帧之间的连接关系
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    //添加当前帧与闭环匹配帧之间的边，这个连接关系不优化
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    //新建一个线程用于全局BA 优化
    //OptimizeEssentialGraph 只是优化了一些主要关键帧的位姿，这里进行全局BA 可以全局优化所有位姿和地图点
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
	//新建一个线程进行全局优化
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release();    

	//更新最近一次闭环帧
    mLastLoopKFid = mpCurrentKF->mnId;   
}

//通过将闭环时相连关键帧地图点投影到这些关键帧中，进行地图点的检查和替换
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
	//创建ORBmatcher 对象
    ORBmatcher matcher(0.8);

	//遍历闭环相连的关键帧
    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
    	//获取关键帧
        KeyFrame* pKF = mit->first;

		//获取该帧sim3 位姿
        g2o::Sim3 g2oScw = mit->second;
		//转换为cv 的矩阵形式
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

		//将闭环相连帧的地图点坐标变换到该帧坐标系，然后投影，检查冲突并融合
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
		//遍历所有闭环地图点
        for(int i=0; i<nLP;i++)
        {
        	//获取需要替换的地图点
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
            	//用闭环的地图点替换该帧的地图点
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
    	//清空回环检测
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
