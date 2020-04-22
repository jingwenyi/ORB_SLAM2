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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

//局部建图
//处理新的关键帧，使用local  BA  完成建图
//创建新的地图点，并根据后面的关键帧优化地图点
void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        //告诉跟踪线程，当前不接受关键帧
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        //等待处理的关键帧不为空
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            //计算关键帧的视觉词袋，将关键帧插入到地图
            ProcessNewKeyFrame();

            // Check recent MapPoints
            //剔除上一步不合格的地图点
            MapPointCulling();

            // Triangulate new MapPoints
            //通过三角化创造性的地图点
            CreateNewMapPoints();

			//已经处理完链表中的最后一个关键帧
            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                //检查并融合当前关键帧和相邻关键帧重复的地图点
                SearchInNeighbors();
            }

            mbAbortBA = false;

			//已经处理完最后一个关键帧，并且闭环检测没有请求停止
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                //BA 优化
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                //检查并剔除相邻关键帧中冗余的关键帧
                //剔除的标准是: 该关键帧90% 的地图点可以别其他关键帧观察到
                //删除冗余的关键帧
                KeyFrameCulling();
            }

			//将当前帧加入到闭环检测链表中
            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}


bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

//处理队列中的关键帧
//计算机视觉词袋，加速三角化新的地图点
void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
		//从缓冲链表中取出一帧关键帧
        mpCurrentKeyFrame = mlNewKeyFrames.front();
		//删除第一个元素
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    //计算关键帧特征点的视觉词袋
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    //获取当前帧的地图点
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

	//遍历当前帧的地图点
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
    	//获取地图点
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
            	//为当前帧在tracking 过程中跟踪到的地图点更新属性
            	//在该地图点的观测帧向量中找不到当前帧
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                	//为地图点添加关键帧
                    pMP->AddObservation(mpCurrentKeyFrame, i);
					//更新地图点平均观测方向和观测距离的范围
                    pMP->UpdateNormalAndDepth();
					//加入关键帧后更新，地图点的最佳描述子
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                	//将双目和rgbd 跟踪过程中新插入的地图点放入mlpRecentAddedMapPoints
                	//等待检查
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    //更新关键帧间的连接关系
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    //将该关键帧插入到地图中
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

//剔除质量不好的地图点
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    //获取最近添加地图点的迭代指针
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
	//获取当前帧的id
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

	//设置阈值
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

	//遍历所有的最近添加地图点
    while(lit!=mlpRecentAddedMapPoints.end())
    {
    	//获取地图点
        MapPoint* pMP = *lit;
		//地图点有问题
        if(pMP->isBad())
        {
        	//删除该地图点
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
		//跟踪到地图点的frame 数相比预计可观测到该地图点的frame 数
		//的比例需大于25%
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
		//从该点建立开始，已经过了不小于两个关键帧
		//但是观测到该点的关键帧数却不超过cnThObs ，那么该点检验不合格
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
		//从建立该地图点起，已经经过了3个关键帧没有被剔除
		//则认为是质量高的点，因此没有SetBadFlag，仅从队列中删除
		//放弃继续对该地图点检测
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

//相机运动中和共视程度比较高的关键帧通过三角恢复出一些地图点
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    //
    int nn = 10;
    if(mbMonocular)
        nn=20;
	//在当前关键帧的共视关键帧中找到共视程度最高的nn 帧相邻帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

	//创建orb 匹配对象
    ORBmatcher matcher(0.6,false);

	//获得当前帧的旋转R  3x3
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation()
    //求转置
    cv::Mat Rwc1 = Rcw1.t();
	//获取当前帧的平移T  1x3
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
	//定义一个3x4  位姿矩阵
    cv::Mat Tcw1(3,4,CV_32F);
	//把R 拷贝到0-3列
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
	//获取当前关键帧在世界坐标系中的坐标
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

	//获取相机内参
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

	//缩放因子
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    //遍历所有的关键帧的共视关键帧
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
    	//查看是否有新的关键帧到来
        if(i>0 && CheckNewKeyFrames())
            return;

		//获取关键帧
        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        //获取关键帧的位姿
        cv::Mat Ow2 = pKF2->GetCameraCenter();
		//基线向量，两个关键帧间的相机位移
        cv::Mat vBaseline = Ow2-Ow1;
		//基线长度
        const float baseline = cv::norm(vBaseline);

		//判断相机运动的基础线是不是足够长
        if(!mbMonocular)
        {
        	//相机运动距离小于双目或rgbd 的基线
        	//如果是立体相机，关键帧间距太小是，不生成3D 点
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
        	//评估当前帧的场景深度
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
			//比例
            const float ratioBaselineDepth = baseline/medianDepthKF2;

			//如果特别远(比例特别小)， 不生成3D 点
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        //根据两个关键帧的位姿，计算他们之间的基本矩阵
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
		//通过极线约束限制匹配时的搜索范围， 进行特征点匹配
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

		//获取李群旋转矩阵
        cv::Mat Rcw2 = pKF2->GetRotation();
		//求转置
        cv::Mat Rwc2 = Rcw2.t();
		//获取李群平移矩阵
        cv::Mat tcw2 = pKF2->GetTranslation();
		//f2 帧的位姿
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

		//f2 相机的内参
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        //对每个匹配，通过三角化生产3D 点
        //获取匹配的大小
        const int nmatches = vMatchedIndices.size();
		//遍历所有匹配
        for(int ikp=0; ikp<nmatches; ikp++)
        {
        	//获取匹配的索引
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

			//获取匹配对应的关键点
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
			//mvuRight 保存着双目的深度，如果不是双目，其值为-1
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            //利用匹配点的反投影得到视场角
            //特征点反投影
            //用像素平面的坐标求空间坐标 xscreen = fx(X/Z)+ cx, yscreen = fy(Y/Z) + cy
            //Z = 1, X = (xscreen - cx)/fx  Y = (yscreen - cy)/fy
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

			//由相机坐标系转到世界坐标系
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
			//计算视差角余弦值  cos(a,b) = a.b/(|a||b|)
			//这里求的是角3
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

			//+1 是为了让cosParallaxStereo 随便初始化一个很大的值
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

			//对应双目，利用双目得到视差角
			//		        p
			//			.
			//		      /|3\		     
			//		    /  | depth
			//		  /1	|    2\
			//            -------- mb
			// angle1 = atan2(mb/2, depth)
            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

			//获取一个最小的视差角
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

			//三角化恢复3D 点
			//cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)  表明视差角正常
			//cosParallaxRays<cosParallaxStereo 视差角很小
			//视差角小时用三角化恢复3D 点，视场角大时用双目恢复3D 点
            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                //生产的3D 点
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            //检查生成的3D 点是否在相机前方
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            //计算3D 点在当前关键帧下的重投影
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
				//基于卡方检验计算出阈值，假设测量有一个像素的偏差
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyfram
            //计算3D 点在另一个关键帧上的重投影
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            //检查尺度连续性
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

			//距离比
            const float ratioDist = dist2/dist1;
			//金字塔尺度因子比例
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            //表明尺度变化是不连续的
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            //
            //三角化生成3D 点成功，构造地图点
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

			//为该地图点添加属性
			//添加观测帧和特征点
            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

			//为关注添加地图带你
            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

			//更新地图点的描述子
            pMP->ComputeDistinctiveDescriptors();

			//更新地图点的平均观测方向和深度范围
            pMP->UpdateNormalAndDepth();

			//把地图点插入到局部地图中
            mpMap->AddMapPoint(pMP);
			//将新产生的点放入检查队列
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

//检查并融合当前帧与相邻帧重复的地图点
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
	//获取nn 数量的最近共视帧
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
	//遍历最近共视帧
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
    	//获取关键帧
        KeyFrame* pKFi = *vit;
		//如果该帧有问题或该帧已经融合过，直接返回
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
		//插入目标关键帧向量， 加入一级相邻
        vpTargetKFs.push_back(pKFi);
		//设置融合标志位，防止重复操作， 标记已加入
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        //搜索邻居的邻居
        //获取最近的5 个共视帧
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
		//遍历共视帧
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
        	//获取公司帧
            KeyFrame* pKFi2 = *vit2;
			//该帧是坏帧，已经是加入，或者该帧跟当前帧为同一帧
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
			//存入二级相邻帧，不用标记加入，因为一级已经标记
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    //创建ORBmatcher 对象
    ORBmatcher matcher;
	//获取当前帧地图点
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
	//遍历当前帧的融合目标帧
	//把当前帧 地图点投影到相邻的帧上，进行地图点的融合
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
    	//获取每个帧
        KeyFrame* pKFi = *vit;

		//融合地图点
		//投影当前帧的地图点到相邻的关键帧中，并判断是否有重复的地图点
        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    //这里开了一个当前帧的地图点个数 x  所有相邻帧的个数大小的向量
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

	//遍历当前帧的融合目标帧
	//把相邻帧地图点重投影到当前帧上，进行地图点的融合
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
    	//获取相邻的每一帧
        KeyFrame* pKFi = *vitKF;

		//获取相邻帧的地图点
        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

		//遍历相邻帧的地图点
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
        	//获取地图点
            MapPoint* pMP = *vitMP;
			//地图点null
            if(!pMP)
                continue;
			
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
			//设置融合标志，防止别的相邻帧加入该地图点
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
			//放入融合候选向量中
            vpFuseCandidates.push_back(pMP);
        }
    }

	//进行融合
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    //更新当前帧的地图点
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
            	//重新计算地图点的描述子
                pMP->ComputeDistinctiveDescriptors();
				//更新地图点平均观测方向和观测距离范围
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    //更新关键帧共视图
    mpCurrentKeyFrame->UpdateConnections();
}

//根据两个关键帧的姿态，计算两个关键帧之间的基础矩阵
//E= t12XR12
//F= inv(K1)*E*inv(K2)
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
	//获取李群的旋转矩阵
    cv::Mat R1w = pKF1->GetRotation();
	//获取李群的平移矩阵
    cv::Mat t1w = pKF1->GetTranslation();

	
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

	//RR  旋转变换
    cv::Mat R12 = R1w*R2w.t();
	//平移变换
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

	//获取t12 的斜对称矩阵
    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


	//返回F 矩阵
    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

//关键帧剔除
//如果关键帧所看的90% 的映射点都被看到，则该关键帧被认为是多余的
//插入一个关键帧的时候去检查所有关键帧是否冗余
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    //获取当前帧的共视关键帧的向量
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

	//遍历共视关键帧
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
    	//获取关键帧
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
		//提取该共视帧的地图点向量
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
		//遍历所有地图点
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
        	//获取地图点
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                    	//如果当前地图点的深度有问题
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
					//观测到该地图点帧的个数是否大于阈值3
                    if(pMP->Observations()>thObs)
                    {
                    	//或去该地图点的图像金字塔层
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
						//获取该地图点的所有观测帧
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;] //统计观测帧个数
                        //遍历该地图的所有观测帧
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                        	//获取关键帧
                            KeyFrame* pKFi = mit->first;
							//关键帧不等于当前地图点所在的帧
                            if(pKFi==pKF)
                                continue;
							//获取地图点在该关联帧的金字塔层
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

							//尺度约束，要求地图点在该局部关键帧的特征尺度大于或近似于
							//其它关键帧的特征尺度
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
								//已经找到3个同尺寸的关键帧能够观测到该地图点，不用在找了
                                if(nObs>=thObs)
                                    break;
                            }
                        }
						//该地图点被三个观测到
                        if(nObs>=thObs)
                        {
                        	//冗余观测计数
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

		//该局部关键帧90 %  的地图点可以被其他关键帧观测到，至少3帧观测到
		//则认为是冗余关键帧
        if(nRedundantObservations>0.9*nMPs)
			//设置地图关键帧标准为bad
            pKF->SetBadFlag();
    }
}

//获取斜对称矩阵
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
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
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
