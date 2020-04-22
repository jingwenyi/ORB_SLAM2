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

//�ֲ���ͼ
//�����µĹؼ�֡��ʹ��local  BA  ��ɽ�ͼ
//�����µĵ�ͼ�㣬�����ݺ���Ĺؼ�֡�Ż���ͼ��
void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        //���߸����̣߳���ǰ�����ܹؼ�֡
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        //�ȴ�����Ĺؼ�֡��Ϊ��
        if(CheckNewKeyFrames())
        {
            // BoW conversion and insertion in Map
            //����ؼ�֡���Ӿ��ʴ������ؼ�֡���뵽��ͼ
            ProcessNewKeyFrame();

            // Check recent MapPoints
            //�޳���һ�����ϸ�ĵ�ͼ��
            MapPointCulling();

            // Triangulate new MapPoints
            //ͨ�����ǻ������Եĵ�ͼ��
            CreateNewMapPoints();

			//�Ѿ������������е����һ���ؼ�֡
            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                //��鲢�ںϵ�ǰ�ؼ�֡�����ڹؼ�֡�ظ��ĵ�ͼ��
                SearchInNeighbors();
            }

            mbAbortBA = false;

			//�Ѿ����������һ���ؼ�֡�����ұջ����û������ֹͣ
            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                //BA �Ż�
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap);

                // Check redundant local Keyframes
                //��鲢�޳����ڹؼ�֡������Ĺؼ�֡
                //�޳��ı�׼��: �ùؼ�֡90% �ĵ�ͼ����Ա������ؼ�֡�۲쵽
                //ɾ������Ĺؼ�֡
                KeyFrameCulling();
            }

			//����ǰ֡���뵽�ջ����������
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

//��������еĹؼ�֡
//������Ӿ��ʴ����������ǻ��µĵ�ͼ��
void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
		//�ӻ���������ȡ��һ֡�ؼ�֡
        mpCurrentKeyFrame = mlNewKeyFrames.front();
		//ɾ����һ��Ԫ��
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    //����ؼ�֡��������Ӿ��ʴ�
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    //��ȡ��ǰ֡�ĵ�ͼ��
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

	//������ǰ֡�ĵ�ͼ��
    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
    	//��ȡ��ͼ��
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
            	//Ϊ��ǰ֡��tracking �����и��ٵ��ĵ�ͼ���������
            	//�ڸõ�ͼ��Ĺ۲�֡�������Ҳ�����ǰ֡
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                	//Ϊ��ͼ����ӹؼ�֡
                    pMP->AddObservation(mpCurrentKeyFrame, i);
					//���µ�ͼ��ƽ���۲ⷽ��͹۲����ķ�Χ
                    pMP->UpdateNormalAndDepth();
					//����ؼ�֡����£���ͼ������������
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // this can only happen for new stereo points inserted by the Tracking
                {
                	//��˫Ŀ��rgbd ���ٹ������²���ĵ�ͼ�����mlpRecentAddedMapPoints
                	//�ȴ����
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    //���¹ؼ�֡������ӹ�ϵ
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    //���ùؼ�֡���뵽��ͼ��
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

//�޳��������õĵ�ͼ��
void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    //��ȡ�����ӵ�ͼ��ĵ���ָ��
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
	//��ȡ��ǰ֡��id
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

	//������ֵ
    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

	//�������е������ӵ�ͼ��
    while(lit!=mlpRecentAddedMapPoints.end())
    {
    	//��ȡ��ͼ��
        MapPoint* pMP = *lit;
		//��ͼ��������
        if(pMP->isBad())
        {
        	//ɾ���õ�ͼ��
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
		//���ٵ���ͼ���frame �����Ԥ�ƿɹ۲⵽�õ�ͼ���frame ��
		//�ı��������25%
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
		//�Ӹõ㽨����ʼ���Ѿ����˲�С�������ؼ�֡
		//���ǹ۲⵽�õ�Ĺؼ�֡��ȴ������cnThObs ����ô�õ���鲻�ϸ�
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
		//�ӽ����õ�ͼ�����Ѿ�������3���ؼ�֡û�б��޳�
		//����Ϊ�������ߵĵ㣬���û��SetBadFlag�����Ӷ�����ɾ��
		//���������Ըõ�ͼ����
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

//����˶��к͹��ӳ̶ȱȽϸߵĹؼ�֡ͨ�����ǻָ���һЩ��ͼ��
void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    //
    int nn = 10;
    if(mbMonocular)
        nn=20;
	//�ڵ�ǰ�ؼ�֡�Ĺ��ӹؼ�֡���ҵ����ӳ̶���ߵ�nn ֡����֡
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

	//����orb ƥ�����
    ORBmatcher matcher(0.6,false);

	//��õ�ǰ֡����תR  3x3
    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation()
    //��ת��
    cv::Mat Rwc1 = Rcw1.t();
	//��ȡ��ǰ֡��ƽ��T  1x3
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
	//����һ��3x4  λ�˾���
    cv::Mat Tcw1(3,4,CV_32F);
	//��R ������0-3��
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
	//��ȡ��ǰ�ؼ�֡����������ϵ�е�����
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

	//��ȡ����ڲ�
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

	//��������
    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    //�������еĹؼ�֡�Ĺ��ӹؼ�֡
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
    	//�鿴�Ƿ����µĹؼ�֡����
        if(i>0 && CheckNewKeyFrames())
            return;

		//��ȡ�ؼ�֡
        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        //��ȡ�ؼ�֡��λ��
        cv::Mat Ow2 = pKF2->GetCameraCenter();
		//���������������ؼ�֡������λ��
        cv::Mat vBaseline = Ow2-Ow1;
		//���߳���
        const float baseline = cv::norm(vBaseline);

		//�ж�����˶��Ļ������ǲ����㹻��
        if(!mbMonocular)
        {
        	//����˶�����С��˫Ŀ��rgbd �Ļ���
        	//���������������ؼ�֡���̫С�ǣ�������3D ��
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
        	//������ǰ֡�ĳ������
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
			//����
            const float ratioBaselineDepth = baseline/medianDepthKF2;

			//����ر�Զ(�����ر�С)�� ������3D ��
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        //���������ؼ�֡��λ�ˣ���������֮��Ļ�������
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
		//ͨ������Լ������ƥ��ʱ��������Χ�� ����������ƥ��
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

		//��ȡ��Ⱥ��ת����
        cv::Mat Rcw2 = pKF2->GetRotation();
		//��ת��
        cv::Mat Rwc2 = Rcw2.t();
		//��ȡ��Ⱥƽ�ƾ���
        cv::Mat tcw2 = pKF2->GetTranslation();
		//f2 ֡��λ��
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

		//f2 ������ڲ�
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        //��ÿ��ƥ�䣬ͨ�����ǻ�����3D ��
        //��ȡƥ��Ĵ�С
        const int nmatches = vMatchedIndices.size();
		//��������ƥ��
        for(int ikp=0; ikp<nmatches; ikp++)
        {
        	//��ȡƥ�������
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

			//��ȡƥ���Ӧ�Ĺؼ���
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
			//mvuRight ������˫Ŀ����ȣ��������˫Ŀ����ֵΪ-1
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            //����ƥ���ķ�ͶӰ�õ��ӳ���
            //�����㷴ͶӰ
            //������ƽ���������ռ����� xscreen = fx(X/Z)+ cx, yscreen = fy(Y/Z) + cy
            //Z = 1, X = (xscreen - cx)/fx  Y = (yscreen - cy)/fy
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

			//���������ϵת����������ϵ
            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
			//�����Ӳ������ֵ  cos(a,b) = a.b/(|a||b|)
			//��������ǽ�3
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

			//+1 ��Ϊ����cosParallaxStereo ����ʼ��һ���ܴ��ֵ
            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

			//��Ӧ˫Ŀ������˫Ŀ�õ��Ӳ��
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

			//��ȡһ����С���Ӳ��
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

			//���ǻ��ָ�3D ��
			//cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998)  �����Ӳ������
			//cosParallaxRays<cosParallaxStereo �Ӳ�Ǻ�С
			//�Ӳ��Сʱ�����ǻ��ָ�3D �㣬�ӳ��Ǵ�ʱ��˫Ŀ�ָ�3D ��
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
                //������3D ��
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
            //������ɵ�3D ���Ƿ������ǰ��
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            //����3D ���ڵ�ǰ�ؼ�֡�µ���ͶӰ
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
				//���ڿ�������������ֵ�����������һ�����ص�ƫ��
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
            //����3D ������һ���ؼ�֡�ϵ���ͶӰ
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
            //���߶�������
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

			//�����
            const float ratioDist = dist2/dist1;
			//�������߶����ӱ���
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            //�����߶ȱ仯�ǲ�������
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            //
            //���ǻ�����3D ��ɹ��������ͼ��
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

			//Ϊ�õ�ͼ���������
			//��ӹ۲�֡��������
            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

			//Ϊ��ע��ӵ�ͼ����
            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

			//���µ�ͼ���������
            pMP->ComputeDistinctiveDescriptors();

			//���µ�ͼ���ƽ���۲ⷽ�����ȷ�Χ
            pMP->UpdateNormalAndDepth();

			//�ѵ�ͼ����뵽�ֲ���ͼ��
            mpMap->AddMapPoint(pMP);
			//���²����ĵ���������
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

//��鲢�ںϵ�ǰ֡������֡�ظ��ĵ�ͼ��
void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
	//��ȡnn �������������֡
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
	//�����������֡
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
    	//��ȡ�ؼ�֡
        KeyFrame* pKFi = *vit;
		//�����֡��������֡�Ѿ��ںϹ���ֱ�ӷ���
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
		//����Ŀ��ؼ�֡������ ����һ������
        vpTargetKFs.push_back(pKFi);
		//�����ںϱ�־λ����ֹ�ظ������� ����Ѽ���
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        //�����ھӵ��ھ�
        //��ȡ�����5 ������֡
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
		//��������֡
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
        	//��ȡ��˾֡
            KeyFrame* pKFi2 = *vit2;
			//��֡�ǻ�֡���Ѿ��Ǽ��룬���߸�֡����ǰ֡Ϊͬһ֡
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
			//�����������֡�����ñ�Ǽ��룬��Ϊһ���Ѿ����
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    //����ORBmatcher ����
    ORBmatcher matcher;
	//��ȡ��ǰ֡��ͼ��
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
	//������ǰ֡���ں�Ŀ��֡
	//�ѵ�ǰ֡ ��ͼ��ͶӰ�����ڵ�֡�ϣ����е�ͼ����ں�
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
    	//��ȡÿ��֡
        KeyFrame* pKFi = *vit;

		//�ںϵ�ͼ��
		//ͶӰ��ǰ֡�ĵ�ͼ�㵽���ڵĹؼ�֡�У����ж��Ƿ����ظ��ĵ�ͼ��
        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    //���￪��һ����ǰ֡�ĵ�ͼ����� x  ��������֡�ĸ�����С������
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

	//������ǰ֡���ں�Ŀ��֡
	//������֡��ͼ����ͶӰ����ǰ֡�ϣ����е�ͼ����ں�
    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
    	//��ȡ���ڵ�ÿһ֡
        KeyFrame* pKFi = *vitKF;

		//��ȡ����֡�ĵ�ͼ��
        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

		//��������֡�ĵ�ͼ��
        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
        	//��ȡ��ͼ��
            MapPoint* pMP = *vitMP;
			//��ͼ��null
            if(!pMP)
                continue;
			
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
			//�����ںϱ�־����ֹ�������֡����õ�ͼ��
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
			//�����ںϺ�ѡ������
            vpFuseCandidates.push_back(pMP);
        }
    }

	//�����ں�
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    //���µ�ǰ֡�ĵ�ͼ��
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
            	//���¼����ͼ���������
                pMP->ComputeDistinctiveDescriptors();
				//���µ�ͼ��ƽ���۲ⷽ��͹۲���뷶Χ
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    //���¹ؼ�֡����ͼ
    mpCurrentKeyFrame->UpdateConnections();
}

//���������ؼ�֡����̬�����������ؼ�֮֡��Ļ�������
//E= t12XR12
//F= inv(K1)*E*inv(K2)
cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
	//��ȡ��Ⱥ����ת����
    cv::Mat R1w = pKF1->GetRotation();
	//��ȡ��Ⱥ��ƽ�ƾ���
    cv::Mat t1w = pKF1->GetTranslation();

	
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

	//RR  ��ת�任
    cv::Mat R12 = R1w*R2w.t();
	//ƽ�Ʊ任
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

	//��ȡt12 ��б�Գƾ���
    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


	//����F ����
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

//�ؼ�֡�޳�
//����ؼ�֡������90% ��ӳ��㶼����������ùؼ�֡����Ϊ�Ƕ����
//����һ���ؼ�֡��ʱ��ȥ������йؼ�֡�Ƿ�����
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    //��ȡ��ǰ֡�Ĺ��ӹؼ�֡������
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

	//�������ӹؼ�֡
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
    	//��ȡ�ؼ�֡
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
		//��ȡ�ù���֡�ĵ�ͼ������
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
		//�������е�ͼ��
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
        	//��ȡ��ͼ��
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                    	//�����ǰ��ͼ������������
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
					//�۲⵽�õ�ͼ��֡�ĸ����Ƿ������ֵ3
                    if(pMP->Observations()>thObs)
                    {
                    	//��ȥ�õ�ͼ���ͼ���������
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
						//��ȡ�õ�ͼ������й۲�֡
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;] //ͳ�ƹ۲�֡����
                        //�����õ�ͼ�����й۲�֡
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                        	//��ȡ�ؼ�֡
                            KeyFrame* pKFi = mit->first;
							//�ؼ�֡�����ڵ�ǰ��ͼ�����ڵ�֡
                            if(pKFi==pKF)
                                continue;
							//��ȡ��ͼ���ڸù���֡�Ľ�������
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

							//�߶�Լ����Ҫ���ͼ���ڸþֲ��ؼ�֡�������߶ȴ��ڻ������
							//�����ؼ�֡�������߶�
                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
								//�Ѿ��ҵ�3��ͬ�ߴ�Ĺؼ�֡�ܹ��۲⵽�õ�ͼ�㣬����������
                                if(nObs>=thObs)
                                    break;
                            }
                        }
						//�õ�ͼ�㱻�����۲⵽
                        if(nObs>=thObs)
                        {
                        	//����۲����
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

		//�þֲ��ؼ�֡90 %  �ĵ�ͼ����Ա������ؼ�֡�۲⵽������3֡�۲⵽
		//����Ϊ������ؼ�֡
        if(nRedundantObservations>0.9*nMPs)
			//���õ�ͼ�ؼ�֡��׼Ϊbad
            pKF->SetBadFlag();
    }
}

//��ȡб�Գƾ���
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
