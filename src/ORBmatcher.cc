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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

//orb �������ɹؼ������������ɣ�������256��λ
const int ORBmatcher::TH_HIGH = 100;//ƥ��ϸ���ֵ 100   ����������
const int ORBmatcher::TH_LOW = 50; //ƥ��ϵ���ֵ50  ����������
const int ORBmatcher::HISTO_LENGTH = 30;//��360��ֳ�30��bin, ÿ��bin 12��

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}


//����ÿ���ֲ�3D ��ͨ��ͶӰ��С��Χ���ҵ���ƥ���2D ��
//�Ӷ�ʵ��frame �Ա��ص�ͼ��׷��
int ORBmatcher::SearchByProjection(Frame &F,  //��ǰ֡
										const vector<MapPoint*> &vpMapPoints,  // ���ص�ͼ
										const float th)		//������Χ����
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

	//��ȡ�ֲ���ͼ��ͼ��
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
    	//��ȡ��ͼ��
        MapPoint* pMP = vpMapPoints[iMP];
		//�жϸõ��Ƿ�Ҫ��ͶӰ
        if(!pMP->mbTrackInView)
            continue;

		//�õ�ͼ���־λ�Ƿ����
        if(pMP->isBad())
            continue;

		//��ȡ��ͼ���ڽ������Ĳ���
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        //���ݹ۲⵽3D ����ӽ�ȷ���������ڵĴ�С
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

		//������������
        if(bFactor)
            r*=th;

		//�ڵ�ǰ֡�ж�Ӧͼ��������Ĵ��������ͶӰ�����ƥ��
		//�ҳ��������ڵ�����������
        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

		//û��ƥ�䵽
        if(vIndices.empty())
            continue;

		//��ȡ��ͼ����ڵ�������
        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        //�Դ���ƥ��õ�����������������ƥ��
        //�ҵ����ƥ��ʹμ�ƥ�䣬�������ƥ�����С����ֵ��
        //������ƥ���������ڴμ�ƥ�����ͼ3D ��͵�ǰ֡����2D ��ƥ��ɹ�
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//��ȡ������������������
            const size_t idx = *vit;

			//���frame ��Ȥ�㣬�Ѿ��ж�Ӧ�ĵ�ͼ�㣬���˳��ô�ѭ��
            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

			//˫Ŀ���м����ͼ��
            if(F.mvuRight[idx]>0)
            {
            	//�����ͼ��ͶӰ�����Ӧ������ľ���
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
				//�������Ƚ�
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

			//������������������
            const cv::Mat &d = F.mDescriptors.row(idx);
			//�����ͼ�������Ӻ������������ӵĺ�������
            const int dist = DescriptorDistance(MPdescriptor,d);

			//�ҵ���С����
            if(dist<bestDist)
            {
                bestDist2=bestDist;		//��С����
                bestDist=dist;			//��С����
                bestLevel2 = bestLevel;  //��С�����Ӧ�Ľ�������
				//��С�����Ӧ�Ľ�������
                bestLevel = F.mvKeysUn[idx].octave;
				//��С�����Ӧ�����������
                bestIdx=idx;
            }
            else if(dist<bestDist2) //��С����
            {
            	//��С�������������
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        //��С����С����ֵ
        if(bestDist<=TH_HIGH)
        {	//��С����ʹ�С������ͬһ���������㣬
        	//��С���벻С��0.9����С���룬 ƥ��ʧ�ܣ�����
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

			//ƥ��ɹ���Ϊframe �е���Ȥ�����ӵ�ͼ��
            F.mvpMapPoints[bestIdx]=pMP;
			//�ɹ�ƥ�����
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998) //����
        return 2.5;
    else				//б��
        return 4.0;
}

//���kp1 ��pKF2 �϶�Ӧ�ļ���
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image
    //l = x1'F12 = [a b c]
    //�ֱ���a b c
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

	//����kp2 �����㵽���ߵľ���
	//���߷���l : ax + by + c =0
	//(u,v) ��l �ľ���Ϊ|av + bv +c| / sqrt(a^2 + b^)
    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

	//�����ƽ��
    const float dsqr = num*num/den;

	//�߶�Խ�󣬷�ΧӦ��ҲԽ��
	//��������Ͳ�һ�����ص��ռһ�����ص㣬
	//�ڵ����ڶ��㣬һ�����ص����1.2�����ص㣬����������߶�Ϊ1.2
    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

//���Ӿ��ʴ��Բο��ؼ�֡�͵�ǰ֡���йؼ������ƥ��
//ÿһ���ؼ����ܶ�����һ����ͼ�㣬�ؼ���ƥ��ɹ���
//��ǰ֡��Ӧ�ĵ�ͼ��Ҳ���ҵ���
int ORBmatcher::SearchByBoW(KeyFrame* pKF, //�ο�֡
								Frame &F, 		//��ǰ֡
								vector<MapPoint*> &vpMapPointMatches)  //��ǰ֡���ٳɹ��ĵ�ͼ��
{
	//��ȡ�ο�֡��ͼ��
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

	//��ʼ����ǰ֡��ͼ��
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

	//��ȡ�ο�֡�Ӿ��ʴ�����������
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

	//��360�㻮����30�����䣬ÿ������12��
    vector<int> rotHist[HISTO_LENGTH];
	//Ϊû�ɹ���������ռ�
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
	//�����������
    const float factor = 1.0f/HISTO_LENGTH;


    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    //��ȡ�ο�֡�Ӿ��ʴ�����������ʼ��ַ
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
	//��ȡ��ǰ֡�Ӿ��ʴ�����������ʼλ��
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
	//��ȡ�ο�֡�Ӿ��ʴ�������������λ��
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
	//��ȡ��ǰ֡�Ӿ��ʴ�������������λ��
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

	
    while(KFit != KFend && Fit != Fend)
    {
    	//�ֱ�ȡ��ͬһnode ��ORB ������
    	//ֻ������ͬһnode ,���п�����ƥ���
        if(KFit->first == Fit->first)
        {
        	//��ȡkf   �Ӿ�ʴ�������������
            const vector<unsigned int> vIndicesKF = KFit->second;
			//��ȡ��ǰ֡�Ӿ�ʴ��������������
            const vector<unsigned int> vIndicesF = Fit->second;

			//����kf �����ڸ�node ��������
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
            	//��ȡkf ��Ӧ�������idx
                const unsigned int realIdxKF = vIndicesKF[iKF];

				//ͨ���������idx  ��ȡ��Ӧ�ĵ�ͼ��
                MapPoint* pMP = vpMapPointsKF[realIdxKF];

				//��ͼ�㲻����
                if(!pMP)
                    continue;
				//��ͼ���ǻ���
                if(pMP->isBad())
                    continue;                

				//ȡ���ο��ؼ�֡����������ڵ�������
                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;//��С����
                int bestIdxF =-1 ;//��¼��ǰ֡��ƥ�䵽�����������
                int bestDist2=256;//�ڶ�С����

				//������ǰ֡���ڸ�node ��������
                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                	//��ȡ��ǰ֡��Ӧ�������idex
                    const unsigned int realIdxF = vIndicesF[iF];

					//�����ǰ֡���������Ѿ�ƥ�䵽��ͼ��
                    if(vpMapPointMatches[realIdxF])
                        continue;
					//��ȡ��ǰ֡���������Ӧ��������
                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);

					//����ο�֡������͵�ǰ֡�����������ӵĺ�������
                    const int dist =  DescriptorDistance(dKF,dF);

					//�ҳ����ڲο��ؼ�֡��һ�������㣬 ���ڵĵ�ǰ֡��������
					//����������С�͵ڶ�С��������
                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

				//������ֵ�ͽǶ�ͶƱ���޳���ƥ��
                if(bestDist1<=TH_LOW)
                {
                	//static_cast ����ǿ��ת������mfNNratio = 0.9
                	//�����С����С��0.9 ���ڶ�С�ľ���
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                    	//��ǰ֡������ƥ��ɹ���������ڵ�ͼ��
                        vpMapPointMatches[bestIdxF]=pMP;

						//��ȡ�ο��ؼ�֡�ùؼ���
                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

						//������
                        if(mbCheckOrientation)
                        {
                        	//angle:ÿ������������ȡ������ʱ����ת������Ƕȣ�
                        	//���ͼ����ת�ˣ�����ǶȽ������ı䣬
                        	//������������ĽǶȱ仯Ӧ����һ�µģ�
                        	//ͨ��ֱ��ͼͳ�Ƶõ���׼ȷ�ĽǶȱ仯ֵ
                        	//�ؼ�֡�Ƕ�������õ���������ĽǶȱ仯
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
							//��������
                            int bin = round(rot*factor); 
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
							//��rot ���䵽bin ��
                            rotHist[bin].push_back(bestIdxF);
                        }
						//ƥ������Լ�
                        nmatches++;
                    }
                }

            }

			//node �Լ�
            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
        	//�ο��ؼ�֡ �ҵ��뵱ǰ֡��Ӧ��node
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
        	//��ǰ֡�ҵ���ο��ؼ�֡��Ӧ��node
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


	//���ݷ����޳���ƥ���
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

		//�ҳ�������3������ֵ
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
            	//�޳������쳣�ĵ�ͼ�㣬static_cast ����ǿ��ת��
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
				//ƥ�������Լ�
                nmatches--;
            }
        }
    }

    return nmatches;
}

//���ڱջ�����н���ͼ��͹ؼ�֡����������й���
//����sim3 �任����ÿ����ͼ�㣬ͶӰ��pkf �ϣ������ݳ߶�ȷ��һ����������
//���ݸõ�ͼ�����������������ڵ����������ƥ�䣬���ƥ�����С��50 ��
//��ƥ��ɹ�������ƥ��
int ORBmatcher::SearchByProjection(KeyFrame* pKF, //�ջ����ĵ�ǰ֡
										cv::Mat Scw, 	//ʱ������ϵ����ǰ֡��sim3 �任����
										const vector<MapPoint*> &vpPoints, //�ջ����ĵ�ͼ��
										vector<MapPoint*> &vpMatched, //ƥ��ɹ��ĵ�ͼ��
										int th)							//��ֵ  10
{
    // Get Calibration Parameters for later projection
    //��ȡ����ڲ�
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    //scwd����ʽΪ[sR, st]
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
	//����õ��߶�s
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
	//R = sR / s
    cv::Mat Rcw = sRcw/scw;
	//t = st / t
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
	//��������ϵ�£� pkf ����������ϵ��λ�ˣ�������pkfָ����������ϵ
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    //�ѱջ��ĵ�ͼ�����set �����У����ٲ���ƥ��
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
	//ɾ�������е�����null �ĵ�ͼ��
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    //����ÿһ����ͼ�㣬Ϊ��ͼ���ҵ�ƥ���������
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
    	//��ȡ��ͼ��
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        //�������ĵ�ͼ����Ѿ�ƥ��ĵ�ͼ��
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        //��ȡ�õ�ͼ�����������3D ��
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        //�ѵ�ͼ��ת�����������ϵ
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        //��ȱ������0�� ����Ƽ�ǰ���ĵ�
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        //�������ƽ���϶�Ӧ���������
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //���������������Ƿ���ͼ����Ч��Χ��
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        //��ȡ��ͼ�㵽�������ȷ�Χ
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
		//�����ͼ�㵽������ĵ�����
        cv::Mat PO = p3Dw-Ow;
		//����������ģ
        const float dist = cv::norm(PO);

		//�þ����Ƿ�����ȷ�Χ��
        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        //ƽ���۲췽��
        cv::Mat Pn = pMP->GetNormal();

		//�����֡�Ե�ͼ��Ĺ۲ⷽ�����ͼ���ƽ���۲ⷽ��ļнǣ� ����60 ��ʾ̫��
		//cos(a,b) = a.b/|a||b|, |Pn| = 1
        if(PO.dot(Pn)<0.5*dist)
            continue;

		//ͨ������Ԥ���ͼ���ڵ�ǰ֡ͼ�����������һ��
        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        //���������뾶��ֵ
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

		//�ҳ��ؼ�֡����ֵ�ڵ�����������
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

		//һ�������㶼û���ҵ�
        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //��ȡ��ͼ���������
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
		//�����ҵ������е�������
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//��ȡ�������idx
            const size_t idx = *vit;
			//�鿴pkf�и��������Ƿ��Ѿ�ƥ���ͼ��
            if(vpMatched[idx])
                continue;

			//��ȡ�������ͼ���������
            const int &kpLevel= pKF->mvKeysUn[idx].octave;

			//�����������ͼ����������Ƿ���֮ǰ���ƵĲ��л�����һ��
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

			//��ȡ���������������
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

			//���������ӵĺ�������
            const int dist = DescriptorDistance(dMP,dKF);

			//�ҵ�����������С���Ǹ�
            if(dist<bestDist)
            {
                bestDist = dist;
				//��Ӧ��������idx
                bestIdx = idx;
            }
        }

		//�����⵽����С�ĺ�������С��50
        if(bestDist<=TH_LOW)
        {
        	//����������Ӷ�Ӧ�ĵ�ͼ��
            vpMatched[bestIdx]=pMP;
			//ƥ��ɹ���1
            nmatches++;
        }

    }

	//����ƥ��ɹ���
    return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1,  								//�ο�֡
										   Frame &F2, 								//��ǰ֡
										   vector<cv::Point2f> &vbPrevMatched, 		//�ο�֡�ؼ��������
										   vector<int> &vnMatches12, 				//��Ҫƥ���ĸ���
										   int windowSize)							//�ؼ���Ĵ��ڴ�С
{
    int nmatches=0;
	//������Ҫƥ������Ϊ�ο�֡�Ĺؼ������
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

	//��360�ȷֳ�30 ��bin , ������Ϊÿһ��bin ���뱣���ռ�
	//ÿ��bin ��Ӧ12�� ������
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
	//��������
    const float factor = 1.0f/HISTO_LENGTH;

	//��ǰ֡ÿ���ؼ�ƥ��������������
    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
	//��ǰ֡ƥ�䵽�ο�֡�ؼ���Ĺ�������
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
    	//˳���ȡ�ؼ���
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;// ��ȡ����������ڽ�������һ���ȡ
		//ֻ�����������0 ����ȡ���Ĺؼ��㣬��ԭͼ
		if(level1>0)
            continue;

		//�Ѳο�֡�Ĺؼ���ŵ���ǰ֡�У� 
		// ���Բο�֡Ϊ���ĵ�100 Ϊ�뾶 �Ĵ������ҵ�ȫ���Ĺؼ���
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

		//û��ƥ�䵽��������һ���ؼ����ƥ��
        if(vIndices2.empty())
            continue;
		//��ȡ�ο�֡ �ؼ���� orb ������
        cv::Mat d1 = F1.mDescriptors.row(i1);

		//�������
        int bestDist = INT_MAX;
		//���뵹���ڶ���
        int bestDist2 = INT_MAX;
		//��¼������̵�idx
        int bestIdx2 = -1;

		//����ƥ�䵽��ÿһ���ؼ�֡
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;
			//��ȡ��ǰ֡�ùؼ����orb ������
            cv::Mat d2 = F2.mDescriptors.row(i2);
			//��ȡ�����ؼ�֡�����ӵľ���
            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

			//�ҵ���С��ǰ��������
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

		//ȷ����С����С����ֵ50
        if(bestDist<=TH_LOW)
        {
        	//��ȷ����С����ҪС�ڴ�С�������mfNNratio=0.9
            if(bestDist<(float)bestDist2*mfNNratio)
            {
            	//����Ѿ�ƥ��
                if(vnMatches21[bestIdx2]>=0)
                {
                	//�Ƴ�ƥ��
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
				//�Ѳο�֡�ؼ����뵱ǰ֡�ؼ������
                vnMatches12[i1]=bestIdx2;
				//�ѵ�ǰ֡�ؼ�����ο�֡�ؼ������
                vnMatches21[bestIdx2]=i1;
				//���浱ǰ֡�ؼ���ƥ�����С����
                vMatchedDistance[bestIdx2]=bestDist;
				//ƥ������1
                nmatches++;

				//�Ƿ��鷽��
                if(mbCheckOrientation)
                {
                	//�ο�֡�ؼ���ķ���- ��ǰ֡�ؼ���ķ���
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
					//�ѷ�������0-360֮��
                    if(rot<0.0)
                        rot+=360.0f;

					//����÷�������ĸ�bin ��
                    int bin = round(rot*factor);
					//�ѵ�ǰ���ĸùؼ���push ����Ӧ��bin ��
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

	//���з���ƥ��
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
		//�ҳ�bin ��Ԫ�ظ�������3��bin����
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

		//�Ѳ�����3����������bin ƥ��Ĺؼ��㶼�Ƴ���
        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    //ͨ���ο�֡�����괫���뵱ǰ֡ƥ�䵽�Ĺؼ��������
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;
	//����ƥ�䵽�ĸ���
    return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

//���û�������f12 ��pKF1��pKF2 ֮��������ƥ��
//����: ��pKF1 ��������û�ж�Ӧ��3D ��ʱͨ��ƥ�������������µ�3D  ��
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, 
											KeyFrame *pKF2, 								//
											cv::Mat F12,									//  �����������
                                       		vector<pair<size_t, size_t> > &vMatchedPairs,  	// ����ƥ��ɹ���
                                       		const bool bOnlyStereo)	//��˫Ŀ��rgbd �����Ҫ������������ͼ����ƥ��
{    
	//��ȡ��֡ͼ��ʴ�����������
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    //����kf1 ��������ĵ�kf2 ͼ��ƽ������꣬ ����������
    //��ȡf1 �������
    cv::Mat Cw = pKF1->GetCameraCenter();
	//��ȡf2 ��ת  
    cv::Mat R2w = pKF2->GetRotation();
	//��ȡf2 ��ƽ�ƾ���
    cv::Mat t2w = pKF2->GetTranslation();
	//f1 ���������f2 ����ϵ�еı�ʾ
    cv::Mat C2 = R2w*Cw+t2w;
	//�õ�kf1��kf2 �ļ�������
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;


	
	//ʹ���Ӿ��ʴ����ٱȽ�
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

	//����pkf1 ��pkf2 ��node �ڵ�
    while(f1it!=f1end && f2it!=f2end)
    {
    	//���flit  ��f2it ����ͬһ��node
        if(f1it->first == f2it->first)
        {
        	//������node �µ�����������
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
            	//��ȡf1 ����node ������������
                const size_t idx1 = f1it->second[i1];

				//��ȡf1 ��Ӧ�ĵ�ͼ��
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                //������������Ѿ��ж�Ӧ�ĵ�ͼ�㣬�ͼ���
                //����Ѱ�ҵ���δƥ�������㣬����Ӧ����null
                if(pMP1)
                    continue;


				//mvuRight ֵ����0 ��ʾ��˫Ŀ�����������������
                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

				//��������ȵ���Ч��
                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                //ȡ��f1 ��Ӧ��������
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

				//ȡ����������ڵ�������
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;

				//����node �ڵ���f2 ������������
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                	//��ȡf2 ����������
                    size_t idx2 = f2it->second[i2];

					//��ȡ��Ӧ�ĵ�ͼ��
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    //����Ѿ�ƥ����߸��������ͼ����ڣ�����
                    if(vbMatched2[idx2] || pMP2)
                        continue;

					//˫Ŀ
                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    //��ȡf2 ���������������
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

					//����f1 �������f2 �����������ӵĺ�������
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

					//��ȡf2 ��������
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
						//����������뼫��̫������ʾf2 ���������Ӧ�ĵ�ͼ����f1 ���̫��
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

					//����������kp2 ��kp1 �ļ��ߵľ����Ƿ�С����ֵ
					//kp1 ����pkf2 ��һ������
					//����Լ��
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                    	//�ҵ�����ʵ�ƥ��㣬
                    	//���㼫��Լ��������������С�ĵ�
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

				//f1 ��û�е�ͼ��Ĺؼ���ƥ�䵽һ��f2 ��û�е�ͼ�Ĺؼ���
                if(bestIdx2>=0)
                {
                	//��ȡf2 �ؼ��������
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
					//����ƥ��ɹ��ĵ�
                    vMatches12[idx1]=bestIdx2;
					//ƥ�������1
                    nmatches++;

					//�Ƕȼ��
                    if(mbCheckOrientation)
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

	//���нǶȼ�飬�޳�����ƥ���
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

		//�ҳ�3���Ƕ�����
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
			//�޳���ƥ���
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
            	//ƥ�����-1
                vMatches12[rotHist[i][j]]=-1;
				//ƥ��ɹ���--
                nmatches--;
            }
        }

    }

	//
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

	//�ѳɹ�ƥ��Էŵ�vMatchedPairs ������
    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

//����ͼ��ͶӰ���ؼ�֡�У��鿴�Ƿ����ظ��ĵ�ͼ��
//1. �����ͼ����ƥ��ؼ�֡�������㣬���Ҹõ��ж�Ӧ�ĵ�ͼ�㣬��������ͼ��ϲ�
//2. �����ͼ����ƥ��ؼ�֡�������㣬���Ҹõ�û�ж�Ӧ�ĵ�ͼ�㣬��ôΪ�õ���ӵ�ͼ��
int ORBmatcher::Fuse(KeyFrame *pKF,  //�ؼ�֡ 
						const vector<MapPoint *> &vpMapPoints,  //��ͼ��
						const float th) //th = 3.0 �����뾶
{
	//��ȡ��֡��ѡ������R
    cv::Mat Rcw = pKF->GetRotation();
	//��ȡ��֡��ƽ������T
    cv::Mat tcw = pKF->GetTranslation();

	//��ȡ����ڲ�
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

	//��ȡ�������λ�þ���
    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

	//��ͼ��ĸ���
    const int nMPs = vpMapPoints.size();

	//����ÿһ����ͼ��
    for(int i=0; i<nMPs; i++)
    {
    	//��ȡ��ͼ��
        MapPoint* pMP = vpMapPoints[i];

		//null
        if(!pMP)
            continue;

		//��ͼ��Ϊ���㣬��֡�Ǹõ�ͼ��Ĺ۲�֡
        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

		//��ȡ�õ�ͼ�����������
        cv::Mat p3Dw = pMP->GetWorldPos();
		//�����������3D ��任����������3D ��
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        //��ȱ���Ϊ��
        if(p3Dc.at<float>(2)<0.0f)
            continue;

		//�������3D ��ͶӰ�����ƽ���2D ��
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //���ͶӰ�ĵ��Ƿ���ͼ����
        if(!pKF->IsInImage(u,v))
            continue;

		//����������ͷ����
        const float ur = u-bf*invz;

		//��ȡ��ͼ�����ȷ�Χ
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
		//�����ͼ�㵽������ĵ�����
        cv::Mat PO = p3Dw-Ow;
		//��������ģ��С
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        //����Ƿ��ڷ�Χ��
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        //��ȡ��ͼ���ƽ���۲ⷽ��
        cv::Mat Pn = pMP->GetNormal();

		//|Pn| = 1
		//�����ͼ��۲�н��Ƿ�С��60��
        if(PO.dot(Pn)<0.5*dist3D)
            continue;


		//ͨ������Ԥ�⣬��ͼ����ͼ�����������һ��
        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        //���ݵ�ͼ�����ȷ���߶ȣ��Ӷ�ȷ��������Χ
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

		//��ȡ�ùؼ�֡�ϸ÷�Χ�ڵ����йؼ���
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

		//��ȡ��ͼ��������
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
		//����������Χ�ڵ�����������
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//��ȡ��������
            const size_t idx = *vit;

			//��ȡ��������
            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

			//��ȡͼ���������
            const int &kpLevel= kp.octave;

			//�жϸò��Ƿ���nPredictedLevel ����nPredictedLevel -1
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

			//˫Ŀ����ͷ
            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                //��ȡ����������
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
				//�����������
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

				//���������ֱ������
                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

			//��ȡ�������������
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

			//�����ͼ��������������ӵĺ�������
            const int dist = DescriptorDistance(dMP,dKF);


			//�ҵ������Ӿ��������
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        //�������С����ֵ
        if(bestDist<=TH_LOW)
        {
        	//��ȡ������������ĵ�ͼ��
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
            	//��Ӧ�ĵ�ͼ����OK��
            	//��ͼ����кϲ�
                if(!pMPinKF->isBad())
                {
                	//�ĸ���ͼ�㱻�۲�Ķ࣬����˭
                    if(pMPinKF->Observations()>pMP->Observations())
						//�ùؼ�֡��ͼ��������е�ͼ��
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
			//�����������û�й�����ͼ��
            else 
            {
            	//Ϊ��ͼ����ӹ۲�֡�Ͷ�Ӧ������
                pMP->AddObservation(pKF,bestIdx);
				//Ϊ�ؼ�֡��ӵ�ͼ��
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

//ͨ��sim3 �任��ȷ��pkf1 ����������pkf2�еĴ�������ͬ��
//ȷ��pkf2 ����������pkf1 �еĴ�������
//�ڸ�����ͨ�������ӽ���ƥ�䲶��pkf1��pkf2 ֮ǰ©ƥ��������㣬
//����ƥ���
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, 				//f1 ֡
								KeyFrame *pKF2, 				//f2 ֡
								vector<MapPoint*> &vpMatches12,  //��Ҫƥ��ĵ�ͼ��
                             	const float &s12, 				//sim3 s  ����ϵ��
                             	const cv::Mat &R12, 			//f2-->f1 ����ת����
                             	const cv::Mat &t12, 			//f2-->f1 ��ƽ�ƾ���
                             	const float th)					//��ֵ 7.5
{
	//��ȡ����ͷ�ڲ�
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    //��ȡf1 �����R ����
    cv::Mat R1w = pKF1->GetRotation();
	//��ȡf1 �����t ����
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    //��ȡf2 �����R ����
    cv::Mat R2w = pKF2->GetRotation();
	//��ȡf2 �����t ����
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    //[sR  t]
    //����sR
    cv::Mat sR12 = s12*R12;
	//�����f1-->f2  ����ת����
    cv::Mat sR21 = (1.0/s12)*R12.t();
	//�����f1-->f2 ��ƽ�ƾ���
    cv::Mat t21 = -sR21*t12;

	//��ȡf1 ����������ĵ�ͼ��
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
	//��ȡ������ʵ��f1 ������ĸ���
    const int N1 = vpMapPoints1.size();

	//��ȡf2 ����������ĵ�ͼ��
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
	//��ȡ������ʵ��f2 ������ĸ���
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

	//����f1 �����е�ͼ��
    for(int i=0; i<N1; i++)
    {
    	//��ȡ��ͼ��
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
        	//�ɹ���ȡ��ͼ�������Ѿ�ƥ���־
            vbAlreadyMatched1[i]=true;
			//��ȡ�õ�ͼ����f2 ���������idx
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
			//�õ�ͼ���f2 Ҳ�����ˣ�����f2 ƥ���־
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    //��f1�ĵ�ͼ��ת����f2 �ϣ���������
    for(int i1=0; i1<N1; i1++)
    {
    	//��ȡ��ͼ��
        MapPoint* pMP = vpMapPoints1[i1];

		//����Ѿ�ƥ������
        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

		//��ȡ��ͼ�����������ϵ
        cv::Mat p3Dw = pMP->GetWorldPos();
		//��f1 ��ͼ�������ϵ����������ת����f1 �������
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
		//ͨ��sim3 ����������ֱ�ӵ���ת��ƽ�ƾ���
		//��f1 �������ϵ�µĵ�ת����f2 �������ϵ��
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        //��ͼ����Ȳ���С��0
        if(p3Dc2.at<float>(2)<0.0)
            continue;

		//ͨ���������ϵ�µĵ�ͼ��������ƽ�����������
        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //����Ƿ���f2 ͼ����Ч������
        if(!pKF2->IsInImage(u,v))
            continue;

		//��ȡ�õ�ͼ��ľ��뷶Χ
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
		//����õ�ͼ���ģ
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        //��ͼ��ľ����Ƿ��ڷ�Χ��
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        //ͨ���������õ�ͼ����f2 �Ͽ��ܵĽ�������
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        //���������뾶��ֵ
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

		//��f2 �ϸ���ֵ���������е�������
        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

		//û���ҵ�һ��������
        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //��ȡ�õ�ͼ���������
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
		//������f2 ��������������������
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//��ȡ�������idex
            const size_t idx = *vit;

			//��ȡ����������
            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

			//�������������ڵĽ��������Ƿ���Ԥ���Ĳ����һ����
            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

			//��ȡ���������������
            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

			//����������ĺ͵�ͼ��ĺ�������
            const int dist = DescriptorDistance(dMP,dKF);

			//��ȡ����������С�ĵ�������
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

		//����С��100
        if(bestDist<=TH_HIGH)
        {
        	//ƥ��ɹ�����¼ƥ�䵽��idx
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF1 and search
    //��f2 �ĵ�ͼ��ת����f1 �ϣ�����ƥ������
    for(int i2=0; i2<N2; i2++)
    {
    	//��ȡf2 �ĵ�ͼ��
        MapPoint* pMP = vpMapPoints2[i2];

		//�����ͼ�㲻���ڻ����Ѿ�ƥ�䣬������
        if(!pMP || vbAlreadyMatched2[i2])
            continue;

		//�����ͼ���ǻ���������
        if(pMP->isBad())
            continue;
		//��ȡ��ͼ�����������3D
        cv::Mat p3Dw = pMP->GetWorldPos();
		//�ѵ�ͼ����������3D ��ת����f2 ��������µ�3D  ��
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
		//ͨ��sim3 �仯����f2 ��������µ�3D ��ת����f1 ��������µ�3D ��
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        //��ȱ������0
        if(p3Dc1.at<float>(2)<0.0)
            continue;

		//ͨ��f1��������µ�3D ����f1 ���ƽ���Ӧ�����ص�
        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //����������ص��Ƿ���f1 �������Ч���ص���
        if(!pKF1->IsInImage(u,v))
            continue;

		//��ȡ��ͼ�������ȷ�Χ
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
		//��ȡ�õ�ͼ����f1 ��������µľ���
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        //�����ڷ�Χ��������
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        //���Ƹõ�Ӧ����f1 �������Ĳ���
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        //���������뾶
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

		//f1 ��ָ���������������е�������
        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

		//��������������Ϊ0 ������
        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //��ȡ��ͼ���������
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
		//����������������������
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//��ȡ�������idx
            const size_t idx = *vit;

			//��ȡ�����������
            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

			//������Ľ��������Ƿ���Ԥ���Ĳ����һ����
            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

			//��ȡ���������������
            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

			//�����ͼ��͸�������������ӵĺ�������
            const int dist = DescriptorDistance(dMP,dKF);

			//�ҵ�f1 ����õ�ͼ�㺺��������С��������
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

		//�������С��100, ƥ��ɹ�
        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

	//����ƥ���������f1 ��f2 ƥ��������ͬ�ĵ㣬����ƥ��ɹ�
    for(int i1=0; i1<N1; i1++)
    {
    	//��ȡf1 ������������� ��ͼ��ƥ�䵽��f2 ��������
        int idx2 = vnMatch1[i1];

		//ƥ�����������
        if(idx2>=0)
        {
        	//��ȡf2 ��Ӧ����������f1 ��ƥ�䵽�ĵ�ͼ�������������
            int idx1 = vnMatch2[idx2];
			//�����Ƿ���ȣ���ȱ�ʾ��ƥ�䵽���ң���Ҳƥ�䵽����
            if(idx1==i1)
            {
            	//��ƥ�䵽�ĵ�ͼ����й���
                vpMatches12[i1] = vpMapPoints2[idx2];
				//ƥ�䵽�ĸ�����1
                nFound++;
            }
        }
    }

    return nFound;
}

//����һ֡ÿ��3D ��ͨ��ͶӰ��С��Χ���ҵ���ƥ���2D ��
//�Ӷ�ʵ�ֵ�ǰ֡����һ֡3D ���ƥ�����
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, //��ǰ֡
										const Frame &LastFrame, //��һ֡
										const float th, 		//��ֵ��ƥ��������Ĵ��ڴ�С
										const bool bMono)		//�Ƿ�Ϊ��Ŀ
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    //360���30 ������ ��ÿ������ 12��
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
	//������������
    const float factor = 1.0f/HISTO_LENGTH;

	//���㵱ǰ֡��תR
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
	//���㵱ǰ֡ƽ��T
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw;

	//��ȡ���һ֡����ת��ƽ��
    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat tlc = Rlw*twc+tlw;

	//�ж�ǰ�����Ǻ��ˣ����Դ�Ԥ���������ڵ�ǰ֡���ڵĽ���������
	//�ǵ�Ŀ������£����z ���ڻ��ߣ����ʾǰ��
    const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;
	//�ǵ�Ŀ����£����z С�ڻ��ߣ����ʾ����
    const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;

	//�������һ֡������
    for(int i=0; i<LastFrame.N; i++)
    {
    	//��ȡ��ͼ��
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

		//��ͼ�����
        if(pMP)
        {
        	//���õ�ͼ�㲻����㣬�ڵ�Ϊ��Ч��
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                //��ȡ����������������ϵ
                cv::Mat x3Dw = pMP->GetWorldPos();
				//��������3D �㵽�������3D ��
                cv::Mat x3Dc = Rcw*x3Dw+tcw;
				//xc = X, yc = Y , invzc = 1/Z
                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                if(invzc<0)
                    continue;

				//�������ƽ���Ӧ��2D ������
				//xscreen = fx*(X/Z)  + cx,  yscreen = fy*(X/Z) + cy
                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

				//�ж���������ƽ��������Ƿ��ڵ�ǰ֡��Ч������
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

				//��ȡ��ǰ֡����������
                int nLastOctave = LastFrame.mvKeys[i].octave;

                // Search in a window. Size depends on scale
                //��ȡƥ�䷶Χ�İ뾶�����ڴ�С
                //�߶�Խ��������ΧԽ��
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;

				//�������ƽ������Ϊ���ģ�radius Ϊ���ڰ뾶Ѱ�Ҷ�Ӧ�Ĺؼ�֡
                if(bForward)
					//����������ǰ�˶� ��������ͻᱻ�Ŵ󣬻�ȡ���������
					//ͼ������������ͻ�Խ��
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
					//������������˶���������ͻᱻ��С����ȡ����������
					//��ͼ��������ͻ�Խ��
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
					//ͼ���������ÿ��ƥ��
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

				//û��ƥ�䵽������
                if(vIndices2.empty())
                    continue;

				//��ȡ�õ�ͼ���Ӧ��������
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

				//���������ҵ���������
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;
					//������������Ѿ��ж�Ӧ�ĵ�ͼ���ˣ��ͼ���
                    if(CurrentFrame.mvpMapPoints[i2])
						//���۲⵽�õ�ͼ��ؼ�֡�ĸ����Ƿ����0
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

					//�����˫Ŀ�����
                    if(CurrentFrame.mvuRight[i2]>0)
                    {
                    	//��Ҫ��֤��ͼ�ĵ�Ҳ��������Χ��
                    	//����ͶӰ���� ��ͼ��Ӧ��λ��
                        const float ur = u - CurrentFrame.mbf*invzc;
						//������ͼͶӰ�����Ӧ�������Ƿ��ڴ�����
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);

						//������ڴ�����ֱ������
                        if(er>radius)
                            continue;
                    }

					//��ȡ��ǰ֡ƥ��ɹ����������������
                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

					//�����ͼ��͸�������������Ӻ�������
                    const int dist = DescriptorDistance(dMP,d);

					//�ҵ�����������С�ĵ�
                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

				//Ѱ�ҵ�����С����С����ֵ
                if(bestDist<=TH_HIGH)
                {
                	//�ҵ��õ�ͼ���ڵ�ǰ֡��Ӧ�Ĺؼ���
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
					//ƥ������1
                    nmatches++;

					//������
                    if(mbCheckOrientation)
                    {
                    	//����������ķ����
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
						//��������
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
						//���뵽��Ӧ������
                        rotHist[bin].push_back(bestIdx2); 
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    //ͨ�������⣬�޳���ƥ���
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

		//����3������������
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                	//ɾ����ƥ��������
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

	//����ƥ��ɹ��ĵ�ͼ�����
    return nmatches;
}

int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

//��histo �����е�3 ������ֵ,��������Ӧ��������
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
    	//��ȡhisto ��i ��Ԫ��������ֻ�ĸ���
        const int s = histo[i].size();
        if(s>max1)//���s > max1 , ��Ҫ���¸� max1 max2 max3 ��ֵ
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2) //���s > max2, ��Ҫ���¸�max2 max3 ��ֵ
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)//���s > max3, ��Ҫ���¸�max3 ��ֵ
        {
            max3=s;
            ind3=i;
        }
    }

	//���max2 С��max1 ��1/10, ��Ҫmax2 max3
    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }  //���max3 С��max1��1/10�� ��Ҫmax3
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
//�������������ӵĺ�������
//�ú��������������������м���1�� ����Խ��˵�����������Խ��
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

	//32 * 8 = 256
    for(int i=0; i<8; i++, pa++, pb++)
    {
    	//���� *pa = 0111 0111,
    	//              *pb = 1011 0110
    	//		       v = 1100   0001
        unsigned  int v = *pa ^ *pb;
		//v >> 1 	    							      = 0110 0000
		//0x55555555 = 0101 0101 0101 0101 0101 0101 0101 0101
		//(v >> 1) & 0x55555555 = 0100 0000
		//v - ((v >> 1) & 0x55555555) = 1100 0001 - 
		//							     0100 0000
		//v =   10  00  00  01�� 10��ʾ��������2��1�� 01�� ��ʾ��������1��1
		//��32λ��Ϊ16��,��ÿһ�����м���1
        v = v - ((v >> 1) & 0x55555555);

		//v 					       = 1000 0001
		//0x33333333 = 0011 0011 0011 0011
		//v&0x33333333 = 0000 0001
		//v >> 2 			       = 0010 0000
		//0x33333333 = 0011 0011 0011 0011
		//(v >> 2) & 0x33333333 = 0010 0000
		//v = 0000 0001 +
		//       0010 0000
		//v=  0010 0001   //0010  ��ʾv = 1100 0001 ǰ4λ��2��1�� 0001�� ��ʾv = 1100 0001  ����λ��1��1
		//��32λ��Ϊ8��,������ʵ����ԭ�����Ǹ���ÿ4����λ���м���1.
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		//v = 0010 0001
		//v>>4 = 0000 0010
		//v + (v >> 4) = 0010 0001 +
		//			       0000 0010 
		// 		               							    =0010 0011
		//0xF0F0F0F = 0000 1111 0000 1111 0000 1111 0000 1111
		//(v + (v >> 4)) & 0xF0F0F0F 			            =0000 0011
		//((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101 = 0001 0000 0001 0000 0001 0000 0001 +
		//										      0010 0000 0010 0000 0010 0000 0010					       		 
		//                                                                              = 0011 0000 0011 0000 0011 0000 0011
		//dist+ = 0000 0011 ��ʾv = 1100   0001 ����3��1
		//��32λ�ֳ�2�飬 ÿ��16��λ����֮ǰv ���м���1
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
