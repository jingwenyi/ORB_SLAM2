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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

//orb 特征点由关键点和描述子组成，描述子256个位
const int ORBmatcher::TH_HIGH = 100;//匹配较高阈值 100   个汉明距离
const int ORBmatcher::TH_LOW = 50; //匹配较低阈值50  个汉明距离
const int ORBmatcher::HISTO_LENGTH = 30;//将360°分成30个bin, 每个bin 12°

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}


//对于每个局部3D 点通过投影在小范围内找到最匹配的2D 点
//从而实现frame 对本地地图的追踪
int ORBmatcher::SearchByProjection(Frame &F,  //当前帧
										const vector<MapPoint*> &vpMapPoints,  // 本地地图
										const float th)		//搜索范围因子
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

	//获取局部地图地图点
    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
    	//获取地图点
        MapPoint* pMP = vpMapPoints[iMP];
		//判断该点是否要被投影
        if(!pMP->mbTrackInView)
            continue;

		//该地图点标志位是否可用
        if(pMP->isBad())
            continue;

		//获取地图点在金字塔的层数
        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        //根据观测到3D 点的视角确定搜索窗口的大小
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

		//调整搜索窗口
        if(bFactor)
            r*=th;

		//在当前帧中对应图像金字塔的窗口中与该投影点进行匹配
		//找出该区域内的所有特征点
        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

		//没有匹配到
        if(vIndices.empty())
            continue;

		//获取地图点对于的描述子
        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        //对粗略匹配得到的特征进行描述子匹配
        //找到最佳匹配和次佳匹配，如果最优匹配误差小于阈值，
        //且最优匹配明显优于次佳匹配则地图3D 点和当前帧特征2D 点匹配成功
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//获取搜索到的特征点坐标
            const size_t idx = *vit;

			//如果frame 兴趣点，已经有对应的地图点，则退出该次循环
            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

			//双目还有检查右图像
            if(F.mvuRight[idx]>0)
            {
            	//就算地图重投影点跟对应特征点的距离
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
				//距离误差比较
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

			//计算该特征点的描述子
            const cv::Mat &d = F.mDescriptors.row(idx);
			//计算地图点描述子和特征点描述子的汉明距离
            const int dist = DescriptorDistance(MPdescriptor,d);

			//找到最小距离
            if(dist<bestDist)
            {
                bestDist2=bestDist;		//次小距离
                bestDist=dist;			//最小距离
                bestLevel2 = bestLevel;  //次小距离对应的金字塔层
				//最小距离对应的金字塔层
                bestLevel = F.mvKeysUn[idx].octave;
				//最小距离对应的特征点序号
                bestIdx=idx;
            }
            else if(dist<bestDist2) //次小距离
            {
            	//次小距离金字塔层数
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        //最小距离小于阈值
        if(bestDist<=TH_HIGH)
        {	//最小距离和次小距离在同一个金字塔层，
        	//最小距离不小于0.9倍次小距离， 匹配失败，继续
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

			//匹配成功，为frame 中的兴趣点增加地图点
            F.mvpMapPoints[bestIdx]=pMP;
			//成功匹配个数
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998) //正视
        return 2.5;
    else				//斜视
        return 4.0;
}

//求出kp1 在pKF2 上对应的极线
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image
    //l = x1'F12 = [a b c]
    //分别求a b c
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

	//计算kp2 特征点到极线的距离
	//极线方程l : ax + by + c =0
	//(u,v) 到l 的距离为|av + bv +c| / sqrt(a^2 + b^)
    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

	//距离的平方
    const float dsqr = num*num/den;

	//尺度越大，范围应该也越大
	//金字塔最低层一个像素点就占一个像素点，
	//在倒数第二层，一个像素点等于1.2个像素点，假设金字塔尺度为1.2
    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

//用视觉词袋对参考关键帧和当前帧进行关键点进行匹配
//每一个关键可能对于有一个地图点，关键点匹配成功，
//当前帧对应的地图点也就找到了
int ORBmatcher::SearchByBoW(KeyFrame* pKF, //参考帧
								Frame &F, 		//当前帧
								vector<MapPoint*> &vpMapPointMatches)  //当前帧跟踪成功的地图点
{
	//获取参考帧地图点
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

	//初始化当前帧地图点
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

	//获取参考帧视觉词袋的特征向量
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

	//把360°划分了30个区间，每个区间12°
    vector<int> rotHist[HISTO_LENGTH];
	//为没干过区间申请空间
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
	//计算比例因子
    const float factor = 1.0f/HISTO_LENGTH;


    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    //获取参考帧视觉词袋特征迭代开始地址
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
	//获取当前帧视觉词袋特征迭代开始位置
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
	//获取参考帧视觉词袋特征迭代结束位置
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
	//获取当前帧视觉词袋特征迭代结束位置
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

	
    while(KFit != KFend && Fit != Fend)
    {
    	//分别取出同一node 的ORB 特征点
    	//只有属于同一node ,才有可能是匹配点
        if(KFit->first == Fit->first)
        {
        	//获取kf   视距词袋中特征点向量
            const vector<unsigned int> vIndicesKF = KFit->second;
			//获取当前帧视距词袋中特征点的向量
            const vector<unsigned int> vIndicesF = Fit->second;

			//遍历kf 中属于该node 的特征点
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
            	//获取kf 对应特征点的idx
                const unsigned int realIdxKF = vIndicesKF[iKF];

				//通过特征点的idx  获取对应的地图点
                MapPoint* pMP = vpMapPointsKF[realIdxKF];

				//地图点不存在
                if(!pMP)
                    continue;
				//地图点是坏点
                if(pMP->isBad())
                    continue;                

				//取出参考关键帧该特征点对于的描述子
                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1=256;//最小距离
                int bestIdxF =-1 ;//记录当前帧中匹配到的最佳特征点
                int bestDist2=256;//第二小距离

				//遍历当前帧属于该node 的特征点
                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
                	//获取当前帧对应特征点的idex
                    const unsigned int realIdxF = vIndicesF[iF];

					//如果当前帧该特征点已经匹配到地图点
                    if(vpMapPointMatches[realIdxF])
                        continue;
					//获取当前帧该特征点对应的描述子
                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);

					//计算参考帧特征点和当前帧特征点描述子的汉明距离
                    const int dist =  DescriptorDistance(dKF,dF);

					//找出对于参考关键帧的一个特征点， 对于的当前帧中特征点
					//汉明距离最小和第二小的特征点
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

				//根据阈值和角度投票，剔除误匹配
                if(bestDist1<=TH_LOW)
                {
                	//static_cast 类型强制转换符，mfNNratio = 0.9
                	//如果最小距离小于0.9 倍第二小的距离
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                    	//当前帧特征点匹配成功，保存对于地图点
                        vpMapPointMatches[bestIdxF]=pMP;

						//获取参考关键帧该关键点
                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

						//方向检查
                        if(mbCheckOrientation)
                        {
                        	//angle:每个特征点在提取描述子时的旋转主方向角度，
                        	//如果图形旋转了，这个角度将发生改变，
                        	//所有有特征点的角度变化应该是一致的，
                        	//通过直方图统计得到最准确的角度变化值
                        	//关键帧角度相减，得到该特征点的角度变化
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;
                            if(rot<0.0)
                                rot+=360.0f;
							//四舍五入
                            int bin = round(rot*factor); 
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
							//将rot 分配到bin 组
                            rotHist[bin].push_back(bestIdxF);
                        }
						//匹配个数自加
                        nmatches++;
                    }
                }

            }

			//node 自加
            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
        	//参考关键帧 找到与当前帧对应的node
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
        	//当前帧找到与参考关键帧对应的node
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


	//根据方向剔除误匹配点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

		//找出向量中3个极大值
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
            	//剔除方向异常的地图点，static_cast 类型强制转换
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
				//匹配点个数自减
                nmatches--;
            }
        }
    }

    return nmatches;
}

//用于闭环检测中将地图点和关键帧的特征点进行关联
//根据sim3 变换，将每个地图点，投影到pkf 上，并根据尺度确定一个搜索区域
//根据该地图点的描述子与该区域内的特征点进行匹配，如果匹配误差小于50 ，
//即匹配成功，更新匹配
int ORBmatcher::SearchByProjection(KeyFrame* pKF, //闭环检测的当前帧
										cv::Mat Scw, 	//时间坐标系到当前帧的sim3 变换矩阵
										const vector<MapPoint*> &vpPoints, //闭环检测的地图点
										vector<MapPoint*> &vpMatched, //匹配成功的地图点
										int th)							//阈值  10
{
    // Get Calibration Parameters for later projection
    //获取相机内参
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    //scwd的形式为[sR, st]
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
	//计算得到尺度s
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
	//R = sR / s
    cv::Mat Rcw = sRcw/scw;
	//t = st / t
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
	//世界坐标系下， pkf 到世界坐标系的位姿，方向由pkf指向世界坐标系
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    //把闭环的地图点放入set 集合中，加速查找匹配
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
	//删掉集合中的所有null 的地图点
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    //遍历每一个地图点，为地图点找到匹配的特征点
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
    	//获取地图点
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        //丢弃坏的地图点和已经匹配的地图点
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        //获取该地图点的世界坐标3D 点
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        //把地图点转到摄像机坐标系
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        //深度必须大于0， 在设计及前方的点
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        //计算相机平面上对应的相机坐标
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //检查计算的相机坐标是否在图像有效范围内
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        //获取地图点到相机的深度范围
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
		//计算地图点到相机中心的向量
        cv::Mat PO = p3Dw-Ow;
		//计算向量的模
        const float dist = cv::norm(PO);

		//该距离是否在深度范围内
        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        //平均观察方向
        cv::Mat Pn = pMP->GetNormal();

		//计算该帧对地图点的观测方向与地图点的平均观测方向的夹角， 大于60 表示太大
		//cos(a,b) = a.b/|a||b|, |Pn| = 1
        if(PO.dot(Pn)<0.5*dist)
            continue;

		//通过距离预测地图点在当前帧图像金字塔的那一层
        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        //计算搜索半径阈值
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

		//找出关键帧在阈值内的所有特征点
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

		//一个特征点都没有找到
        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //获取地图点的描述子
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
		//遍历找到的所有的特征点
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//获取特征点的idx
            const size_t idx = *vit;
			//查看pkf中该特征点是否已经匹配地图点
            if(vpMatched[idx])
                continue;

			//获取特征点的图像金字塔层
            const int &kpLevel= pKF->mvKeysUn[idx].octave;

			//检查该特征点的图像金字塔层是否在之前估计的层中或者下一层
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

			//获取该特征点的描述子
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

			//计算描述子的汉明距离
            const int dist = DescriptorDistance(dMP,dKF);

			//找到汉明距离最小的那个
            if(dist<bestDist)
            {
                bestDist = dist;
				//对应的特征点idx
                bestIdx = idx;
            }
        }

		//如果检测到的最小的汉明距离小于50
        if(bestDist<=TH_LOW)
        {
        	//给特征点添加对应的地图点
            vpMatched[bestIdx]=pMP;
			//匹配成功加1
            nmatches++;
        }

    }

	//返回匹配成功数
    return nmatches;
}

int ORBmatcher::SearchForInitialization(Frame &F1,  								//参考帧
										   Frame &F2, 								//当前帧
										   vector<cv::Point2f> &vbPrevMatched, 		//参考帧关键点的坐标
										   vector<int> &vnMatches12, 				//需要匹配点的个数
										   int windowSize)							//关键点的窗口大小
{
    int nmatches=0;
	//设置需要匹配点个数为参考帧的关键点个数
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

	//把360度分成30 个bin , 这里是为每一个bin 申请保留空间
	//每个bin 对应12° 的区间
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
	//比例因子
    const float factor = 1.0f/HISTO_LENGTH;

	//当前帧每个关键匹配的最近距离向量
    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
	//当前帧匹配到参考帧关键点的关联向量
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
    	//顺序获取关键点
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;// 提取该特征点的在金字塔那一层获取
		//只处理金字塔第0 层提取到的关键点，即原图
		if(level1>0)
            continue;

		//把参考帧的关键点放到当前帧中， 
		// 在以参考帧为中心的100 为半径 的窗口中找到全部的关键点
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

		//没有匹配到，继续下一个关键点的匹配
        if(vIndices2.empty())
            continue;
		//获取参考帧 关键点的 orb 描述子
        cv::Mat d1 = F1.mDescriptors.row(i1);

		//距离最短
        int bestDist = INT_MAX;
		//距离倒数第二短
        int bestDist2 = INT_MAX;
		//记录距离最短的idx
        int bestIdx2 = -1;

		//遍历匹配到的每一个关键帧
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;
			//获取当前帧该关键点的orb 描述子
            cv::Mat d2 = F2.mDescriptors.row(i2);
			//获取两个关键帧描述子的距离
            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

			//找到最小的前两个距离
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

		//确保最小距离小于阈值50
        if(bestDist<=TH_LOW)
        {
        	//再确保最小距离要小于次小距离乘以mfNNratio=0.9
            if(bestDist<(float)bestDist2*mfNNratio)
            {
            	//如果已经匹配
                if(vnMatches21[bestIdx2]>=0)
                {
                	//移除匹配
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
				//把参考帧关键点与当前帧关键点关联
                vnMatches12[i1]=bestIdx2;
				//把当前帧关键点与参考帧关键点关联
                vnMatches21[bestIdx2]=i1;
				//保存当前帧关键点匹配的最小距离
                vMatchedDistance[bestIdx2]=bestDist;
				//匹配数加1
                nmatches++;

				//是否检查方向
                if(mbCheckOrientation)
                {
                	//参考帧关键点的方向- 当前帧关键点的方向
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
					//把方向整理到0-360之间
                    if(rot<0.0)
                        rot+=360.0f;

					//计算该方向差在哪个bin 中
                    int bin = round(rot*factor);
					//把当前整的该关键点push 到对应的bin 中
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

	//进行方向匹配
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
		//找出bin 中元素个数最大的3个bin区间
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

		//把不是这3个最大个数的bin 匹配的关键点都移除点
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
    //通过参考帧的坐标传出与当前帧匹配到的关键点的坐标
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;
	//返会匹配到的个数
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

//利用基本矩阵f12 在pKF1和pKF2 之间找特征匹配
//作用: 当pKF1 中特征点没有对应的3D 点时通过匹配的特征点产出新的3D  点
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, 
											KeyFrame *pKF2, 								//
											cv::Mat F12,									//  相机基础矩阵
                                       		vector<pair<size_t, size_t> > &vMatchedPairs,  	// 传出匹配成功队
                                       		const bool bOnlyStereo)	//在双目或rgbd 情况下要求特征点在右图存在匹配
{    
	//获取两帧图像词袋的特征向量
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    //计算kf1 的相机中心到kf2 图像平面的坐标， 即极点坐标
    //获取f1 相机中心
    cv::Mat Cw = pKF1->GetCameraCenter();
	//获取f2 旋转  
    cv::Mat R2w = pKF2->GetRotation();
	//获取f2 的平移矩阵
    cv::Mat t2w = pKF2->GetTranslation();
	//f1 相机中心在f2 坐标系中的表示
    cv::Mat C2 = R2w*Cw+t2w;
	//得到kf1在kf2 的极点坐标
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


	
	//使用视觉词袋加速比较
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

	//遍历pkf1 和pkf2 的node 节点
    while(f1it!=f1end && f2it!=f2end)
    {
    	//如果flit  和f2it 属于同一个node
        if(f1it->first == f2it->first)
        {
        	//遍历该node 下的所有特征点
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
            	//获取f1 属于node 的特征点索引
                const size_t idx1 = f1it->second[i1];

				//获取f1 对应的地图点
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                //如果该特征点已经有对应的地图点，就继续
                //我们寻找的是未匹配特征点，所有应该是null
                if(pMP1)
                    continue;


				//mvuRight 值大于0 表示是双目，而且特征点有深度
                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

				//不考虑深度的有效性
                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                //取出f1 对应的特征点
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];

				//取出特征点对于的描述子
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;

				//遍历node 节点下f2 的所有特征点
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                	//获取f2 特征点索引
                    size_t idx2 = f2it->second[i2];

					//获取对应的地图点
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    //如果已经匹配或者该特征点地图点存在，跳过
                    if(vbMatched2[idx2] || pMP2)
                        continue;

					//双目
                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    //获取f2 该特征点的描述子
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

					//计算f1 特征点和f2 特征点描述子的汉明距离
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

					//获取f2 的特征点
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
						//该特征点距离极点太近，表示f2 该特征点对应的地图点离f1 相机太近
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

					//计算特征点kp2 到kp1 的极线的距离是否小于阈值
					//kp1 对于pkf2 的一条极线
					//极线约束
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                    	//找到最合适的匹配点，
                    	//满足极线约束：汉明距离最小的点
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

				//f1 中没有地图点的关键点匹配到一个f2 的没有地图的关键点
                if(bestIdx2>=0)
                {
                	//获取f2 关键点的坐标
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
					//保存匹配成功的点
                    vMatches12[idx1]=bestIdx2;
					//匹配个数加1
                    nmatches++;

					//角度检查
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

	//进行角度检查，剔除错误匹配点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

		//找出3个角度最大的
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
			//剔除误匹配点
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
            	//匹配对象-1
                vMatches12[rotHist[i][j]]=-1;
				//匹配成功对--
                nmatches--;
            }
        }

    }

	//
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

	//把成功匹配对放到vMatchedPairs 向量中
    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

//将地图点投影到关键帧中，查看是否有重复的地图点
//1. 如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，将两个地图点合并
//2. 如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加地图点
int ORBmatcher::Fuse(KeyFrame *pKF,  //关键帧 
						const vector<MapPoint *> &vpMapPoints,  //地图点
						const float th) //th = 3.0 搜索半径
{
	//获取该帧的选择向量R
    cv::Mat Rcw = pKF->GetRotation();
	//获取该帧的平移向量T
    cv::Mat tcw = pKF->GetTranslation();

	//获取相机内参
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

	//获取相机中心位置矩阵
    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

	//地图点的个数
    const int nMPs = vpMapPoints.size();

	//遍历每一个地图点
    for(int i=0; i<nMPs; i++)
    {
    	//获取地图点
        MapPoint* pMP = vpMapPoints[i];

		//null
        if(!pMP)
            continue;

		//地图点为坏点，该帧是该地图点的观测帧
        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

		//获取该地图点的世界坐标
        cv::Mat p3Dw = pMP->GetWorldPos();
		//把世界坐标的3D 点变换到相机坐标的3D 点
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        //深度必须为正
        if(p3Dc.at<float>(2)<0.0f)
            continue;

		//把相机的3D 点投影到相机平面的2D 点
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //检查投影的点是否在图像内
        if(!pKF->IsInImage(u,v))
            continue;

		//计算右摄像头坐标
        const float ur = u-bf*invz;

		//获取地图点的深度范围
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
		//计算地图点到相机中心的向量
        cv::Mat PO = p3Dw-Ow;
		//计算向量模大小
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        //深度是否在范围内
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        //获取地图点的平均观测方向
        cv::Mat Pn = pMP->GetNormal();

		//|Pn| = 1
		//计算地图点观测夹角是否小于60°
        if(PO.dot(Pn)<0.5*dist3D)
            continue;


		//通过距离预测，第图点在图像金字塔的那一层
        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        //根据地图点深度确定尺度，从而确地搜索范围
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

		//获取该关键帧上该范围内的所有关键点
        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

		//获取地图的描述子
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
		//遍历搜索范围内的所有特征点
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//获取特征点编号
            const size_t idx = *vit;

			//获取特征坐标
            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

			//获取图像金字塔层
            const int &kpLevel= kp.octave;

			//判断该层是否在nPredictedLevel 或者nPredictedLevel -1
            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

			//双目摄像头
            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                //获取特征点坐标
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
				//计算坐标误差
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

				//如果误差过大，直接跳过
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

			//获取特征点的描述子
            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

			//计算地图点和特征点描述子的汉明距离
            const int dist = DescriptorDistance(dMP,dKF);


			//找到描述子距离最近的
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        //最近距离小于阈值
        if(bestDist<=TH_LOW)
        {
        	//获取该特征点关联的地图点
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
            	//对应的地图点是OK的
            	//地图点进行合并
                if(!pMPinKF->isBad())
                {
                	//哪个地图点被观测的多，就用谁
                    if(pMPinKF->Observations()>pMP->Observations())
						//用关键帧地图点代替现有地图点
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
			//如果该特征点没有关联地图点
            else 
            {
            	//为地图点添加观测帧和对应特征点
                pMP->AddObservation(pKF,bestIdx);
				//为关键帧添加地图点
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

//通过sim3 变换，确定pkf1 的特征点在pkf2中的大致区域，同理
//确定pkf2 的特征点在pkf1 中的大致区域
//在该区域通过描述子进行匹配捕获pkf1和pkf2 之前漏匹配的特征点，
//更新匹配对
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, 				//f1 帧
								KeyFrame *pKF2, 				//f2 帧
								vector<MapPoint*> &vpMatches12,  //需要匹配的地图点
                             	const float &s12, 				//sim3 s  比例系数
                             	const cv::Mat &R12, 			//f2-->f1 的旋转矩阵
                             	const cv::Mat &t12, 			//f2-->f1 的平移矩阵
                             	const float th)					//阈值 7.5
{
	//获取摄像头内参
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    //获取f1 相机的R 矩阵
    cv::Mat R1w = pKF1->GetRotation();
	//获取f1 相机的t 矩阵
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    //获取f2 相机的R 矩阵
    cv::Mat R2w = pKF2->GetRotation();
	//获取f2 相机的t 矩阵
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    //[sR  t]
    //计算sR
    cv::Mat sR12 = s12*R12;
	//计算从f1-->f2  的旋转矩阵
    cv::Mat sR21 = (1.0/s12)*R12.t();
	//计算从f1-->f2 的平移矩阵
    cv::Mat t21 = -sR21*t12;

	//获取f1 特征点关联的地图点
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
	//获取个数其实是f1 特征点的个数
    const int N1 = vpMapPoints1.size();

	//获取f2 特征点关联的地图点
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
	//获取个数其实是f2 特征点的个数
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

	//遍历f1 的所有地图点
    for(int i=0; i<N1; i++)
    {
    	//获取地图点
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
        	//成功获取地图点设置已经匹配标志
            vbAlreadyMatched1[i]=true;
			//获取该地图点在f2 中特征点的idx
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
			//该地图点跟f2 也对上了，设置f2 匹配标志
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    //把f1的地图点转换到f2 上，进行搜索
    for(int i1=0; i1<N1; i1++)
    {
    	//获取地图点
        MapPoint* pMP = vpMapPoints1[i1];

		//如果已经匹配跳过
        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

		//获取地图点的世界坐标系
        cv::Mat p3Dw = pMP->GetWorldPos();
		//把f1 地图点的坐标系从世界坐标转换到f1 相机坐标
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
		//通过sim3 计算出的相机直接的旋转和平移矩阵
		//把f1 相机坐标系下的点转换到f2 相机坐标系下
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        //地图点深度不能小于0
        if(p3Dc2.at<float>(2)<0.0)
            continue;

		//通过相机坐标系下的地图点计算相机平面的像素坐标
        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //检查是否在f2 图像有效像素内
        if(!pKF2->IsInImage(u,v))
            continue;

		//获取该地图点的距离范围
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
		//计算该地图点的模
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        //地图点的距离是否在范围内
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        //通过距离计算该地图点在f2 上可能的金字塔层
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        //计算搜索半径阈值
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

		//在f2 上该阈值内搜索所有的特征点
        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

		//没有找到一个特征点
        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //获取该地图点的描述子
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
		//遍历在f2 上搜索到的所有特征点
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//获取特征点的idex
            const size_t idx = *vit;

			//获取特征点坐标
            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

			//计算特征点所在的金字塔层是否在预估的层或下一层上
            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

			//获取该特征点的描述子
            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

			//计算特征点的和地图点的汉明距离
            const int dist = DescriptorDistance(dMP,dKF);

			//获取汉明距离最小的点和其距离
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

		//距离小于100
        if(bestDist<=TH_HIGH)
        {
        	//匹配成功，记录匹配到的idx
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF1 and search
    //把f2 的地图点转换到f1 上，进行匹配搜索
    for(int i2=0; i2<N2; i2++)
    {
    	//获取f2 的地图点
        MapPoint* pMP = vpMapPoints2[i2];

		//如果地图点不存在或者已经匹配，则跳过
        if(!pMP || vbAlreadyMatched2[i2])
            continue;

		//如果地图点是坏点则跳过
        if(pMP->isBad())
            continue;
		//获取地图点的世界坐标3D
        cv::Mat p3Dw = pMP->GetWorldPos();
		//把地图点世界坐标3D 点转换到f2 相机坐标下的3D  点
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
		//通过sim3 变化，把f2 相机坐标下的3D 点转换到f1 相机坐标下的3D 点
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        //深度必须大于0
        if(p3Dc1.at<float>(2)<0.0)
            continue;

		//通过f1相机坐标下的3D 点求f1 相机平面对应的像素点
        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        //检查所求像素点是否在f1 相机的有效像素点内
        if(!pKF1->IsInImage(u,v))
            continue;

		//获取地图坐标的深度范围
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
		//获取该地图点在f1 相机坐标下的距离
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        //距离在范围外则跳过
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        //估计该点应该在f1 金字塔的层数
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        //计算搜索半径
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

		//f1 该指定区域内搜索所有的特征点
        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

		//搜索到的特征点为0 ，跳过
        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        //获取地图点的描述子
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
		//遍历搜索到的所有特征点
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
        	//获取特征点的idx
            const size_t idx = *vit;

			//获取特征点的坐标
            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

			//特征点的金字塔层是否在预估的层后下一层上
            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

			//获取该特征点的描述子
            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

			//计算地图点和该特征点的描述子的汉明距离
            const int dist = DescriptorDistance(dMP,dKF);

			//找到f1 上与该地图点汉明距离最小的特征点
            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

		//如果距离小于100, 匹配成功
        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

	//遍历匹配结果，如果f1 和f2 匹配结果都相同的点，才算匹配成功
    for(int i1=0; i1<N1; i1++)
    {
    	//获取f1 的特征点关联的 地图点匹配到的f2 的特征点
        int idx2 = vnMatch1[i1];

		//匹配特征点存在
        if(idx2>=0)
        {
        	//获取f2 对应的特征点在f1 上匹配到的地图点关联的特征点
            int idx1 = vnMatch2[idx2];
			//二者是否相等，相等表示你匹配到了我，我也匹配到了你
            if(idx1==i1)
            {
            	//把匹配到的地图点进行关联
                vpMatches12[i1] = vpMapPoints2[idx2];
				//匹配到的个数加1
                nFound++;
            }
        }
    }

    return nFound;
}

//对上一帧每个3D 点通过投影在小范围内找到最匹配的2D 点
//从而实现当前帧对上一帧3D 点的匹配跟踪
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, //当前帧
										const Frame &LastFrame, //上一帧
										const float th, 		//阈值，匹配点搜索的窗口大小
										const bool bMono)		//是否为单目
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    //360°分30 个区间 ，每个区间 12°
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
	//计算缩放因子
    const float factor = 1.0f/HISTO_LENGTH;

	//计算当前帧旋转R
    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
	//计算当前帧平移T
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat twc = -Rcw.t()*tcw;

	//获取最后一帧的旋转和平移
    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3);

    const cv::Mat tlc = Rlw*twc+tlw;

	//判断前进还是后退，并以此预测特征点在当前帧所在的金字塔层数
	//非单目的情况下，如果z 大于基线，则表示前进
    const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;
	//非单目情况下，如果z 小于基线，则表示后退
    const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;

	//遍历最后一帧特征点
    for(int i=0; i<LastFrame.N; i++)
    {
    	//获取地图点
        MapPoint* pMP = LastFrame.mvpMapPoints[i];

		//地图点存在
        if(pMP)
        {
        	//检查该地图点不是外点，内点为有效点
            if(!LastFrame.mvbOutlier[i])
            {
                // Project
                //获取该坐标点的世界坐标系
                cv::Mat x3Dw = pMP->GetWorldPos();
				//世界坐标3D 点到相机坐标3D 点
                cv::Mat x3Dc = Rcw*x3Dw+tcw;
				//xc = X, yc = Y , invzc = 1/Z
                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                if(invzc<0)
                    continue;

				//计算相机平面对应的2D 点坐标
				//xscreen = fx*(X/Z)  + cx,  yscreen = fy*(X/Z) + cy
                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

				//判断所求的相机平面坐标点是否在当前帧有效坐标内
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

				//获取当前帧金字塔层数
                int nLastOctave = LastFrame.mvKeys[i].octave;

                // Search in a window. Size depends on scale
                //获取匹配范围的半径，窗口大小
                //尺度越大，搜索范围越大
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                vector<size_t> vIndices2;

				//以求出的平面坐标为中心，radius 为窗口半径寻找对应的关键帧
                if(bForward)
					//如果相机是向前运动 ，特征点就会被放大，获取到特征点的
					//图像金字塔层数就会越高
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave);
                else if(bBackward)
					//如果相机是向后运动，特征点就会被缩小，获取到的特征点
					//的图像金字塔就会越低
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave);
                else
					//图像金字塔的每层匹配
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1);

				//没有匹配到特征点
                if(vIndices2.empty())
                    continue;

				//获取该地图点对应的描述子
                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

				//遍历所有找到的特征点
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit;
					//如果该特征点已经有对应的地图点了，就继续
                    if(CurrentFrame.mvpMapPoints[i2])
						//检查观测到该地图点关键帧的个数是否大于0
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                            continue;

					//如果是双目的情况
                    if(CurrentFrame.mvuRight[i2]>0)
                    {
                    	//需要保证右图的点也在搜索范围内
                    	//计算投影点在 右图对应的位置
                        const float ur = u - CurrentFrame.mbf*invzc;
						//计算右图投影点与对应特征点是否在窗口内
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);

						//如果不在窗口内直接跳过
                        if(er>radius)
                            continue;
                    }

					//获取当前帧匹配成功的特征点的描述子
                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

					//计算地图点和该特征点的描述子汉明距离
                    const int dist = DescriptorDistance(dMP,d);

					//找到汉明距离最小的点
                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

				//寻找到的最小距离小于阈值
                if(bestDist<=TH_HIGH)
                {
                	//找到该地图点在当前帧对应的关键点
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
					//匹配数加1
                    nmatches++;

					//方向检查
                    if(mbCheckOrientation)
                    {
                    	//计算特征点的方向差
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
						//四舍五入
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
						//插入到对应的区域
                        rotHist[bin].push_back(bestIdx2); 
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    //通过方向检测，剔除误匹配点
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

		//计算3个最大的区域方向
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                	//删除误匹配的坐标点
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }
    }

	//返回匹配成功的地图点个数
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

//求histo 向量中的3 个极大值,并传出对应的索引号
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
    	//获取histo 第i 个元素向量中只的个数
        const int s = histo[i].size();
        if(s>max1)//如果s > max1 , 需要重新给 max1 max2 max3 赋值
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2) //如果s > max2, 需要重新给max2 max3 赋值
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)//如果s > max3, 需要重新给max3 赋值
        {
            max3=s;
            ind3=i;
        }
    }

	//如果max2 小于max1 的1/10, 则不要max2 max3
    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }  //如果max3 小于max1的1/10， 则不要max3
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
//计算两个描述子的汉明距离
//该函数是求两个矩阵异或后有几个1， 距离越大，说明描述子相差越大
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

	//32 * 8 = 256
    for(int i=0; i<8; i++, pa++, pb++)
    {
    	//假如 *pa = 0111 0111,
    	//              *pb = 1011 0110
    	//		       v = 1100   0001
        unsigned  int v = *pa ^ *pb;
		//v >> 1 	    							      = 0110 0000
		//0x55555555 = 0101 0101 0101 0101 0101 0101 0101 0101
		//(v >> 1) & 0x55555555 = 0100 0000
		//v - ((v >> 1) & 0x55555555) = 1100 0001 - 
		//							     0100 0000
		//v =   10  00  00  01， 10表示改组中有2个1， 01， 表示该组中有1个1
		//将32位分为16组,看每一组中有几个1
        v = v - ((v >> 1) & 0x55555555);

		//v 					       = 1000 0001
		//0x33333333 = 0011 0011 0011 0011
		//v&0x33333333 = 0000 0001
		//v >> 2 			       = 0010 0000
		//0x33333333 = 0011 0011 0011 0011
		//(v >> 2) & 0x33333333 = 0010 0000
		//v = 0000 0001 +
		//       0010 0000
		//v=  0010 0001   //0010  表示v = 1100 0001 前4位有2个1， 0001， 表示v = 1100 0001  后四位有1个1
		//将32位分为8组,看的其实还是原来的那个数每4个单位里有几个1.
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
		//dist+ = 0000 0011 表示v = 1100   0001 中有3个1
		//将32位分成2组， 每组16个位，看之前v 中有几个1
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
