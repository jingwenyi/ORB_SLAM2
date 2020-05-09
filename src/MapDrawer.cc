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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

}

//显示地图点
void MapDrawer::DrawMapPoints()
{
	//取出所有地图点
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
	//取出局部地图点
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

	//将向量转换成set 容器类型，便于使用set::count 快速统计
    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

	//显示所有的地图点(不包括局部地图点)  ，大小为两个像素，黑色
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
	//颜色黑色
    glColor3f(0.0,0.0,0.0);

	//遍历全部地图点
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
    	//如果该地图点是坏的，或者是局部地图点，跳过
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
		//获取地图点的3D 坐标
        cv::Mat pos = vpMPs[i]->GetWorldPos();
		//用opengl 显示3D 点
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();

	//显示局部地图点，大小为2 个像素，红色
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

	//遍历局部地图点
    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
		//获取局部地图的坐标
        cv::Mat pos = (*sit)->GetWorldPos();
		//opengl 显示局部地图
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));

    }

    glEnd();
}

//显示关键帧
void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
	//历史关键帧图标: 宽度占总宽度比例为0.05
    const float &w = mKeyFrameSize;
	//高度
    const float h = w*0.75;
	//深度
    const float z = w*0.6;
	//取出所有的关键帧
    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

	//显示所有关键帧图标
	//通过显示界面 选择是否显示历史关键帧图标
    if(bDrawKF)
    {
    	//遍历所有的关键帧
        for(size_t i=0; i<vpKFs.size(); i++)
        {
        	//获取关键帧
            KeyFrame* pKF = vpKFs[i];
			//获取该帧的位姿的逆的转置
			//转置，opengl中的矩阵为列优先存储
            cv::Mat Twc = pKF->GetPoseInverse().t();

            glPushMatrix();

			//由于使用了glPushMatrix 函数，因此当前帧的矩阵为世界坐标系下的单位矩阵
			//因为opengl 中的矩阵为列优先存储，因此实际为tcw, 机相机在时间坐标下的位姿
            glMultMatrixf(Twc.ptr<GLfloat>(0));

			//设置绘制图形时线的宽度
            glLineWidth(mKeyFrameLineWidth);
			//设置当前颜色为蓝色(关键帧图标显示为蓝色)
            glColor3f(0.0f,0.0f,1.0f);
			//用线将下面的顶点两两相连
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

	//显示所有关键帧位姿图
	//通过显示界面选择是否显示关键帧连接关系
    if(bDrawGraph)
    {
    	//设置绘制图形时线的宽度
        glLineWidth(mGraphLineWidth);
		//设置共视图连接线为绿色，透明度为0.6f
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

		//遍历所有关键帧
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            //共视程度比较高的共视关键帧用线连接
            //获取该关键帧的共视关键帧
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
			//获取该帧在世界坐标系下的相机坐标
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
            	//遍历共视程度高的关键帧
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                	//如果该共视帧在该帧的前面，跳过
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
					//获取该共视帧在世界坐标系下的相机坐标
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
					//显示该帧和共视帧
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            //连接最小生成树
            //获取当前帧的父关键帧
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
            	//获取父关键帧在世界坐标系下的相机坐标
                cv::Mat Owp = pParent->GetCameraCenter();
				//显示该帧和父关键帧
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            //连接闭环时形成的连接关系
            //获取该帧闭环连接的边的集合
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
			//遍历所有连接边对应的帧
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
            	//该帧在vpKFs[i] 帧的前面，跳过
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
				//获取该帧在世界坐标系下的相机坐标
                cv::Mat Owl = (*sit)->GetCameraCenter();
				//显示这两帧
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

//绘制当前摄像机
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
	//相机模型大小: 宽度占总宽度比例为0.08
    const float &w = mCameraSize;
	//计算高度
    const float h = w*0.75;
	//计算深度
    const float z = w*0.6;

    glPushMatrix();

	//将4*4的矩阵twc.m 右乘一个当前矩阵
	//由于使用了glPushMatrix函数，因此当前帧矩阵为世界坐标系下的单位矩阵
	//因为opengl 中的矩阵为列优先存储，因此实际为，tcw, 机相机在世界坐标系下的位姿
#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

	//设置绘制图形时线的宽度
    glLineWidth(mCameraLineWidth);
	//设置当前颜色为绿色( 相机图标显示为绿色 )
    glColor3f(0.0f,1.0f,0.0f);
	//用线将下面的顶点两两相连
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}


//将相机位姿由Mat 类型转化为OpenGlMatrix 类型
void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

} //namespace ORB_SLAM
