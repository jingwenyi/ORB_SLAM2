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

//��ʾ��ͼ��
void MapDrawer::DrawMapPoints()
{
	//ȡ�����е�ͼ��
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
	//ȡ���ֲ���ͼ��
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

	//������ת����set �������ͣ�����ʹ��set::count ����ͳ��
    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

	//��ʾ���еĵ�ͼ��(�������ֲ���ͼ��)  ����СΪ�������أ���ɫ
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
	//��ɫ��ɫ
    glColor3f(0.0,0.0,0.0);

	//����ȫ����ͼ��
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
    	//����õ�ͼ���ǻ��ģ������Ǿֲ���ͼ�㣬����
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
		//��ȡ��ͼ���3D ����
        cv::Mat pos = vpMPs[i]->GetWorldPos();
		//��opengl ��ʾ3D ��
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();

	//��ʾ�ֲ���ͼ�㣬��СΪ2 �����أ���ɫ
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

	//�����ֲ���ͼ��
    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
		//��ȡ�ֲ���ͼ������
        cv::Mat pos = (*sit)->GetWorldPos();
		//opengl ��ʾ�ֲ���ͼ
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));

    }

    glEnd();
}

//��ʾ�ؼ�֡
void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
	//��ʷ�ؼ�֡ͼ��: ���ռ�ܿ�ȱ���Ϊ0.05
    const float &w = mKeyFrameSize;
	//�߶�
    const float h = w*0.75;
	//���
    const float z = w*0.6;
	//ȡ�����еĹؼ�֡
    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

	//��ʾ���йؼ�֡ͼ��
	//ͨ����ʾ���� ѡ���Ƿ���ʾ��ʷ�ؼ�֡ͼ��
    if(bDrawKF)
    {
    	//�������еĹؼ�֡
        for(size_t i=0; i<vpKFs.size(); i++)
        {
        	//��ȡ�ؼ�֡
            KeyFrame* pKF = vpKFs[i];
			//��ȡ��֡��λ�˵����ת��
			//ת�ã�opengl�еľ���Ϊ�����ȴ洢
            cv::Mat Twc = pKF->GetPoseInverse().t();

            glPushMatrix();

			//����ʹ����glPushMatrix ��������˵�ǰ֡�ľ���Ϊ��������ϵ�µĵ�λ����
			//��Ϊopengl �еľ���Ϊ�����ȴ洢�����ʵ��Ϊtcw, �������ʱ�������µ�λ��
            glMultMatrixf(Twc.ptr<GLfloat>(0));

			//���û���ͼ��ʱ�ߵĿ��
            glLineWidth(mKeyFrameLineWidth);
			//���õ�ǰ��ɫΪ��ɫ(�ؼ�֡ͼ����ʾΪ��ɫ)
            glColor3f(0.0f,0.0f,1.0f);
			//���߽�����Ķ�����������
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

	//��ʾ���йؼ�֡λ��ͼ
	//ͨ����ʾ����ѡ���Ƿ���ʾ�ؼ�֡���ӹ�ϵ
    if(bDrawGraph)
    {
    	//���û���ͼ��ʱ�ߵĿ��
        glLineWidth(mGraphLineWidth);
		//���ù���ͼ������Ϊ��ɫ��͸����Ϊ0.6f
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

		//�������йؼ�֡
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            //���ӳ̶ȱȽϸߵĹ��ӹؼ�֡��������
            //��ȡ�ùؼ�֡�Ĺ��ӹؼ�֡
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
			//��ȡ��֡����������ϵ�µ��������
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
            	//�������ӳ̶ȸߵĹؼ�֡
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                	//����ù���֡�ڸ�֡��ǰ�棬����
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
					//��ȡ�ù���֡����������ϵ�µ��������
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
					//��ʾ��֡�͹���֡
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            //������С������
            //��ȡ��ǰ֡�ĸ��ؼ�֡
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
            	//��ȡ���ؼ�֡����������ϵ�µ��������
                cv::Mat Owp = pParent->GetCameraCenter();
				//��ʾ��֡�͸��ؼ�֡
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            //���ӱջ�ʱ�γɵ����ӹ�ϵ
            //��ȡ��֡�ջ����ӵıߵļ���
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
			//�����������ӱ߶�Ӧ��֡
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
            	//��֡��vpKFs[i] ֡��ǰ�棬����
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
				//��ȡ��֡����������ϵ�µ��������
                cv::Mat Owl = (*sit)->GetCameraCenter();
				//��ʾ����֡
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

//���Ƶ�ǰ�����
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
	//���ģ�ʹ�С: ���ռ�ܿ�ȱ���Ϊ0.08
    const float &w = mCameraSize;
	//����߶�
    const float h = w*0.75;
	//�������
    const float z = w*0.6;

    glPushMatrix();

	//��4*4�ľ���twc.m �ҳ�һ����ǰ����
	//����ʹ����glPushMatrix��������˵�ǰ֡����Ϊ��������ϵ�µĵ�λ����
	//��Ϊopengl �еľ���Ϊ�����ȴ洢�����ʵ��Ϊ��tcw, ���������������ϵ�µ�λ��
#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

	//���û���ͼ��ʱ�ߵĿ��
    glLineWidth(mCameraLineWidth);
	//���õ�ǰ��ɫΪ��ɫ( ���ͼ����ʾΪ��ɫ )
    glColor3f(0.0f,1.0f,0.0f);
	//���߽�����Ķ�����������
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


//�����λ����Mat ����ת��ΪOpenGlMatrix ����
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
