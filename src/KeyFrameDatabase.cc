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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}


//根据关键帧的词包，更新数据库的倒排索引
//为每一个视觉词袋的word 添加关键帧
void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

//关键帧被删除后，更新数据库的倒排索引
void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    //每一个关键帧包含多个words, 遍历mvInvertedFile 中的这些words, 然后在word 中删除关键帧
    //遍历该帧的所有视觉词袋
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
    	//在mvInvertedFile 中获取到视觉词袋关键帧的list
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

		//遍历该单词的list , 删除该关键帧
        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
        {
            if(pKF==*lit)
            {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

//在闭环检测中找到与该关键帧可能闭环的关键帧
//1. 找出和当前关键帧具有公共单词的所有关键帧(不包括与当前帧相连的关键帧)
//2. 只和具有共同单词较多的关键帧进行相似度计算
//3. 将与关键帧(权重最高) 的前10 个关键帧归为一组，计算累计得分
//4. 只返回累计得分较高的组中分数最高的关键帧
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
	//获取当前关键帧的共视关键帧
	//这里相连的关键帧都是局部相连
	//在闭环检测的时候将被剔除
	//这里使用set 不用vector 是set 中没有重复元素，方便查找
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames();
	//这里用list 不用vector 是list 不需要分配的空间连续，vector 需要连续空间
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    //找出和当前具有公共单词的所有关键帧(不包括当前帧共视的关键帧)
    {
        unique_lock<mutex> lock(mMutex);

		//遍历该关键帧的每一个视觉词袋的word
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++)
        {
        	//提取所有包含该word 的关键帧队列
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];


			//遍历关键帧队列
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
            	//获取队列中的关键帧
                KeyFrame* pKFi=*lit;
				//该帧还没有标记为候选帧
                if(pKFi->mnLoopQuery!=pKF->mnId)
                {
                    pKFi->mnLoopWords=0;
					//查看该帧是否跟当前帧是共视帧
					//共视帧不参与回环检测
                    if(!spConnectedKeyFrames.count(pKFi))
                    {
                    	//把该帧加入到候选帧
                        pKFi->mnLoopQuery=pKF->mnId;
						//放入lKFsSharingWords 队列
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
				//记录该帧和当前帧具有相同的word 个数
                pKFi->mnLoopWords++;
            }
        }
    }

	//如果没有找到直接返回null
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
	//遍历找到的所有关键帧
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
    	//找到与当前帧word  最多的个数
        if((*lit)->mnLoopWords>maxCommonWords)
            maxCommonWords=(*lit)->mnLoopWords;
    }

	//阈值设置，最小的word 个数 0.8 倍最多的个数
    int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    //遍历找到的所有的关键帧
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
    	//获取关键帧
        KeyFrame* pKFi = *lit;

		//minCommonWords 和maxCommonWords  阈值范围内的帧进行相似度得分计算
        if(pKFi->mnLoopWords>minCommonWords)
        {
            nscores++;

			//计算相似度得分
            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec);

			//记录该帧和当前帧的相似度
            pKFi->mLoopScore = si;
			//相似度大于最小相似度
            if(si>=minScore)
				//把该帧放入队列中
                lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    //在这一批得分很高的帧中，获取每个帧相似度最大的10 个共视帧
    //统计在这10个共视帧中有多少也是当前帧的可能闭环备选帧
    //进行相似度累计得分
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
    	//获取队列中关键帧
        KeyFrame* pKFi = it->second;
		//获取该关键帧的10 个共视帧
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

		//用当前帧的得分初始化得分统计
        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
		//遍历每一个共视帧
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
        	//获取共视帧
            KeyFrame* pKF2 = *vit;
			//查看该共视帧是否是当前帧闭环候选帧，
			//同时满足，得分大于minCommonWords
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
            	//累计得分
                accScore+=pKF2->mLoopScore;
				//距离得分最大的帧
                if(pKF2->mLoopScore>bestScore)
                {
                	//保存得分最好的帧
                    pBestKF=pKF2;
					//保存最高得分
                    bestScore = pKF2->mLoopScore;
                }
            }
        }
		//把每一组的累计得分和单个得分最高的帧成对，push 到lAccScoreAndMatch 中
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
		//记录最近累计得分
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    //得分阈值计算
    float minScoreToRetain = 0.75f*bestAccScore;

	//set 可以很好的插入元素， 不能有重复
    set<KeyFrame*> spAlreadyAddedKF;
	//vector访问元素很快，可以由重复
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

	//遍历找到的每组中累计得分最高和当个得分最高的帧
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
    	//进一步缩小可能范围
        if(it->first>minScoreToRetain)
        {
        	//获取帧
            KeyFrame* pKFi = it->second;
			//查找spAlreadyAddedKF 是否已经包含该帧
			//这里的set 容器主要是不让有重复的
            if(!spAlreadyAddedKF.count(pKFi))
            {
            	//没有包含该帧
            	//把该帧push 到向量中
                vpLoopCandidates.push_back(pKFi);
				//把该帧插入到set 集合中，防止有重复
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

//在重定位中找到与该帧相似的关键帧
// 1. 找出和当前帧具有公共单词的所有关键帧
// 2. 只和具有共同单词较多的关键帧进行相似度计算
// 3. 将与关键帧相连(权值最高) 的前10 个关键帧归为一组，计算累计得分
// 4. 只返回累计得分较高的组中分数最高的关键帧
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
	//相对于关键帧的闭环检测DetectLoopCandidates, 重定位检测中没法获得相连的关键帧
	//用于保存可能与F 形成回环的候选帧(只要有相同的word, 且不属于局部相连帧)
    list<KeyFrame*> lKFsSharingWords;

	//找出和当前帧具有公共单词的所有关键帧
    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

		//words  是检测图像是否匹配的枢纽，遍历当前帧的每一个word
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
        	//找到word 对应的帧list
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first];

			//遍历所有的帧list
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++)
            {
            	//获取关键帧
                KeyFrame* pKFi=*lit;
				//该帧还没有标记为F 的候选帧
                if(pKFi->mnRelocQuery!=F->mnId)
                {
                    pKFi->mnRelocWords=0;
					//标记重定位候选帧
                    pKFi->mnRelocQuery=F->mnId;
					//加入候选帧
                    lKFsSharingWords.push_back(pKFi);
                }
				//该帧累计得分加1
                pKFi->mnRelocWords++;
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
	//找到最高的积分
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnRelocWords>maxCommonWords)
            maxCommonWords=(*lit)->mnRelocWords;
    }

	//最高的积分0.8 设置阈值
    int minCommonWords = maxCommonWords*0.8f;

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    //遍历所有闭环候选帧，把满足要求的帧和得分进行匹配
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit;

        if(pKFi->mnRelocWords>minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec);
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi));
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    //计算候选帧组得分，得到最高组得分， 并以此决定阈值
    //单单计算当前帧和某一关键帧的相似度是不够的，这里将于关键帧相连(权值最高，共视程度最高)
    //的前十个关键帧归为一组，计算累计得分
    //具体而言，每一个候选关键帧都把与自己共视程度较高的帧归为一组，每一组会计算得分并记录该
    //组分数最高的关键帧
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
    	//获取关键候选帧
        KeyFrame* pKFi = it->second;
		//获取该候选帧共视程度最高的10帧
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first; //该组最高分数
        float accScore = bestScore; //该组累计得分
        KeyFrame* pBestKF = pKFi;  //该组最高分数对应的关键帧
        //遍历所有的共视帧
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
        	//获取共视帧
            KeyFrame* pKF2 = *vit;
			//该帧是候选帧，就跳过
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

			//该组累计得分
            accScore+=pKF2->mRelocScore;
			//统计得到组里分数最高的关键帧
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
		//把最高的关键帧和组对应累计得分匹配放到list 中
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
		//记录所有组的最高分数
		if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    //得到组得分大于阈值的，组内得分最高的关键帧
    //阈值设定
    float minScoreToRetain = 0.75f*bestAccScore;
    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
	//遍历组得分
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
    	//获取得分
        const float &si = it->first;
		//得分大于阈值
        if(si>minScoreToRetain)
        {
        	//获取关键帧
            KeyFrame* pKFi = it->second;
			//防止重复添加
            if(!spAlreadyAddedKF.count(pKFi))
            {
            	//把候选帧插入向量
                vpRelocCandidates.push_back(pKFi);
				//防止重复添加
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
