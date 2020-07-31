import numpy as np
import pandas as pd
import re
import os

exercise_num = 306
knowledge_point_num = 25

## create table
underPath = os.getcwd() + '\\'

chapter4_target = pd.read_excel(underPath+'初中物理8年级教研精选练习目标描述.xlsx')
# 每个知识点对应的题目数量: {"target_code": num }
chapter4_requirement = dict(zip(chapter4_target['target_code'],chapter4_target['建议配题数量']))
# {id : "knowledge/target_code"}
chapter4_knowledge_itos = dict(zip([i for i in range(25)],chapter4_target['target_code']))

chapter4_exercises = pd.read_excel(underPath+'初中物理8年级教研精选题306题.xlsx')
chapter4_exercises = chapter4_exercises['exercise_code']

pre = ''
# 章节四的每个知识点有多少题量 {"knowlege" : num}
chapter4_exercises_num = {}
for exercise in chapter4_exercises:
    if pre!= exercise[:-3]:
        pre=exercise[:-3]
        chapter4_exercises_num[pre] = 1
    else:
        chapter4_exercises_num[exercise[:-3]] +=1


#用户列表
userlist = {0:"test"}
userlist_stoi = {"test":0}
# 用户的做题记录表
userExerciseTable = np.zeros([1,exercise_num,2])
# 用户知识点学习情况
userKnowledgeTable = np.zeros([1,knowledge_point_num,2])


def login(username,userlist):
    if username in userlist.values():
        print("登录成功")
        return True
    else:
        print("登录失败")
        return False

def signIn(username,userlist):
    if username in userlist.values():
        print("用户名已经存在")
    else:
        userlist[len(userlist)] = username
        userlist_stoi[username] = len(userlist_stoi)
        print("注册成功")


# 按顺序查询用户的知识点学习情况

def judgeKnowledge(userKnowledge):
    # 判断知识点的学习情况，返回还需要推题的知识点
    # 如果都要求的配题都完成了，返回 knowledge_point_num

    for id in range(knowledge_point_num):
        knowledge = chapter4_knowledge_itos[id]
        if userKnowledge[id][0] >= chapter4_requirement[knowledge]:
            continue
        else:
            return id
        return knowledge_point_num


# 随机抽取知识点的题目
def randomExercise(knowledgeId):
    knowledge = chapter4_knowledge_itos[knowledgeId]
    num = np.random.randint(chapter4_exercises_num[knowledge])
    return computeExerciseId(knowledgeId, num)


def computeExerciseId(knowledgeId, num):
    exerciseId = 0
    for i in range(knowledgeId):
        knowledge = chapter4_knowledge_itos[i]
        exerciseId += chapter4_exercises_num[knowledge]
    return exerciseId + num


def computeExerciseScore(exercise):
    # exercise = [rigth times, wrong times]
    return ((exercise[0] + 1) / (exercise[1] + 2))


def run():
    username = 'test'
    if (login(username, userlist)):
        userId = userlist_stoi[username]
        # 加载用户的信息
        userKnowledge = userKnowledgeTable[userId]
        userExercise = userExerciseTable[userId]

        # 判断知识点学习情况
        knowledgeId = judgeKnowledge(userKnowledge)

        if knowledgeId == knowledge_point_num:
            print("本章知识点学习完成")
            return

        while(knowledgeId != knowledge_point_num):

            # 随机抽取指定知识点的题目
            exerciseId = randomExercise(knowledgeId)



run()




