import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
import json

exercise_num = 306
knowledge_point_num = 25

## create table

underPath = os.getcwd() + '\\'

chapter4_target = pd.read_excel(underPath + '初中物理8年级教研精选练习目标描述.xlsx')
# 每个知识点对应的题目数量: {"target_code": num }
chapter4_requirement = dict(zip(chapter4_target['target_code'], chapter4_target['建议配题数量']))
# {id : "knowledge/target_code"}
chapter4_knowledge_itos = dict(zip([i for i in range(25)], chapter4_target['target_code']))

chapter4_exercises_file = pd.read_excel(underPath + '初中物理8年级教研精选题306题.xlsx')
chapter4_exercises = chapter4_exercises_file['exercise_code']

pre = ''
# 章节四的每个知识点有多少题量 {"knowlege" : num}
chapter4_exercises_num = {}
for exercise in chapter4_exercises:
    if pre != exercise[:-3]:
        pre = exercise[:-3]
        chapter4_exercises_num[pre] = 1
    else:
        chapter4_exercises_num[exercise[:-3]] += 1

# 用户列表
# userlist = {0: "test"}
# userlist_stoi = {"test": 0}
with open('userlist.json','r',encoding='utf-8') as f:
    userlist = json.load(f)

with open('userlist_stoi.json','r',encoding='utf-8') as f:
    userlist_stoi = json.load(f)

def initTable():
    # 用户的做题记录表
    userExerciseTable = np.zeros([1, exercise_num, 2])
    # 用户知识点学习情况
    userKnowledgeTable = np.zeros([1, knowledge_point_num, 2])
    return userExerciseTable,userKnowledgeTable

#添加用户的知识点和习题记录
def addUserTable(UserExerciseTable,UserKnowledgeTable):
    newUserExerciseTable, newUserrKnowledgeTable = initTable()
    UserExerciseTable = np.concatenate((UserExerciseTable,newUserExerciseTable),axis=0)
    UserKnowledgeTable = np.concatenate((UserKnowledgeTable,newUserrKnowledgeTable),axis=0)
    return UserExerciseTable,UserKnowledgeTable

def login(username, userlist):
    if username in userlist.values():
        print("登录成功")
        return True
    else:
        print("登录失败")
        return False


def signIn(userlist,userExerciseTable,userKnowledgeTable):

    while True:
        username = input("输入注册用户名(输入exit退出注册界面)：")
        if username == 'exit':
            break
        if username in userlist.values():
            print("用户名已经存在")
        else:
            userlist[len(userlist)] = username
            userlist_stoi[username] = len(userlist_stoi)
            print("注册成功")
            # 更新表单
            return addUserTable(userExerciseTable, userKnowledgeTable)


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
def randomExercise(knowledgeId, num=None):
    # 如果exerciseId=None 则随机抽取
    # 如果不为空则在原有题号的基础上更新为下一题
    knowledge = chapter4_knowledge_itos[knowledgeId]
    if num == None:
        num = np.random.randint(chapter4_exercises_num[knowledge])
    else:
        num = (num + 1) % chapter4_exercises_num[knowledge]
    return computeExerciseId(knowledgeId, num)


def computeExerciseId(knowledgeId, num):
    exerciseId = 0
    for i in range(knowledgeId):
        knowledge = chapter4_knowledge_itos[i]
        exerciseId += chapter4_exercises_num[knowledge]
    return exerciseId + num


def judgeExerciseScore(exercise):
    if computeExerciseScore(exercise) > 0.5:
        return True
    else:
        return False


def computeExerciseScore(exercise):
    # exercise = [rigth times, wrong times]
    return ((exercise[0] + 1) / (exercise[1] + 2))

def fetchExercise(exerciseId):
    table = chapter4_exercises_file.loc[[exerciseId]]
    exerciseTitle = table[['title']].values[0][0]
    exerciseAnswer = table[['right_answer']].values[0][0]
    exerciseTip = table[['study_tips']].values[0][0]
    return exerciseTitle,exerciseAnswer,exerciseTip






def run():
    userExerciseTablePath = Path(underPath+'userExerciseTable.npz')

    #加载所有用户信息
    if userExerciseTablePath.is_file():
        userExerciseTable = np.load('userExerciseTable.npz')['arr_0']
        userKnowledgeTable= np.load('userKnowledgeTable.npz')['arr_0']
    else:
        userExerciseTable, userKnowledgeTable = initTable()

    haveAcount = input("有账号的用户输入任意键跳过，没有账号的用户输入‘注册’：")

    if haveAcount == '注册':
        #返回新的table
        userExerciseTable, userKnowledgeTable = signIn(userlist,userExerciseTable,userKnowledgeTable)

    username = input("输入你的用户名")

    if (login(username, userlist)):
        userId = userlist_stoi[username]
        # 加载用户的信息
        userKnowledge = userKnowledgeTable[userId]
        userExercise = userExerciseTable[userId]

        # 判断知识点学习情况
        knowledgeId = judgeKnowledge(userKnowledge)
        print(f"knowledgeId:{knowledgeId}")
        if knowledgeId == knowledge_point_num:
            print("本章知识点学习完成")

        # 用户做完所有知识点的题目时结束
        while (knowledgeId != knowledge_point_num):
            # 随机抽取指定知识点的题目
            exerciseId = randomExercise(knowledgeId)
            # 判断随机的题目是否掌握
            while judgeExerciseScore(userExercise[exerciseId]):
                exerciseId = randomExercise(knowledgeId,exerciseId)
            # 显示题目
            print(f"exerciseId:{exerciseId}")
            exerciseTitle, exerciseAnswer, exerciseTip = fetchExercise(exerciseId)
            print(exerciseTitle)
            # 做题
            print(f"请作答第{exerciseId+2}题：(输入'exit'停止作答)")
            userAnswer = input()

            if userAnswer == 'exit':
                break

            # 显示正确答案
            if userAnswer == exerciseAnswer:
                print("回答正确")
                # 更新用户数据
                userExercise[exerciseId][0]+=1
                userKnowledge[knowledgeId][0]+=1
            else:
                print("回答错误")
                # 更新用户数据
                userExercise[exerciseId][1] += 1
                userKnowledge[knowledgeId][1] += 1
            print(f"正确答案:{exerciseAnswer}")
            #if not np.isnan(exerciseTip):
            print(f"{exerciseTip}")
            print(f"{input('按任意键进入下一题:')}")
            print("*"*20)

            # 判断知识点学习情况
            knowledgeId = judgeKnowledge(userKnowledge)
            print(f"knowledgeId:{knowledgeId}")
            if knowledgeId == knowledge_point_num:
                print("本章知识点学习完成")

        # 更新用户信息
        userKnowledgeTable[userId] = userKnowledge
        userExerciseTable[userId] = userExercise

        # 存储用户信息
        np.savez('userKnowledgeTable',userKnowledgeTable)
        np.savez('userExerciseTable', userExerciseTable)

        # 存储用户名列表
        with open('userlist.json','w',encoding='utf-8') as fw:
            json.dump(userlist,fw)
        with open('userlist_stoi.json','w',encoding='utf-8') as fw:
            json.dump(userlist_stoi,fw)

run()
