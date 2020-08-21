import numpy as np
import pandas as pd
import re
import os
from pathlib import Path
import json
import math
from tqdm import tqdm
import time
from operator import itemgetter
from collections import defaultdict


class AdaptiveLearning(object):

    def __init__(self, data_path):
        self.user_list = dict()
        self.knowledge_nums = 25
        self.exercises_nums = 306
        self.exercises_Code2Id = dict()
        self.exercises_Id2Code = dict()
        self.knowledge_Code2Id = dict()
        self.knowledge_Id2Code = dict()
        self.chapter4_requirement = dict()
        self.users_exercise_info = dict()
        self.users_knowledge_info = dict()
        self.knowledge_exercise_id_range = dict()
        self.exercises_info = pd.read_excel(data_path + '初中物理8年级教研精选题306题.xlsx')

    def load_data(self, data_path):
        print("导入数据中......")
        chapter4_target = pd.read_excel(data_path + '初中物理8年级教研精选练习目标描述.xlsx')
        for i, target_code in enumerate(chapter4_target['target_code']):
            self.knowledge_Code2Id[target_code] = i
            self.knowledge_Id2Code[i] = target_code
        for i, target in enumerate(chapter4_target['建议配题数量']):
            self.chapter4_requirement[i] = target
        pre_code = 'WL08B04010101'
        id = 0
        for i, exercise_code in enumerate(self.exercises_info['exercise_code']):
            if pre_code != exercise_code[:-4]:
                self.knowledge_exercise_id_range[id] = i
                id += 1
                pre_code = exercise_code[:-4]
            self.exercises_Code2Id[exercise_code] = i
            self.exercises_Id2Code[i] = exercise_code
        self.knowledge_exercise_id_range[id] = self.exercises_nums

        with open(data_path + 'user_list.json', 'r', encoding='utf-8') as f:
            self.user_list = json.load(f)

        with open(data_path + 'users_exercise_info.json', 'r', encoding='utf-8') as f:
            self.users_exercise_info = json.load(f)

        with open(data_path + 'users_knowledge_info.json', 'r', encoding='utf-8') as f:
            self.users_knowledge_info = json.load(f)
        # for _ in tqdm(range(100)):
        #     time.sleep(0.01)
        print("数据导入完成\n************************")

    def login(self):
        username = input("用户名：")
        password = input("密码：")
        if username in self.user_list:
            if password == self.user_list[username]:
                print("登录成功")
                return username
        self.login()

    def sign_on(self):
        username = input("输入你需要注册的用户名：")
        password = input("输入密码：")
        if username not in self.user_list:
            self.user_list[username] = password
            self.users_exercise_info.setdefault(username, defaultdict(list))
            self.users_knowledge_info.setdefault(username, defaultdict(int))
        else:
            print("用户名已被占用")

    def fetch_exercise(self, exercise_id):
        table = self.exercises_info.loc[[exercise_id]]
        # values = to_numpy
        exercise_title = table[['title']].values[0][0]
        exercise_answer = table[['right_answer']].values[0][0]
        exercise_tip = table[['study_tips']].values[0][0]
        return exercise_title, exercise_answer, exercise_tip

    def preprocess(self, exercise_title):
        exercise = exercise_title.split('\n')

        pattern = re.compile(
            r'<p class="tb_exam_line">|</p>|<span class="tiankong" style="text-decoration:underline;">|</span>')
        blankspace = re.compile(r'\u3000')
        tag = re.compile('<span class="rx">')

        exercise_desc = []
        exercise_desc.append('')
        for i in range(len(exercise)):
            exercise[i] = re.sub(pattern, '', exercise[i])
            exercise[i] = re.sub(blankspace, '_', exercise[i])
            if re.match(tag, exercise[i]):
                exercise_desc.append(re.sub(tag, '', exercise[i]))
            else:
                exercise_desc[0] += '\n' + exercise[i]

        return exercise_desc

    def preprocess_tip(self,exercise_tip):
        tag = re.compile(r'<.*?>')
        if type(exercise_tip) != type(np.nan):
            exercise_tip = re.sub(tag,'',exercise_tip)
        else:
            exercise_tip = '略'
        return exercise_tip

    def show_and_judge_exercise(self, exerciseId, exercise_title, exercise_answer, exercise_tip):
        Exit = False
        True_or_False = True
        exercise_desc = self.preprocess(exercise_title)
        for line in exercise_desc:
            print(line)
        print(f"请作答第{exerciseId + 1}题：(输入'exit'停止作答)")
        userAnswer = input()
        for _ in tqdm(range(30), desc="正在判题"):
            time.sleep(0.01)
        if userAnswer == exercise_answer:
            print("回答正确")
        else:
            print("回答错误")
            print(f"正确答案{exercise_answer}")
            True_or_False = False
        if exercise_tip != np.nan:
            print(exercise_tip)
        print("-" * 30)
        return True_or_False

    def masterExercise(self, user, exercise_id):
        if self.users_exercise_info[user][exercise_id][0] > self.users_exercise_info[user][exercise_id][1]:
            return True
        else:
            return False

    def recommend_for_GUI(self):
        exercise_id = np.random.randint(3,self.exercises_nums)
        _, _, exercise_tip = self.fetch_exercise(exercise_id)
        exercise_tip = self.preprocess_tip(exercise_tip)
        while(exercise_tip == '略'):
            exercise_id = np.random.randint(3, self.exercises_nums)
            _, _, exercise_tip = self.fetch_exercise(exercise_id)
            exercise_tip = self.preprocess_tip(exercise_tip)
        return exercise_id

    def caculateSim(self):
        exercise_user = dict()
        user_sim_matrix = dict()
        for user, exercises in self.users_exercise_info.items():
            for exercise in exercises:
                if exercise not in exercise_user:
                    exercise_user[exercise] = set()
                exercise_user[exercise].add(user)

        for exercise, users in exercise_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    if self.masterExercise(u, exercise) == self.masterExercise(v, exercise):
                        user_sim_matrix.setdefault(u, {})
                        user_sim_matrix[u].setdefault(v, 0)
                        user_sim_matrix[u][v] += 1
        # 计算相似度
        for u, related_users in user_sim_matrix.items():
            for v, count in related_users.items():
                user_sim_matrix[u][v] = count / math.sqrt(len(self.users_exercise_info[u])) * math.sqrt(
                    len(self.users_exercise_info[v]))
        return user_sim_matrix

    def userCF(self, username):
        user_sim_matrix = self.caculateSim()
        K = 3
        N = 1
        rank = {}
        done_exercises = self.users_exercise_info[username]

        for v, wuv in sorted(user_sim_matrix[username].items(), key=itemgetter(1), reverse=True)[0:K]:
            for exercise_id in self.users_exercise_info[v]:
                if exercise_id in done_exercises:
                    continue
                rank.setdefault(exercise_id, 0)
                rank[exercise_id] += wuv
            if len(rank) >= N:
                return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:N], 0
            else:
                return rank, N - len(rank)

    def exerciseId2knowledgeId(self, exercise_id):
        for knowledge_id, num in self.knowledge_exercise_id_range.items():
            if exercise_id < num:
                return knowledge_id

    def updataTable(self, result, username, knowledge_id, exercise_id):
        if result:
            self.users_exercise_info[username].setdefault(exercise_id, [0, 0])
            self.users_exercise_info[username][exercise_id][0] += 1
        else:
            self.users_exercise_info[username].setdefault(exercise_id, [0, 0])
            self.users_exercise_info[username][exercise_id][1] += 1

    def save(self, data_path):
        with open(data_path + 'user_list.json', 'w', encoding='utf-8') as fw:
            json.dump(self.user_list, fw,ensure_ascii=False)
        with open(data_path + 'users_exercise_info.json', 'w', encoding='utf-8') as fw:
            json.dump(self.users_exercise_info, fw,ensure_ascii=False)
        with open(data_path + 'users_knowledge_info.json', 'w', encoding='utf-8') as fw:
            json.dump(self.users_knowledge_info, fw,ensure_ascii=False)


class User(object):

    def __init__(self, username, exercise_table, knowledge_table, chapter_requirement, knowledge_exercise_id_range):
        self.username = username
        self.knowledge_nums = 25
        self.exercises_nums = 306
        self.exercise_table = exercise_table
        self.knowledge_table = knowledge_table
        self.chapter_requirement = chapter_requirement
        self.knowledge_exercise_id_range = knowledge_exercise_id_range

    def exerciseId2knowledgeId(self, exercise_id):
        for knowledge_id, num in self.knowledge_exercise_id_range.items():
            if int(exercise_id) < num:
                return knowledge_id

    def updateKnowledgeTable(self):
        for exercise_id,desc in self.exercise_table.items():
            knowledge_id = self.exerciseId2knowledgeId(exercise_id)
            if self.masterExercise(exercise_id):
                self.knowledge_table.setdefault(exercise_id,0)
                self.knowledge_table.setdefault(knowledge_id,0)
                self.knowledge_table[knowledge_id]+=1


    def recommend(self):
        knowledge_id = self.judge_knowledge_id()
        if knowledge_id == self.knowledge_nums:
            return -1
        else:
            return self.recommend_random_exercise(knowledge_id)

    def masterExercise(self, exercise_id):
        self.exercise_table.setdefault(exercise_id, [0, 0])
        if self.exercise_table[exercise_id][0] > self.exercise_table[exercise_id][1]:
            return True
        else:
            return False

    def recommend_random_exercise(self, knwoledge_id):
        if knwoledge_id == 0:
            low = 0
        else:
            low = self.knowledge_exercise_id_range[knwoledge_id - 1]
        high = self.knowledge_exercise_id_range[knwoledge_id]
        while (True):
            exercise_id = np.random.randint(low, high)
            if not self.masterExercise(exercise_id):
                return exercise_id

    def judge_knowledge_id(self):
        for knowledge_id, requirement_nums in self.chapter_requirement.items():
            if (knowledge_id not in self.knowledge_table) or (self.knowledge_table[knowledge_id] < requirement_nums):
                return knowledge_id
        return self.knowledge_nums

    def updataTable(self, result, knwoledge_id, exercise_id):
        if result:
            self.exercise_table.setdefault(exercise_id,[0,0])
            self.exercise_table[exercise_id][0] += 1
        else:
            self.exercise_table.setdefault(exercise_id, [0, 0])
            self.exercise_table[exercise_id][1] += 1


if __name__ == '__main__':
    data_path = os.getcwd() + '\\data\\'
    system = AdaptiveLearning(data_path)
    system.load_data(data_path)
    need_sign_on = input("有账号的用户输入任意键跳过，没有账号的用户输入‘注册’：")
    if need_sign_on == '注册':
        system.sign_on()

    username = system.login()
    system.users_knowledge_info.setdefault(username,defaultdict())
    user = User(username, system.users_exercise_info[username], system.users_knowledge_info[username], \
                system.chapter4_requirement, system.knowledge_exercise_id_range)

    # while(True)
    exercise_id = user.recommend()
    if exercise_id != -1 and False:
        exercise_title, exercise_answer, exercise_tip = system.fetch_exercise(exercise_id)
        result = system.show_and_judge_exercise(exercise_id, exercise_title, exercise_answer, exercise_tip)
        knowledge_id = system.exerciseId2knowledgeId(exercise_id)
        # system.updataTable(result,username,knowledge_id,exercise_id)
        user.updataTable(result, knowledge_id, exercise_id)
        user.updateKnowledgeTable()
    else:
        exercises_list, remains = system.userCF(username)
        for exercise in exercises_list:
            exercise_id = int(exercise[0])
            exercise_title, exercise_answer, exercise_tip = system.fetch_exercise(exercise_id)
            result = system.show_and_judge_exercise(exercise_id, exercise_title, exercise_answer, exercise_tip)
            knowledge_id = system.exerciseId2knowledgeId(exercise_id)
            # system.updataTable(result, username, knowledge_id, exercise_id)
            user.updataTable(result, knowledge_id, exercise_id)
            user.updateKnowledgeTable()
        while (remains):
            exercise_id = user.recommend()
            if exercise_id not in exercises_list:
                remains -= 1
                exercise_title, exercise_answer, exercise_tip = system.fetch_exercise(exercise_id)
                result = system.show_and_judge_exercise(exercise_id, exercise_title, exercise_answer, exercise_tip)
                knowledge_id = system.exerciseId2knowledgeId(exercise_id)
                # system.updataTable(result, username, knowledge_id, exercise_id)
                user.updataTable(result, knowledge_id, exercise_id)
                user.updateKnowledgeTable()

    system.save(data_path)
