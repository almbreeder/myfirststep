import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as msg
from recommend import User, AdaptiveLearning
import os
import json
import matplotlib.pyplot as plt

class LoginWindow(tk.Tk):
    """
    创建登录窗体的GUI界面已经登录的方法
    """

    def __init__(self):
        super().__init__()  # 先执行tk这个类的初始化
        self.title("登录界面")
        # self.geometry("620x420")
        self.resizable(0, 0)  # 窗体大小不允许变，两个参数分别代表x轴和y轴
        # self.iconbitmap("."+os.sep+"img"+os.sep+"favicon.ico")
        # self["bg"] = "royalblue"
        # 加载窗体
        self.setup_UI()


    def setup_UI(self):
        # ttk中控件使用style对象设定
        self.Style01 = ttk.Style()
        self.Style01.configure("user.TLabel", font=("华文黑体", 20, "bold"), foreground="royalblue")
        self.Style01.configure("TEntry", font=("华文黑体", 20, "bold"))
        self.Style01.configure("TButton", font=("华文黑体", 20, "bold"), foreground="royalblue")
        # 创建一个Label标签展示图片
        self.Login_image = tk.PhotoImage(file="." + os.sep + "img" + os.sep + "开始答题.png")
        self.Label_image = tk.Label(self, image=self.Login_image)
        self.Label_image.pack(padx=10, pady=10)
        # 创建一个Label标签 + Entry   --- 用户名
        self.Label_user = ttk.Label(self, text="用户名:", style="user.TLabel")
        self.Label_user.pack(side=tk.LEFT, padx=10, pady=10)
        self.Entry_user = tk.Entry(self, width=12)
        self.Entry_user.pack(side=tk.LEFT, padx=10, pady=10)
        # 创建一个Label标签 + Entry   --- 密码
        self.Label_password = ttk.Label(self, text="密码:", style="user.TLabel")
        self.Label_password.pack(side=tk.LEFT, padx=10, pady=10)
        self.Entry_password = tk.Entry(self, width=12, show="*")
        self.Entry_password.pack(side=tk.LEFT, padx=10, pady=10)
        # 创建一个按钮    --- 登录
        self.Button_login = ttk.Button(self, text="登录", width=4,command= lambda :self.login())
        self.Button_login.pack(side=tk.LEFT, padx=20, pady=10)

    def login(self):
        global login_name
        login_name = self.Entry_user.get()
        password = self.Entry_password.get()
        with open('./data/user_list.json','r',encoding='utf-8') as f:
            userlist = json.load(f)
        if login_name in userlist:
            if password == userlist[login_name]:
                msg.showinfo('提示','登录成功')
                # time.sleep(1)
                self.load_main()
            else:
                msg.showinfo('提示', '用户名或密码有误！')
        else:
            msg.showinfo('提示', '用户名或密码有误！')

    def load_main(self):
        # 关闭当前窗体
        self.destroy()

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.username = login_name

        self.exercise_num = system.N

        self.title("答题界面")
        self.homepage = tk.Label(bg='white', text='学生自测界面', font=("微软雅黑", 20)).pack(fill=tk.X, pady=20)
        self["bg"] = "white"
        self.resizable(width=True, height=True)

        self.show_status_image = tk.PhotoImage(file="." + os.sep + "img" + os.sep + "开始答题.png")
        # self.show_status_image = tk.Label(self, image=self.show_status_image)

        self.right_and_wrong = [0,0]

        self.width = 1920
        self.height = 1080
        self.screenwidth = self.winfo_screenwidth()
        self.screenheight = self.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (
        self.width, self.height, (self.screenwidth - self.width) / 2, (self.screenheight - self.height) / 2)
        self.geometry(alignstr)

        self.initGUI = True
        self.initJudge = True
        self.startGUI = True
        self.startCv = True
        self.content =''
        self.choices =[]
        # 题目推荐列表，如果为1则为协同算法推荐
        self.exercises_id_list = dict()
        self.initUI()
        # self.run()
        self.mainloop()

    def initUI(self):
        if not self.startCv:
            self.piepicture()
        self.startCv = False
        self.right_and_wrong = [0,0]
        self.startGUI = True
        if not self.initGUI:
            self.question_frame.destroy()
            self.button_frame.destroy()
            self.Radio_frame.destroy()
            self.explaination_frame.destroy()
            self.initGUI = False

        self.frame_info = tk.Frame(self,bg = 'white')
        self.frame_info.pack(fill=tk.X)
        tk.Label(self.frame_info, bg='white', text='姓名:  '+self.username, font=("微软雅黑", 16), padx=10, pady=10).pack()

        self.frame_cv = tk.Frame(self,bg = 'white')
        self.frame_cv.pack(fill=tk.X)
        canvas_width = 750
        canvas_height = 500
        self.cv = tk.Canvas(self.frame_cv,bg = 'white',width=canvas_width,height = canvas_height)
        img =  self.show_status_image
        self.cv.create_image(canvas_width/2,canvas_height/2,image = img)
        self.cv.pack()

        self.frame_xuanze = tk.Frame(self,bg = 'white')
        self.frame_xuanze.pack(pady=50)
        #single_button =  tk.Button(frame,background = 'white',text = '单个题',image = self.Login_image).pack(padx=10, pady=10)
        single_button = tk.Button(self.frame_xuanze, text='单题', width=50, font=("华文楷体", 20),command = lambda :self.run(True)).pack(padx=50, pady=10)
        multi_button =  tk.Button(self.frame_xuanze, text = '套题',width = 50, font=("华文楷体", 20),command = lambda : self.run()).pack(padx=50, pady=10)


    def run(self,tag = False):
        exercises_id_list = []
        if tag:
            self.exercise_num = 1
            exercise_id = system.recommend_for_GUI()
            self.exercises_id_list.setdefault(exercise_id, 0)
            exercises_id_list.append(exercise_id)
        else:
            exercises, remains = system.userCF(self.username)
            for id in exercises:
                #去除非选择题
                exercise_id = system.judge_choices_for_GUI(id[0])
                if exercise_id == id[0]:
                    self.exercises_id_list.setdefault(exercise_id,1)
                    self.exercises_id_list[exercise_id] =1
                    exercises_id_list.append(exercise_id)
                else:
                    self.exercises_id_list.setdefault(exercise_id, 0)
                    self.exercises_id_list[exercise_id] = 0
                    exercises_id_list.append(exercise_id)

        while len(exercises_id_list) < system.N:
            exercise_id = system.recommend_for_GUI()
            self.exercises_id_list.setdefault(exercise_id,0)
            exercises_id_list.append(exercise_id)

        self.current_exercise_id = exercises_id_list[self.exercise_num-1]
        exercise_title, self.exercise_answer, self.exercise_tip = system.fetch_exercise(exercises_id_list[self.exercise_num-1])
        exercise_desc = system.preprocess(exercise_title)
        self.exercise_tip = system.preprocess_tip(self.exercise_tip)

        self.content = exercise_desc[0].strip()
        self.choices = [line for line in exercise_desc[1:]]

        if self.startGUI:
            self.frame_info.destroy()
            self.frame_cv.destroy()
            self.frame_xuanze.destroy()
            self.startGUI = False

        if not self.initGUI:
            self.question_frame.destroy()
            self.button_frame.destroy()
            self.Radio_frame.destroy()
            self.explaination_frame.destroy()
        self.initGUI = False
        self.setUI()


    def setUI(self):
        self.question_frame = tk.Frame(self, bg='white')
        self.question_frame.pack(fill=tk.X)
        # tk.Label(self.question_frame, bg='white', text='题目:', font=("微软雅黑", 16), padx=10, pady=10).pack(side=tk.LEFT, pady=5)
        questionLabel = tk.Label(self.question_frame, bg='white', text=self.content, wraplength=400, justify='left',
                                 font=("微软雅黑", 16))
        questionLabel.pack(fill=tk.X, pady=30)

        self.Radio_frame = tk.Frame(self, bg='white')
        self.Radio_frame.pack(fill=tk.X)

        self.vanswer = tk.IntVar()
        self.vanswer.set(0)
        for i in range(len(self.choices)):
            tk.Radiobutton(self.Radio_frame, text=self.choices[i], font=("微软雅黑", 16), bg='white', variable =self.vanswer,value=i,
                           indicatoron=1,command = lambda :self.judge()).pack(padx=10, pady=10)

        self.button_frame = tk.Frame(self, bg='white')
        self.button_frame.pack( )
        print(self.exercise_num)
        if self.exercise_num == 1:
            self.exercise_num = system.N
            tk.Button(self.button_frame, text='提交', bg='green', width=20, font=("微软雅黑", 12), command= lambda :self.initUI()).pack(
                side=tk.LEFT,padx=10, pady=10)
        else:
            self.exercise_num -= 1
            tk.Button(self.button_frame, text='下一题', bg='royalblue', width=20, font=("微软雅黑", 12), command= lambda :self.run()).pack(
                side=tk.RIGHT,padx=10, pady=10)

        self.explaination_frame = tk.Frame(self, bg='white')
        self.explaination_frame.pack(fill=tk.X)
        tk.Label(self.explaination_frame, bg='white', text='', font=("微软雅黑", 16), padx=10, pady=10).pack(pady=5)
        tk.Label(self.explaination_frame, bg='white', text='', font=("微软雅黑", 16), padx=10,
                 pady=10).pack(pady=5)

    def submitAnswer(self):
        msg.showinfo('提示', '提交完成')

    def judge(self):
        self.explaination_frame.destroy()
        print(self.vanswer.get())
        num2alpha = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E'}
        answer = num2alpha[str(self.vanswer.get())]
        if answer == self.exercise_answer:
            self.right_and_wrong[0] += 1
            msg.showinfo('提示', '回答正确')
        else:
            self.right_and_wrong[1] += 1
            msg.showinfo('提示', '回答错误')

        self.explaination_frame = tk.Frame(self, bg='white')
        self.explaination_frame.pack(fill=tk.X)
        tk.Label(self.explaination_frame, bg='white', text='', font=("微软雅黑", 16), padx=10, pady=10).pack(pady=5)
        tk.Label(self.explaination_frame, bg='yellow', text='正确答案：'+self.exercise_answer, font=("微软雅黑", 16), padx=10, pady=10).pack(pady=5)
        tk.Label(self.explaination_frame, bg='grey', text='解析：'+self.exercise_tip, font=("微软雅黑", 16), padx=10,
                 pady=10).pack(pady=5)
        if self.exercises_id_list[self.current_exercise_id] == 0:
            tk.Label(self.explaination_frame, bg='grey', text='根据薄弱知识点推荐', font=("微软雅黑", 16), padx=10, pady=10).pack(pady=5)
        else:
            tk.Label(self.explaination_frame, bg='grey', text='根据协同过滤算法推荐', font=("微软雅黑", 16), padx=10, pady=10).pack(
                pady=5)

    def piepicture(self):
        labels = 'right', 'wrong'
        sizes = self.right_and_wrong
        colors = 'lightgreen', 'lightcoral'
        explode = 0, 0
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=50)
        plt.axis('equal')
        plt.savefig('./img/status.png')
        self.show_status_image = tk.PhotoImage(file="." + os.sep + "img" + os.sep + "status.png")
        plt.close()


if __name__ == '__main__':
    login_name = '朱文龙'
    data_path = os.getcwd() + '\\data\\'
    system = AdaptiveLearning(data_path)
    system.load_data(data_path)

    this_login = LoginWindow()
    this_login.mainloop()
    user = User(login_name, system.users_exercise_info[login_name], system.users_knowledge_info[login_name], \
                system.chapter4_requirement, system.knowledge_exercise_id_range)

    mainWindow = MainWindow()
