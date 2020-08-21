import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as msg
from recommend import User, AdaptiveLearning
import os
import json


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
        self.Login_image = tk.PhotoImage(file="." + os.sep + "img" + os.sep + "logingui.png")
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
        self.title("答题界面")
        self.homepage = tk.Label(bg='white', text='学生自测界面', font=("微软雅黑", 20)).pack(fill=tk.X, pady=20)
        self["bg"] = "white"
        self.resizable(width=True, height=True)

        self.width = 600
        self.height = 900
        self.screenwidth = self.winfo_screenwidth()
        self.screenheight = self.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (
        self.width, self.height, (self.screenwidth - self.width) / 2, (self.screenheight - self.height) / 2)
        self.geometry(alignstr)

        self.initGUI = True
        self.initJudge = True
        self.content =''
        self.choices =[]
        self.run()
        self.mainloop()

    def run(self):

        exercises, remains = system.userCF(self.username)

        if remains == 0:
            exercise_id = system.recommend_for_GUI()
        else:
            exercise_id = int(exercises[0][0])

        exercise_title, self.exercise_answer, self.exercise_tip = system.fetch_exercise(exercise_id)
        exercise_desc = system.preprocess(exercise_title)
        self.exercise_tip = system.preprocess_tip(self.exercise_tip)

        self.content = exercise_desc[0]
        self.choices = [line for line in exercise_desc[1:]]
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
        tk.Label(self.question_frame, bg='white', text='题目:', font=("微软雅黑", 16), padx=10, pady=10).pack(side=tk.LEFT, pady=5)
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
        tk.Button(self.button_frame, text='提交', bg='green', width=20, font=("微软雅黑", 12), command= lambda :self.run()).pack(
            side=tk.LEFT,padx=10, pady=10)
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
            msg.showinfo('提示', '回答正确')
        else:
            msg.showinfo('提示', '回答错误')
        self.explaination_frame = tk.Frame(self, bg='white')
        self.explaination_frame.pack(fill=tk.X)
        tk.Label(self.explaination_frame, bg='white', text='', font=("微软雅黑", 16), padx=10, pady=10).pack(pady=5)
        tk.Label(self.explaination_frame, bg='yellow', text='正确答案：'+self.exercise_answer, font=("微软雅黑", 16), padx=10, pady=10).pack(pady=5)
        tk.Label(self.explaination_frame, bg='grey', text='解析：'+self.exercise_tip, font=("微软雅黑", 16), padx=10,
                 pady=10).pack(pady=5)

if __name__ == '__main__':
    import git
    login_name = ''
    data_path = os.getcwd() + '\\data\\'
    system = AdaptiveLearning(data_path)
    system.load_data(data_path)

    this_login = LoginWindow()
    this_login.mainloop()
    user = User(login_name, system.users_exercise_info[login_name], system.users_knowledge_info[login_name], \
                system.chapter4_requirement, system.knowledge_exercise_id_range)
    mainWindow = MainWindow()
