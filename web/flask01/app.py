from flask import Flask,request
import settings


app = Flask(__name__)
app.config.from_object(settings)

@app.route('/') # 路由
def hello_world(): # 视图函数
    return 'Hello World!'

@app.route('/greet', defaults={'name': 'Programmer'})
@app.route('/greet/<name>')
def greet(name):
    return '<h1>Hello, %s!</h1>' % name

if __name__ == '__main__':
    app.run(port = 5001,debug=True)
