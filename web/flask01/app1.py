from flask import Flask
import settings
app = Flask(__name__)

app.config.from_object(settings)
print(app.config)

# 路由
@app.route('/')
# 视图函数
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(port=8080)