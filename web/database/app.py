import os
import sys

import click
from flask import Flask
from flask import redirect, url_for, abort, render_template, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired
from flask_migrate import Migrate

# WIN = sys.platform.startswith('win')
# if WIN:
#     prefix = 'sqlite:///'
# else:
#     prefix = 'sqlite:////'

app = Flask(__name__)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret string')

app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql+pymysql://root:root@localhost/mydb')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

#实例化migrate类
migrate = Migrate(app, db)

# 当你使用flask shell命令启动Python Shell时，所有使用
# app.shell_context_processor装饰器注册的shell上下文处理函数都会被自
# 动执行，这会将db和Note对象推送到Python Shell上下文里
@app.shell_context_processor
def make_shell_context():
    return dict(db=db,Note=Note,Author=Author,Article=Article,Student=Student,Teacher=Teacher)


@app.cli.command()
@click.option('--drop', is_flag=True, help='Create after drop.')
def initdb(drop):
    """Initialize the database."""
    if drop:
        db.drop_all()
    db.create_all()
    click.echo('Initialized database.')

# Forms
class NewNoteForm(FlaskForm):
    body = TextAreaField('Body', validators=[DataRequired()])
    submit = SubmitField('Save')


class EditNoteForm(FlaskForm):
    body = TextAreaField('Body', validators=[DataRequired()])
    submit = SubmitField('Update')


class DeleteNoteForm(FlaskForm):
    submit = SubmitField('Delete')


class Note(db.Model):

    # 表名称要对应
    __tablename__ = 'notes'
    id = db.Column(db.Integer, primary_key=True)
    body = db.Column(db.Text)

    # optional
    def __repr__(self):
        return '<Note %r>' % self.body

class Author(db.Model):
    __tablename__ = 'author'
    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(70),unique=True)
    phone = db.Column(db.String(20))
    # 与Article类建立关系
    # db.relationship('Article',uselist=False) 建立一对一关系
    articles = db.relationship('Article')


class Article(db.Model):
    __tablename__ = 'article'
    id = db.Column(db.Integer,primary_key=True)
    title = db.Column(db.String(70),unique=True)
    body = db.Column(db.Text)
    #建立多对一关系（外键）
    author_id = db.Column(db.Integer,db.ForeignKey('author.id'))

# 通过关联表建立多对多关系，关联表由SQLAlchemy接管，所以不需要额外操作
association_table = db.Table('association',
                             db.Column('student_id', db.Integer, db.ForeignKey('student.id')),
                             db.Column('teacher_id', db.Integer, db.ForeignKey('teacher.id'))
                             )

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(70), unique=True)
    grade = db.Column(db.String(20))
    teachers = db.relationship('Teacher',
                               secondary=association_table,
                               back_populates='students')  # collection
    def __repr__(self):
        return '<Student %r>' % self.name

class Teacher(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(70), unique=True)
    office = db.Column(db.String(20))
    students = db.relationship('Student',
                               secondary=association_table,
                               back_populates='teachers')  # collection
    def __repr__(self):
        return '<Teacher %r>' % self.name


# 数据库添加数据
# note1 = Note(body='remember Sammy Jankis')
# note2 = Note(body='SHAVE')
# note3 = Note(body='DON\'T BELIEVE HIS LIES, HE IS THE ONE, KILL HIM')
# db.session.add(note1)
# db.session.add(note2)
# db.session.add(note3)
# db.session.commit()

# foo = Author(name='Foo')
# spam = Article(title='Spam')
# ham = Article(title='Ham')
# db.session.add(foo)
# db.session.add(spam)
# db.session.add(ham)
# spam.author_id = 1
# db.session.commit()

@app.route('/')
def index():
    form = DeleteNoteForm()
    notes = Note.query.all()
    return render_template('index.html',notes = notes,form = form)

@app.route('/edit_note/<int:note_id>', methods=['GET','POST'])
def edit_note(note_id):
    form = EditNoteForm()
    note = Note.query.get(note_id)
    if form.validate_on_submit():
        note.body = form.body.data
        db.session.commit()
        flash('Your note is updated.')
        return redirect(url_for('index'))
    # 保留原有笔记内容
    form.body.data = note.body
    return render_template('edit_note.html', form=form)

@app.route('/new', methods=['GET', 'POST'])
def new_note():
    form = NewNoteForm()
    if form.validate_on_submit():
        body = form.body.data
        note = Note(body=body)
        db.session.add(note)
        db.session.commit()
        flash('Your note is saved.')
        return redirect(url_for('index'))
    return render_template('new_note.html', form=form)

@app.route('/delete/<int:note_id>', methods=['POST'])
def delete_note(note_id):
    form = DeleteNoteForm()
    if form.validate_on_submit():
        note = Note.query.get(note_id)
        db.session.delete(note)
        db.session.commit()
        flash('Your note is deleted.')
    else:
        abort(400)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run()