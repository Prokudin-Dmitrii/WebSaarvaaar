import os
import base64
import pickle
import json
import pandas as pd
import numpy as np
import ast

from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from flask import Flask, request, url_for, render_template_string, flash
from flask import render_template, redirect
from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired, NumberRange, Optional, InputRequired, ValidationError
from wtforms import StringField, SubmitField, FileField, IntegerField, DecimalField
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='./html')
app.config['BOOTSTRAP_SERVE_LOCAL'] = True
app.config['SECRET_KEY'] = 'hello'
Bootstrap(app)

messages = []

rf_model = pickle.load(open(os.path.join('./artifacts/', "RF.pkl"), "rb"))
gb_model = pickle.load(open(os.path.join('./artifacts/', "GB.pkl"), "rb"))
rf_train_dataset_path = './kc_house_data.csv'
rf_val_dataset_path = './kc_house_data.csv'
gb_train_dataset_path = './kc_house_data.csv'
gb_val_dataset_path = './kc_house_data.csv'

class Message:
    header = ''
    text = ''

class FileForm(FlaskForm):
    file_path = FileField('Path', validators=[
        DataRequired('Specify file'),
        FileAllowed(['csv'], 'CSV only!')
    ])
    submit = SubmitField('Open File')

class BadInputForm(FlaskForm):
    text = StringField('При обучении произошла ошибка, состояние модели возвращено к изначальному')
    submit = SubmitField('Вернуться на главную страницу')

class BadInputForm_2(FlaskForm):
    text = StringField('Некорректный датасет или имя целевой переменной')
    submit = SubmitField('Вернуться на главную страницу')

class GoodInputForm(FlaskForm):
    text = StringField('Модель завершила обучение')
    submit = SubmitField('Вернуться на главную страницу')

class Mmenu(FlaskForm):
    create = SubmitField('Создание и обучение модели')
    inference = SubmitField('Предсказания обученной модели')
    params = SubmitField('Параметры модели')

class FastTravel(FlaskForm):
    travel = SubmitField('Перейти на главную страницу')

data_path = './'

@app.route('/', methods=['GET', 'POST'])
def zero():
    sub = FastTravel()
    if sub.validate_on_submit():
        return redirect(url_for('mainer'))
    return render_template('trybg.html', form=sub)

@app.route('/bad_input', methods=['GET', 'POST'])
def bad():
    sub = BadInputForm()
    string = request.args.get('text')
    sub.text.data = string
    if sub.validate_on_submit():
        return redirect(url_for('mainer'))
    return render_template('from_form_5.html', form=sub)

@app.route('/bad_data', methods=['GET', 'POST'])
def bad_data():
    sub = BadInputForm_2()
    string = request.args.get('text')
    sub.text.data = string
    if sub.validate_on_submit():
        return redirect(url_for('mainer'))
    return render_template('from_form_5.html', form=sub)

@app.route('/good_input', methods=['GET', 'POST'])
def good():
    sub = GoodInputForm()
    string = request.args.get('text')
    sub.text.data = string
    if sub.validate_on_submit():
        return redirect(url_for('mainer'))
    return render_template('from_form_8.html', form=sub)

@app.route('/home', methods=['GET', 'POST'])
def mainer():
    try:
        submitter = Mmenu()
        
        if submitter.validate_on_submit():
            if submitter.create.data:
                return redirect(url_for('chooser'))
            elif submitter.inference.data:
                return redirect(url_for('predictions'))
            elif submitter.params.data:
                return redirect(url_for('params'))
        
        return render_template('from_form_2.html', form=submitter)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

class Cmenu(FlaskForm):
    rf = SubmitField('Random Forest (MSE)')
    gb = SubmitField('Gradient Boosting (MSE)')
    returner = SubmitField('Вернуться на главную страницу')

def validate_name(form, field):
    try:
        json_acceptable_string = str(field.data).replace("'", "\"")
        diction = json.loads(json_acceptable_string)
        if not isinstance(diction, dict):
            raise ValidationError('Ввод не является корректным словарём')
    except Exception as exc:
        raise ValidationError('Ввод не является корректным словарём')

@app.route('/choose_model', methods=['GET', 'POST'])
def chooser():
    try:
        submitter = Cmenu()
        
        if submitter.validate_on_submit():
            if submitter.rf.data:
                return redirect(url_for('rf_train'))
            elif submitter.gb.data:
                return redirect(url_for('gb_train'))
            elif submitter.returner.data:
                return redirect(url_for('mainer'))

        return render_template('from_form_3.html', form=submitter)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

class RFTrainForm(FlaskForm):
    n_estimators = IntegerField('Количество деревьев в лесе', validators=[InputRequired(), NumberRange(min=1, max=None, message='Количество деревьев должно быть натуральным числом')], default=100)
    max_depth = IntegerField('Максимальная глубина', validators=[InputRequired(),  NumberRange(min=0, max=None, message='Глубина должна быть натуральным числом')], default=0)
    fss = IntegerField('Размер признакового пространства для деревьев', validators=[InputRequired(),  NumberRange(min=0, max=None, message='Количество признаков должно быть натуральным числом')], default=0)
    r_state_seed = IntegerField('Random_state seed', validators=[InputRequired(), NumberRange(min=0, max=None, message='Seed - целое неотрицательное число')], default=0)

    param_dict = StringField('Словарь с остальными параметрами для деревьев', validators=[InputRequired(), validate_name], default={})
    
    train_data = FileField('Датасет для обучения', validators=[
        DataRequired('Укажите файл'),
        FileAllowed(['csv'], 'Формат файла должен быть CSV')
    ])

    target = StringField('Название столбца с целевой переменной', validators=[InputRequired()])

    val_data = FileField('Датасет для валидации', validators=[
        Optional('Укажите файл'),
        FileAllowed(['csv'], 'Формат файла должен быть CSV')
    ])

    submit = SubmitField('Обучить модель')

@app.route('/RF_train', methods=['GET', 'POST'])
def rf_train():
    global rf_model, rf_train_dataset_path, rf_val_dataset_path
    try:
        train_form = RFTrainForm()

        if train_form.validate_on_submit():
            n_estimators = train_form.n_estimators.data
            max_depth = train_form.max_depth.data
            fss = train_form.fss.data
            rs = train_form.r_state_seed.data
            json_acceptable_string = str(train_form.param_dict.data).replace("'", "\"")
            params = json.loads(json_acceptable_string)

            if max_depth == 0:
                max_depth = None
            if rs == 0:
                rs = None
            if fss == 0:
                fss = None

            target = train_form.target.data
            train_filename = secure_filename(train_form.train_data.data.filename)
            try:
                val_filename = train_form.val_data.data
                val_filename = secure_filename(train_form.val_data.data.filename)
            except:
                val_filename = None
            if val_filename == '':
                val_filename = None
            try:
                X = pd.read_csv(train_filename)
                y = np.array(X[target])
                X = X.drop([target], axis=1)
                X = np.array(X)

                X_val = None
                y_val = None

                if not(val_filename is None):
                    X_val = pd.read_csv(val_filename)
                    y_val = np.array(X_val[target])
                    X_val = X_val.drop([target], axis=1)
                    X_val = np.array(X_val)
            except Exception as ans:
                text = str(ans)
                return redirect(url_for('bad_data', text=text))

            try:
                rf_train_dataset_path = train_filename
                rf_val_dataset_path = val_filename
                rf_model.n = n_estimators
                rf_model.r_state = rs
                rf_model.fss = fss
                rf_model.max_depth = max_depth
                rf_model.trees_params = params
                
                rf_model.fit(X, y, X_val, y_val)

            except Exception as ans:
                rf_model = pickle.load(open(os.path.join('./', "RF.pkl"), "rb"))
                rf_train_dataset_path = './kc_house_data.csv'
                rf_val_dataset_path = './kc_house_data.csv'
                text = str(ans)
                return redirect(url_for('bad', text=text))
            return redirect(url_for('good', text='Успех'))
        return render_template('from_form_6.html', form=train_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

class GBTrainForm(FlaskForm):
    n_estimators = IntegerField('Количество деревьев', validators=[InputRequired(), NumberRange(min=1, max=None, message='Количество деревьев должно быть натуральным числом')], default=100)
    max_depth = IntegerField('Максимальная глубина', validators=[InputRequired(),  NumberRange(min=0, max=None, message='Глубина должна быть натуральным числом')], default=0)
    fss = IntegerField('Размер признакового пространства для деревьев', validators=[InputRequired(),  NumberRange(min=0, max=None, message='Количество признаков должно быть натуральным числом')], default=0)
    r_state_seed = IntegerField('Random_state seed', validators=[InputRequired(), NumberRange(min=0, max=None, message='Seed - целое неотрицательное число')], default=0)
    lr = DecimalField('Learning rate', validators=[InputRequired(), NumberRange(min=0, max=None, message='')], default=0.1)

    param_dict = StringField('Словарь с остальными параметрами для деревьев', validators=[InputRequired(), validate_name], default={})
    
    train_data = FileField('Датасет для обучения', validators=[
        DataRequired('Укажите файл'),
        FileAllowed(['csv'], 'Формат файла должен быть CSV')
    ])

    target = StringField('Название столбца с целевой переменной', validators=[InputRequired()])

    val_data = FileField('Датасет для валидации', validators=[
        Optional('Укажите файл'),
        FileAllowed(['csv'], 'Формат файла должен быть CSV')
    ])

    submit = SubmitField('Обучить модель')

@app.route('/GB_train', methods=['GET', 'POST'])
def gb_train():
    global gb_model, gb_train_dataset_path, gb_val_dataset_path
    try:
        train_form = GBTrainForm()

        if train_form.validate_on_submit():
            n_estimators = train_form.n_estimators.data
            max_depth = train_form.max_depth.data
            fss = train_form.fss.data
            rs = train_form.r_state_seed.data
            lr = float(train_form.lr.data)
            json_acceptable_string = str(train_form.param_dict.data).replace("'", "\"")
            params = json.loads(json_acceptable_string)

            if max_depth == 0:
                max_depth = 5
            if rs == 0:
                rs = None
            if fss == 0:
                fss = None

            target = train_form.target.data
            train_filename = secure_filename(train_form.train_data.data.filename)
            try:
                val_filename = train_form.val_data.data
                val_filename = secure_filename(train_form.val_data.data.filename)
            except:
                val_filename = None
            if val_filename == '':
                val_filename = None
            try:
                X = pd.read_csv(train_filename)
                y = np.array(X[target])
                X = X.drop([target], axis=1)
                X = np.array(X)

                X_val = None
                y_val = None

                if not(val_filename is None):
                    X_val = pd.read_csv(val_filename)
                    y_val = np.array(X_val[target])
                    X_val = X_val.drop([target], axis=1)
                    X_val = np.array(X_val)
            except Exception as ans:
                text = str(ans)
                return redirect(url_for('bad_data', text=text))

            try:
                gb_train_dataset_path = train_filename
                gb_val_dataset_path = val_filename
                gb_model.n = n_estimators
                gb_model.r_state = rs
                gb_model.fss = fss
                gb_model.max_depth = max_depth
                gb_model.lr = lr
                gb_model.trees_params = params
                
                gb_model.fit(X, y, X_val, y_val)

            except Exception as ans:
                gb_model = pickle.load(open(os.path.join('./', "GB.pkl"), "rb"))
                gb_train_dataset_path = './kc_house_data.csv'
                gb_val_dataset_path = './kc_house_data.csv'
                text = str(ans)
                return redirect(url_for('bad', text=text))
            return redirect(url_for('good', text='Успех'))
        return render_template('from_form_6.html', form=train_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

class PRmenu(FlaskForm):
    rf = SubmitField('Предсказание Random Forest')
    gb = SubmitField('Предсказание Gradient Boosting')
    returner = SubmitField('Вернуться на главную страницу')

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    try:
        submitter = PRmenu()
        
        if submitter.validate_on_submit():
            if submitter.rf.data:
                return redirect(url_for('rf_predictions'))
            elif submitter.gb.data:
                return redirect(url_for('gb_predictions'))
            elif submitter.returner.data:
                return redirect(url_for('mainer'))

        return render_template('from_form_9.html', form=submitter)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

class Inference(FlaskForm):
    inference_data = FileField('Датасет для предсказания', validators=[
        DataRequired('Укажите файл'),
        FileAllowed(['csv'], 'Формат файла должен быть CSV')
    ])

    submit = SubmitField('Получить предсказания')

class Result(FlaskForm):
    results = StringField('Результат', validators=[Optional()])
    home = SubmitField('Вернуться на главную страницу')
    try_again = SubmitField('Получить предсказания ещё раз')
    

@app.route('/predictions/RF', methods=['GET', 'POST'])
def rf_predictions():
    try:
        submitter = Inference()

        if submitter.validate_on_submit():
            filename = secure_filename(submitter.inference_data.data.filename)
            return redirect(url_for('rf_result', filename=filename))

        return render_template('from_form_10.html', form=submitter)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    
@app.route('/predictions/RF/result', methods=['GET', 'POST'])
def rf_result():
    global rf_model
    try:
        submitter = Result()
        datafile = request.args.get('filename')
        try:
            data = pd.read_csv(datafile)
            data = np.array(data)
            prediction = list(rf_model.predict(data))
            prediction = json.dumps(prediction)
        except Exception as ans:
            text = str(ans)
            return redirect(url_for('bad_data', text=text))

        submitter.results.data = prediction
        if submitter.validate_on_submit():
            if submitter.home.data:
                return redirect(url_for('mainer'))
            else:
                return redirect(url_for('rf_predictions'))
        return render_template('from_form_11.html', form=submitter)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/predictions/GB', methods=['GET', 'POST'])
def gb_predictions():
    try:
        submitter = Inference()

        if submitter.validate_on_submit():
            filename = secure_filename(submitter.inference_data.data.filename)
            return redirect(url_for('gb_result', filename=filename))

        return render_template('from_form_10.html', form=submitter)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
    
@app.route('/predictions/GB/result', methods=['GET', 'POST'])
def gb_result():
    global gb_model
    try:
        submitter = Result()
        datafile = request.args.get('filename')
        try:
            data = pd.read_csv(datafile)
            data = np.array(data)
            prediction = list(gb_model.predict(data))
            prediction = json.dumps(prediction)
        except Exception as ans:
            text = str(ans)
            return redirect(url_for('bad_data', text=text))

        submitter.results.data = prediction
        if submitter.validate_on_submit():
            if submitter.home.data:
                return redirect(url_for('mainer'))
            else:
                return redirect(url_for('gb_predictions'))
        return render_template('from_form_11.html', form=submitter)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))


class Pmenu(FlaskForm):
    rf = SubmitField('Параметры Random Forest')
    gb = SubmitField('Параметры Gradient Boosting')
    returner = SubmitField('Вернуться на главную страницу')

@app.route('/parameters', methods=['GET', 'POST'])
def params():
    try:
        submitter = Pmenu()
        
        if submitter.validate_on_submit():
            if submitter.rf.data:
                return redirect(url_for('rf_params'))
            elif submitter.gb.data:
                return redirect(url_for('gb_params'))
            elif submitter.returner.data:
                return redirect(url_for('mainer'))

        return render_template('from_form_7.html', form=submitter)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

class Response(FlaskForm):

    hyperparams = StringField('Гиперпараметры модели', validators=[Optional()])
    train_iter_results = StringField('Значение функции потерь по итерациям на обучающем датасете', validators=[Optional()])
    val_iter_results = StringField('Значение функции потерь по итерациям на валидационном датасете', validators=[Optional()])

    train_dataset = StringField('train_dataset', validators=[Optional()])
    val_dataset = StringField('test_dataset', validators=[Optional()])

    submit = SubmitField('Вернуться на главную страницу')

@app.route('/parameters/RF', methods=['GET', 'POST'])
def rf_params():
    global rf_model, rf_train_dataset_path, rf_val_dataset_path
    try:
        response_form = Response()

        rf_dict = rf_model.__dict__.copy()

        train_results = rf_dict['train_iter_results']
        val_results = rf_dict['val_iter_results']

        rf_dict.pop('train_iter_results', None)
        rf_dict.pop('val_iter_results', None)
        rf_dict.pop('forest', None)
        rf_dict.pop('forest_feat', None)

        f_1 = json.dumps(train_results)
        f_2 = json.dumps(val_results)
        f_3 = json.dumps(rf_dict)

        response_form.hyperparams.data = f_3
        response_form.train_iter_results.data = f_1
        response_form.val_iter_results.data = f_2

        if response_form.validate_on_submit():
            return redirect(url_for('mainer'))

        response_form.train_dataset.data = rf_train_dataset_path
        response_form.val_dataset.data = rf_val_dataset_path

        return render_template('from_form_12.html', form=response_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/parameters/GB', methods=['GET', 'POST'])
def gb_params():
    global gb_model, gb_train_dataset_path, gb_val_dataset_path
    try:
        response_form = Response()

        gb_dict = gb_model.__dict__.copy()

        train_results = gb_dict['train_iter_results']
        val_results = gb_dict['val_iter_results']

        gb_dict.pop('train_iter_results', None)
        gb_dict.pop('val_iter_results', None)
        gb_dict.pop('forest', None)
        gb_dict.pop('alphas', None)
        gb_dict.pop('forest_feat', None)

        f_1 = json.dumps(train_results)
        f_2 = json.dumps(val_results)
        f_3 = json.dumps(gb_dict)

        response_form.hyperparams.data = f_3
        response_form.train_iter_results.data = f_1
        response_form.val_iter_results.data = f_2

        if response_form.validate_on_submit():
            return redirect(url_for('mainer'))

        response_form.train_dataset.data = gb_train_dataset_path
        response_form.val_dataset.data = gb_val_dataset_path

        return render_template('from_form_12.html', form=response_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.errorhandler(404)
def handle_404(e):
    return redirect(url_for('error_handler'))

@app.route('/error_handler', methods=['GET', 'POST'])
def error_handler():
    try:
        subm = FastTravel()
        if subm.validate_on_submit():  
            return redirect(url_for('mainer'))
        return render_template('from_form_4.html', form=subm)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

"""
@app.route('/result', methods=['GET', 'POST'])
def get_result():
    try:
        response_form = Response()

        if response_form.validate_on_submit():
            return redirect(url_for('get_text_score'))

        score = request.args.get('score')
        sentiment = request.args.get('sentiment')

        response_form.score.data = score
        response_form.sentiment.data = sentiment

        return render_template('from_form.html', form=response_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/sentiment', methods=['GET', 'POST'])
def get_text_score():
    try:
        text_form = TextForm()

        if text_form.validate_on_submit():
            app.logger.info('On text: {0}'.format(text_form.text.data))
            score, sentiment = score_text(text_form.text.data)
            app.logger.info("Score: {0:.3f}, Sentiment: {1}".format(score, sentiment))
            text_form.text.data = ''
            return redirect(url_for('get_result', score=score, sentiment=sentiment))
        return render_template('from_form.html', form=text_form)
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/fail')
def get_fail():
    raise ValueError('Fail')


@app.route('/messages', methods=['GET', 'POST'])
def prepare_message():
    message = Message()

    if request.method == 'POST':
        message.header, message.text = request.form['header'], request.form['text']
        messages.append(message)

        return redirect(url_for('prepare_message'))

    return render_template('messages.html', messages=messages)


@app.route('/clear_messages', methods=['POST'])
def clear_messages():
    messages.clear()
    return redirect(url_for('prepare_message'))


@app.route('/file', methods=['GET', 'POST'])
def file():
    file_form = FileForm()

    if request.method == 'POST' and file_form.validate_on_submit():
        lines = file_form.file_path.data.stream.readlines()
        print(f'Uploaded {len(lines)} lines')
        return redirect(url_for('file'))

    return render_template('from_form.html', form=file_form)

"""