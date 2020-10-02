from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn import metrics

import pickle

# load the model from disk
loaded_model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html', show_prediction=False)


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('real_2018.csv')
    my_prediction = loaded_model.predict(df.iloc[:, :-1].values)
    df['my_prediction'] = my_prediction

    predictions = np.array(df['my_prediction'])
    test_labels = np.array(df['PM 2.5'])

    errors = np.round(abs(predictions - test_labels), 2)
    mape = np.round(100 * np.mean(errors / test_labels), 2)
    accuracy = 100 - mape
    # r2_score = np.round(metrics.r2_score(test_labels, predictions), 2)
    # rmse = np.round(np.sqrt(metrics.mean_squared_error(test_labels, predictions), 2))

    return render_template('home.html', show_prediction=True, tables=[df.to_html(classes='data')], titles=df.columns.values,
                           mape=mape, accuracy=accuracy, r2_score=1, rmse=1)


import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


@app.route('/plot.png')
def plot_png():
    fig = create_figure_scatter()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def create_figure_scatter():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    df = pd.read_csv('real_2018.csv')
    my_prediction = loaded_model.predict(df.iloc[:, :-1].values)
    df['my_prediction'] = my_prediction

    xs = df['PM 2.5']
    ys = df['my_prediction']
    axis.scatter(xs, ys)
    return fig


if __name__ == '__main__':
    app.run(debug=True)
