from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
