# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import json
from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/api', methods=['POST'])
# ML Code

def home():
    content_type = request.headers.get('Content-Type')
    if content_type == 'application/json':
        try:
            json_data = request.json
            symbol = json_data['symbol']
            cols = ['Open', 'High', 'Low']
            values = {
                'Open': json_data['Open'],
                'High': json_data['High'],
                'Low': json_data['Low']
            }

            stock = yf.Ticker(symbol)
            df = pd.DataFrame(stock.history(period='2y'))
            # creating the up = 1 and down = 0 target feature for the dataset

            # create a previous close column
            prev_close = pd.DataFrame(df['Close'].shift(1))
            prev_close.rename(columns={'Close': 'Prev_Close'}, inplace=True)

            #
            df = df.drop(['Dividends', 'Stock Splits'], axis=1)
            df = pd.concat([df, prev_close], axis=1)
            df['Target'] = np.where(df['Close'] > df['Prev_Close'], 1, 0)
            df = df.iloc[1:, :]

            # creating the model
            x = df.drop(['Target', 'Prev_Close', 'Volume', 'Close'], axis=1)
            y = df['Close']

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            # train Model
            lr = LinearRegression()
            lr.fit(x_train, y_train)

            # request prediction
            index = [pd.to_datetime('today').normalize()]
            query = pd.DataFrame(values, columns=cols, index=index)
            prediction = {'prediction': lr.predict(query)[0]}

            print(prediction)
            return json.dumps(prediction)

        except:
            return json.dumps({'Error': 'Something went wrong'})
