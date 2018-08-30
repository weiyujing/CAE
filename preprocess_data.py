import pandas as pd
import plotly.offline as offline
import plotly.plotly as py
import matplotlib
from plotly.graph_objs import *
import matplotlib.pyplot as plt
from plotly.tools import FigureFactory as FF
from plotly import figure_factory as FF
offline.init_notebook_mode()
# import plotly.graph_objs as g

#py.sign_in('rosdyana', 'eVtlDykeB8gMHmp6y4Ff')
#py.sign_in('DemoAccount', '2qdyfjyr7o')
#py.sign_in('weiyujing', 'iP5lCejJnq14yRx79eFy')
py.sign_in('tempname', 'Wq9wH7jiUOeYIhLA93yv')
# read stock data
f = open('E:\stock_csv\深振业A.csv')
df = pd.read_csv(f, header=None, index_col=0)
# drop date and volume columns
df.drop(df.columns[[4, 5]], axis=1, inplace=True)
df = df.astype(str)
separators = pd.DataFrame(', ', df.index, df.columns[:-1])
separators[df.columns[-1]] = '\n'
# print (df + separators).sum(axis=1).sum()
data = df[1:]
# print(data.head())

for i in range(0, len(data), 20):

    c = data[i:i + 20]
    # print(c)
    fig = FF.create_candlestick(
        open=c[1], high=c[2], low=c[3], close=c[4])

    fig['layout'].update({
        'xaxis': dict(visible=False),
        'yaxis': dict(visible=False),
        'paper_bgcolor': 'rgba(1,1,1,1)',
        'plot_bgcolor': 'rgba(1,1,1,1)'
    })


    py.image.save_as(fig, filename='E:/shenzhen/{}.png'.format(i))
    

# resize 224x224 imagemagick
# find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -adaptive-resize 224x224! "{}" "{}"
