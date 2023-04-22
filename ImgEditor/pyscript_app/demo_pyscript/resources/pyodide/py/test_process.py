import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

if __package__ is None or __package__ == '':
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import process

def test_plot_plotly():
    img = np.eye(100)
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, shared_yaxes=True)
    # fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=1)
    # fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=2)
    # fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=2, col=1)
    # fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=2, col=2)


    fig.add_trace(px.imshow(img).data[0], row=1, col=1)
    fig.add_trace(px.imshow(img).data[0], row=1, col=2)
    fig.add_trace(px.imshow(img).data[0], row=2, col=1)
    fig.add_trace(px.imshow(img).data[0], row=2, col=2)

    fig.show()

    pass

def test_myapp_plotly():
    import cv2
    myapp = process.MyApp()
    ref_img_fname = r"C:\Users\ajayn\Pictures\sem1.png"
    ref_img = cv2.imread(ref_img_fname,0)
    myapp.ref_img = ref_img
    fig = myapp.plot_plotly()

    import pandas as pd
    from plotly import express as px 
    df = pd.DataFrame(dict(
        x = [1, 3, 2, 4],
        y = [1, 2, 3, 4]
    ))
    fig = px.line(df, x="x", y="y", title="Unsorted Input") 

    fig_html = fig.to_html(
        include_plotlyjs=False,
        full_html=False,
        default_height='350px'
      )
    print(fig_html)
    # fig.show()
    pass

if __name__ == '__main__':
    # test_plot_plotly()
    test_myapp_plotly()
    pass
