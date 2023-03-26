from datetime import datetime
now = datetime.now()
#display(now.strftime("%m/%d/%Y, %H:%M:%S"))

import matplotlib.pyplot as plt
import numpy as np
def plot():
    plt.rcParams["figure.figsize"] = (22,20)
    fig, ax = plt.subplots()
    data = np.random.rand(100,100)
    plt.imshow(data)
    plt.title("Rating of ice cream flavours of your choice")
    return fig



