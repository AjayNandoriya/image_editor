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

from datetime import datetime as dt


def format_date(dt_, fmt="%m/%d/%Y, %H:%M:%S"):
    return f"{dt_:{fmt}}"


def now(fmt="%m/%d/%Y, %H:%M:%S"):
    return format_date(dt.now(), fmt)


def remove_class(element, class_name):
    element.element.classList.remove(class_name)


def add_class(element, class_name):
    element.element.classList.add(class_name)


