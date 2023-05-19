from utils import plot
import base64
import js
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import plotly.express as px

fig = plot()
display(fig, target="graph-area", append=False)
from pyodide.ffi import create_proxy

def select_flavour(event):
    display(event.target.value,target="py_output", append=False)
ele_proxy = create_proxy(select_flavour)

flavour_elements = js.document.getElementsByName("flavour");
for ele in flavour_elements:
    if ele.value == "ALL":
        ele.checked = True
        current_selected = ele.value
    ele.addEventListener("change", ele_proxy)

read_btn = js.document.getElementById("readImg");
def read_img(e):
    print(e)
    board_canvas = js.document.getElementById("board")

    b64 = board_canvas.toDataURL()
    print(b64)
    ind = b64.find('base64')
    coded_string = b64[ind+7:]
    print(b64[ind+7:ind+10])
    decoded = base64.b64decode(coded_string)
    a = np.frombuffer(decoded, dtype=np.uint8)
    print(dir(board_canvas))
    # a = a.reshape((board_canvas.height, board_canvas.width))
    print(a.shape, type(decoded), ind)
    

    gif_bytes_io = BytesIO() # or io.BytesIO()
    # store the gif bytes to the IO and open as image
    gif_bytes_io.write(decoded)
    image = Image.open(gif_bytes_io)
    img = np.asarray(image)

    plt.rcParams["figure.figsize"] = (22,20)
    fig, ax = plt.subplots()
    plt.imshow(img)
    plt.title("image from py")
    display(fig, target="graph-area", append=False)
    pass

read_btn.addEventListener("click", create_proxy(read_img))


# 

def read_buffer(x):
    print(dir(x))
    decoded = x.to_bytes()
    gif_bytes_io = BytesIO() # or io.BytesIO()
    # store the gif bytes to the IO and open as image
    gif_bytes_io.write(decoded)
    image = Image.open(gif_bytes_io)
    img = np.asarray(image)

    plt.rcParams["figure.figsize"] = (22,20)
    fig, ax = plt.subplots()
    plt.imshow(img)
    plt.title("image from py")
    display(fig, target="graph-area", append=False)
    
    
def read_file(event):
    print(dir(event.target.files))
    data = event.target.files
    print(dir(data.item(0)))
    print(data.item(0).name)
    data.item(0).arrayBuffer().then(read_buffer)
    
inputElement = js.document.getElementById("fileInput")
inputElement.addEventListener("change", create_proxy(read_file))
