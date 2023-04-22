from js import document, console, Uint8Array, window, File
from pyodide.ffi import create_proxy
import asyncio
import io
import matplotlib
matplotlib.use("module://matplotlib_pyodide.html5_canvas_backend")
# matplotlib.use("module://matplotlib_pyodide.wasm_backend")
# matplotlib.use("module://matplotlib_pyodide.browser_backend")

from process import MyApp
from PIL import Image, ImageFilter
import numpy as np
import logging
import json
LOGGER = logging.getLogger(__name__)
myApp = MyApp()

delta = 0.025
                
def demo_mpl():
    import matplotlib.cm as cm
    from matplotlib import pyplot as plt
    delta = 0.025
    x = y = np.arange(-3.0, 3.0, delta)
    X, Y = np.meshgrid(x, y)
    Z1 = np.exp(-(X**2) - Y**2)
    Z2 = np.exp(-((X - 1) ** 2) - (Y - 1) ** 2)
    Z = (Z1 - Z2) * 2
    plt.figure(1)
    plt.imshow(
    Z,
    interpolation="bilinear",
    cmap=cm.RdYlGn,
    origin="lower",
    extent=[-3, 3, -3, 3],
    vmax=abs(Z).max(),
    vmin=-abs(Z).max(),
    )
    plt.show()
    pass

def plot_mpl(fig, ele_id):
    demo_mpl()
    # display(fig, target=ele_id, append=False)
    pass

def plot(fig, ele_id):
    # plot_plotly(fig, ele_id)
    plot_mpl(fig, ele_id)

async def _upload_ref(e):
    print('upload ref')
    LOGGER.info('upload ref')
    myApp.ref_img = await _upload_change_and_show(e)
    # display(myApp.plot(), target="output_upload_pillow", append=False)

    # fig = myApp.plot()
    # plot(fig, "output_upload_pillow")

async def _upload_test(e):
    myApp.test_img = await _upload_change_and_show(e)
    # fig = myApp.plot()
    # plot(fig, "output_upload_pillow")
    

async def _upload_base(e):
    myApp.base_img = await _upload_change_and_show(e)
    # display(myApp.plot(), target="output_upload_pillow", append=False)

    # fig = myApp.plot()
    # plot(fig, "output_upload_pillow")
    

async def _run(e):
    myApp.run()
    # display(myApp.plot(), target="output_upload_pillow", append=False)

    # fig = myApp.plot()
    # plot(fig, "output_upload_pillow")
    

async def _upload_change_and_show(e):
    #Get the first file from upload
    file_list = e.target.files
    first_item = file_list.item(0)

    #Get the data from the files arrayBuffer as an array of unsigned bytes
    array_buf = Uint8Array.new(await first_item.arrayBuffer())

    #BytesIO wants a bytes-like object, so convert to bytearray first
    bytes_list = bytearray(array_buf)
    my_bytes = io.BytesIO(bytes_list) 

    LOGGER.info(my_bytes)
    #Create PIL image from np array
    my_image = Image.open(my_bytes)

    # convert to numpy
    img = np.asarray(my_image)
    LOGGER.info(f'received image with shape {img.shape}')
    print('got image')
    if len(img.shape) == 3:
        img = img[:,:,0]
    return img

    #Log some of the image data for testing
    console.log(f"{my_image.format= } {my_image.width= } {my_image.height= }")

    # Now that we have the image loaded with PIL, we can use all the tools it makes available. 
    # "Emboss" the image, rotate 45 degrees, fill with dark green
    my_image = my_image.filter(ImageFilter.EMBOSS).rotate(45, expand=True, fillcolor=(0,100,50)).resize((300,300))

    # #Convert Pillow object array back into File type that createObjectURL will take
    # my_stream = io.BytesIO()
    # my_image.save(my_stream, format="PNG")

    # #Create a JS File object with our data and the proper mime type
    # image_file = File.new([Uint8Array.new(my_stream.getvalue())], "new_image_file.png", {type: "image/png"})

    # #Create new tag and insert into page
    # new_image = document.createElement('img')
    # new_image.src = window.URL.createObjectURL(image_file)
    # document.getElementById("output_upload_pillow").appendChild(new_image)

# Run image processing code above whenever file is uploaded    
document.getElementById("ref-file-upload-pillow").addEventListener("change", create_proxy(_upload_ref))
document.getElementById("test-file-upload-pillow").addEventListener("change", create_proxy(_upload_test))
document.getElementById("base-file-upload-pillow").addEventListener("change", create_proxy(_upload_base))
document.getElementById("run").addEventListener("click", create_proxy(_run))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # demo_mpl()
    pass