import tkinter as tk

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure the root window
        self.title('Image Editor')
        self.geometry('900x600')
        self.config(bg="skyblue")


if __name__ == '__main__':
    app = MainApp()
    app.mainloop()

