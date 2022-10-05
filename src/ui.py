# CODE FOR ALL THE UI COMPONENTS AND BEHAVIOUR

from tkinter import *
from tkinter import ttk

class UI:
    """
        param root : object, Tk object
        param title : string, window bar title
        param w : real number, width of window
        param h : real number, height of window
    """
    def __init__(self, root, title, w, h):
        self.root = root
        self.title = title

        self.root.title(self.title)
        self.root.geometry("{}x{}".format(w, h))

        frm = ttk.Frame(self.root, padding = 10)
        frm.grid()

    """
        Showa the window
    """
    def show(self):
        self.root.mainloop()

    def table(self, data):
        rows = data.shape[0]
        cols = data.shape[1]

        for i in range(rows):
            for j in range(cols):
                b = Entry(self.root)
                b.grid(row=i, column=j)
                b.insert(END, data[i, j])