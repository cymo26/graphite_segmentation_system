import tkinter as tk
from tkinter import ttk
from gui import GraphiteSegmentationApp


def main():
    root = tk.Tk()
    
    # Styl
    style = ttk.Style()
    style.theme_use('clam')
    
    app = GraphiteSegmentationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
