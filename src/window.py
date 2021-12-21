from utils import OPERATIONS_WINDOW_LABELS, OPERATIONS_WINDOW_TITLES
import tkinter as tk
from PIL import Image, ImageTk
import cv2


class Window:
    def __init__(self, operation: int):
        self.window = tk.Tk()
        self.window.title(OPERATIONS_WINDOW_TITLES[operation])
        self.label = tk.Label(self.window, text=OPERATIONS_WINDOW_LABELS[operation])
        self.canvas = tk.Label(self.window)
        self.button = tk.Button(
            self.window, text="Shot picture", command=self._destroy_with_success
        )

        self.label.pack()
        self.canvas.pack()
        self.button.pack()

        self.shot_button_pressed = False
        self.capture = cv2.VideoCapture(0)
        self.frame = None

        self.start_video_loop()
        self.window.mainloop()

    def start_video_loop(self):
        _, self.frame = self.capture.read()
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image=image)
        self.canvas.imgtk = image
        self.canvas.configure(image=image)
        self.window.after(10, self.start_video_loop)

    def _destroy_with_success(self):
        self.shot_button_pressed = True
        self.window.destroy()
