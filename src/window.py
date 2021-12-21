import tkinter as tk
from PIL import Image, ImageTk
import cv2
from utils import OPERATIONS_WINDOW_LABELS, OPERATIONS_WINDOW_TITLES


class Window:
    def __init__(self, operation: int):
        self.window = tk.Tk()
        self.window.title(OPERATIONS_WINDOW_TITLES[operation])

        self.label = tk.Label(
            self.window,
            text=OPERATIONS_WINDOW_LABELS[operation],
            pady=20,
            font=("sans-serif", 24),
        )
        self.canvas = tk.Label(self.window)
        self.button = tk.Button(
            self.window,
            text="Shot picture",
            padx=20,
            pady=10,
            font=("sans-serif", 18, "bold"),
            command=self._destroy_with_success,
        )

        self.label.pack()
        self.canvas.pack()
        self.button.pack(pady=20)
        self.button.focus_set()

        self.shot_button_pressed = False
        self.capture = cv2.VideoCapture(0)
        self.frame = None

        self.start_video_loop()
        self.window.mainloop()

    def start_video_loop(self):
        _, self.frame = self.capture.read()
        decorated_frame = self._decorate_frame()

        image = cv2.cvtColor(decorated_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image=image)

        self.canvas.imgtk = image
        self.canvas.configure(image=image)
        self.window.after(10, self.start_video_loop)

    def _destroy_with_success(self):
        self.shot_button_pressed = True
        self.window.destroy()

    def _decorate_frame(self):
        height = len(self.frame)
        width = len(self.frame[0])

        ellipse_mask = cv2.numpy.zeros_like(self.frame)
        ellipse_mask = cv2.ellipse(
            ellipse_mask,
            (int(width / 2), int(height / 2)),
            (int(width / 3), int(height / 2)),
            0,
            0,
            360,
            (255, 255, 255),
            -1,
        )

        outer_mask = cv2.bitwise_not(ellipse_mask)
        outer_image = cv2.bitwise_and(self.frame, outer_mask)
        ellipse_image = cv2.bitwise_and(self.frame, ellipse_mask)
        return cv2.addWeighted(ellipse_image, 1.0, outer_image, 0.5, 1)
