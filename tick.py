import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw, ImageOps  # 确保导入 ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import io

# 加载预训练模型
model = load_model('model.h5')

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("手写数字识别")
        self.initUI()

    def initUI(self):
        self.canvas = Canvas(self.master, width=280, height=280, bg='white')
        self.canvas.pack()

        self.clear_btn = Button(self.master, text="清除", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT)

        self.predict_btn = Button(self.master, text="预测", command=self.predict)
        self.predict_btn.pack(side=tk.RIGHT)

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.result_label = tk.Label(self.master, text="", font=("Helvetica", 24))
        self.result_label.pack()

    def paint(self, event):
        x, y = event.x, event.y
        r = 10  # 笔触半径
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")

    def predict(self):
        img = self.image.resize((28, 28), Image.Resampling.LANCZOS).convert('L')  # 使用新的重采样滤波器
        img = ImageOps.invert(img)  # 反转颜色
        img = np.array(img)
        img = img / 255.0  # 归一化
        x = img.reshape(1, 28, 28, 1)

        preds = model.predict(x)
        predicted_digit = np.argmax(preds[0])
        self.result_label.config(text=f"预测的数字是: {predicted_digit}")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()