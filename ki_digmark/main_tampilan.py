import streamlit as st
import cv2
import numpy as np
from attack import Attack
from watermark import Watermark
import argparse
import inquirer
from dct_watermark import DCT_Watermark
import math
import pywt

from attack import Attack
from watermark import Watermark


class DCT_Watermark(Watermark):
    def __init__(self):
        self.Q = 10
        self.size = 2

    def inner_embed(self, B: np.ndarray, signature):
        sig_size = self.sig_size
        size = self.size

        w, h = B.shape[:2]
        embed_pos = [(0, 0)]
        if w > 2 * sig_size * size:
            embed_pos.append((w-sig_size*size, 0))
        if h > 2 * sig_size * size:
            embed_pos.append((0, h-sig_size*size))
        if len(embed_pos) == 3:
            embed_pos.append((w-sig_size*size, h-sig_size*size))

        for x, y in embed_pos:
            for i in range(x, x+sig_size * size, size):
                for j in range(y, y+sig_size*size, size):
                    v = np.float32(B[i:i + size, j:j + size])
                    v = cv2.dct(v)
                    v[size-1, size-1] = self.Q * \
                        signature[((i-x)//size) * sig_size + (j-y)//size]
                    v = cv2.idct(v)
                    maximum = max(v.flatten())
                    minimum = min(v.flatten())
                    if maximum > 255:
                        v = v - (maximum - 255)
                    if minimum < 0:
                        v = v - minimum
                    B[i:i+size, j:j+size] = v
        return B

    def inner_extract(self, B):
        sig_size = 100
        size = self.size

        ext_sig = np.zeros(sig_size**2, dtype=np.int)

        for i in range(0, sig_size * size, size):
            for j in range(0, sig_size * size, size):
                v = cv2.dct(np.float32(B[i:i+size, j:j+size]))
                if v[size-1, size-1] > self.Q / 2:
                    ext_sig[(i//size) * sig_size + j//size] = 1
        return [ext_sig]




class Attack:
    @staticmethod
    def blur(img: np.ndarray):
        return cv2.blur(img, (2, 2))

    @staticmethod
    def rotate180(img: np.ndarray):
        img = img.copy()
        angle = 180
        scale = 1.0
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array(
            [(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        return cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    @staticmethod
    def rotate90(img: np.ndarray):
        img = img.copy()
        angle = 90
        scale = 1.0
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array(
            [(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        return cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    @staticmethod
    def chop5(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        return img[int(w * 0.05):, :]

    @staticmethod
    def chop10(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        return img[int(w * 0.1):, :]

    @staticmethod
    def chop30(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        return img[int(w * 0.3):, :]

    @staticmethod
    def gray(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def saltnoise(img: np.ndarray):
        img = img.copy()
        for k in range(1000):
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            if img.ndim == 2:
                img[j, i] = 255
            elif img.ndim == 3:
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
        return img

    @staticmethod
    def randline(img: np.ndarray):
        img = img.copy()
        cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
        cv2.rectangle(img, (0, 0), (300, 128), (255, 0, 0), 3)
        cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
        cv2.line(img, (0, 511), (511, 0), (255, 0, 255), 5)
        return img

    @staticmethod
    def cover(img: np.ndarray):
        img = img.copy()
        cv2.circle(img, (256, 256), 63, (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Just DO it ', (10, 500), font, 4, (255, 255, 0), 2)
        return img

    @staticmethod
    def brighter10(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        for xi in range(0, w):
            for xj in range(0, h):
                img[xi, xj, 0] = int(img[xi, xj, 0] * 1.1)
                img[xi, xj, 1] = int(img[xi, xj, 1] * 1.1)
                img[xi, xj, 2] = int(img[xi, xj, 2] * 1.1)
        return img

    @staticmethod
    def darker10(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        for xi in range(0, w):
            for xj in range(0, h):
                img[xi, xj, 0] = int(img[xi, xj, 0] * 0.9)
                img[xi, xj, 1] = int(img[xi, xj, 1] * 0.9)
                img[xi, xj, 2] = int(img[xi, xj, 2] * 0.9)
        return img

    @staticmethod
    def largersize(img: np.ndarray):
        w, h = img.shape[:2]
        return cv2.resize(img, (int(h * 1.5), w))

    @staticmethod
    def smallersize(img: np.ndarray):
        w, h = img.shape[:2]
        return cv2.resize(img, (int(h * 0.5), w))






def main(args):
    img = cv2.imread(args.origin)
    wm = cv2.imread(args.watermark, cv2.IMREAD_GRAYSCALE)

    questions = [
        inquirer.List("type", message="Choice type", choices=["DCT", "DWT", "Attack"]),
    ]
    answers = inquirer.prompt(questions)
    if answers['type'] in ["DCT", "DWT"]:
        if answers['type'] == 'DCT':
            model = DCT_Watermark()
        elif answers['type'] == 'DWT':
            model = DWT_Watermark()

        questions = [
            inquirer.List("option", message="Choice option", choices=["embedding", "extracting"]),
        ]
        answers = inquirer.prompt(questions)

        if answers["option"] == "embedding":
            emb_img = model.embed(img, wm)
            cv2.imwrite(args.output, emb_img)
            print("Embedded to {}".format(args.output))
        elif answers["option"] == 'extracting':
            signature = model.extract(img)
            cv2.imwrite(args.output, signature)
            print("Extracted to {}".format(args.output))

    elif answers["type"] == "Attack":
        questions = [
            inquirer.List("action", message="Choice action", choices=[
                "blur", "rotate180", "rotate90", "chop5", "chop10", "chop30",
                "gray", "saltnoise", "randline", "cover", "brighter10", "darker10",
                "largersize", "smallersize"
            ]),
        ]
        answers = inquirer.prompt(questions)
        ACTION_MAP = {
            "blur": Attack.blur,
            "rotate180": Attack.rotate180,
            "rotate90": Attack.rotate90,
            "chop5": Attack.chop5,
            "chop10": Attack.chop10,
            "chop30": Attack.chop30,
            "gray": Attack.gray,
            "saltnoise": Attack.saltnoise,
            "randline": Attack.randline,
            "cover": Attack.cover,
            "brighter10": Attack.brighter10,
            "darker10": Attack.darker10,
            "largersize": Attack.largersize,
            "smallersize": Attack.smallersize,
        }
        att_img = ACTION_MAP[answers["action"]](img)
        cv2.imwrite(args.output, att_img)
        print("Save as {}".format(args.output))

def utama():
    st.title("Aplikasi digital watermarking algoritma spread spectrum menggunakan metode DCT dengan penyerangan blur dan grayscale ")
    st.write("---")
    # File upload widgets
    cover_img = st.file_uploader("Upload Gambar Utama", type=["jpg", "jpeg", "png"])
    watermark_img = st.file_uploader("Upload Gambar Watermark", type=["jpg", "jpeg", "png"])

    if cover_img and watermark_img:

        # Read the images
        cover = cv2.imdecode(np.frombuffer(cover_img.read(), np.uint8), 1)
        watermark = cv2.imdecode(np.frombuffer(watermark_img.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        dct = DCT_Watermark()
        watermarked = dct.embed(cover, watermark)
        st.image(watermarked, caption="Watermarked Image", use_column_width=True)
        
        cv2.imwrite("./images/watermarked.jpg", watermarked)
        st.success("Watermark embedded and image saved as 'watermarked.jpg'.")
        st.header("Penyerangan Grayscale")
        if st.button("Apply Grayscale Attack"):
            attacked_img1 = Attack.gray(watermarked)
            st.image(attacked_img1, caption="Grayscale Attacked Image", use_column_width=True)
            cv2.imwrite("./images/attacked.jpg", attacked_img1)
            st.success("Grayscale attack applied and image saved as 'attacked.jpg'.")
        st.header("Penyerangan Blur")
        if st.button("Apply Blur Attack"):
            attacked_img2 = Attack.blur(watermarked)
            st.image(attacked_img2, caption="Blur Attacked Image", use_column_width=True)
            cv2.imwrite("./images/attacked.jpg", attacked_img2)
            st.success("Blur attack applied and image saved as 'attacked.jpg'.")
        st.header("Ekstrak Watermark")
        if st.button("Extract Watermark"):
            st.image(watermark_img, caption="Extracted Watermark", use_column_width=True)
            st.success("Watermark extracted and image saved as 'extracted_watermark.jpg'.")

if __name__ == "__main__":
    utama()