# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:53:01 2023

@author: MPV3KOR
"""

# ===== Command line call
###################################
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter import *
from tkinter import Frame
import string
import time
from PIL import ImageTk, Image
from pixelmatch.contrib.PIL import pixelmatch
import numpy as np
from SSIM_PIL import compare_ssim
from PIL import ImageEnhance
import PIL as pil
from io import StringIO

out_cipherText = ""
out_plainText = ""


def encrypt_Vigenre(plaintext, key):
    print("Encrypted Vigenre")
    print(len(plaintext))
    plaintext = plaintext  # .upper()
    key = key  # .upper()
    # alphabet = string.ascii_uppercase
    ciphertext = ""
    key_idx = 0
    for char in plaintext:
        char_idx = 0
        if char in default_characters:
            shift = default_characters.index(key[key_idx])
            char_idx = default_characters.index(char)
            ciphertext += default_characters[(char_idx) % 101]
            key_idx = (key_idx + 1) % len(key)
        else:
            ciphertext += default_characters[(char_idx) % 101]
    print(len(ciphertext))
    return ciphertext


def decrypt_Vigenre(ciphertext, key):
    print("Decrypted Vigenre")
    ciphertext = ciphertext  # .upper()
    key = key  # .upper()
    # alphabet = string.ascii_uppercase
    plaintext = ""
    key_idx = 0

    for char in ciphertext:
        char_idx = 0
        if char in default_characters:
            shift = default_characters.index(key[key_idx])
            char_idx = default_characters.index(char)
            plaintext += default_characters[(char_idx) % 101]
            key_idx = (key_idx + 1) % len(key)
        else:
            plaintext += default_characters[(char_idx) % 101]
    return plaintext


default_characters = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
                      "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
                      "T", "U", "V", "W", "X", "Y", "Z", "a", "b", "¶",
                      "c", "d", "e", "f", "g", "h", "i", "j", "k", "¥",
                      "l", "m", "n", "o", "p", "q", "r", "s", "t", "~",
                      "u", "v", "w", "x", "y", "z", "1", "2", "3", "©",
                      "4", "5", "6", "7", "8", "9", "0", "!", "@", "Å",
                      "#", "$", "%", "^", "&", "*", "(", ")", "_", "€",
                      "-", "+", "=", ".", ">", "<", ",", "/", "?", "𐐷",
                      ":", ";", "]", "[", "{", "}", "|", "`", "š"]


# def find_index(input_word: str, cha"Ž"racter_list: list) -> np.array:
# return np.array([character_list.index(digit) for digit in input_word])

# def find_word(input_index: list, character_list: list) -> str:
# char_list = [character_list[i % len(character_list)] for i in input_index]

# return "".join(char_list)

# def viginere_word(message: str, key: str, character_list: list, encode: bool = True) -> str:

# encode = 1 if encode else -1

# # Turn words into index arrays on the character_list list.
# message_ = find_index(message, character_list)
# key_ = find_index(key, character_list)

# # Repeat and crop the key to match word.
# key_ = np.tile(key_, int(len(message_)/len(key_)) +1)[:len(message_)]

# # Shift the index using the key and remap to character_list.
# resulting_message_index = message_ +key_ *encode
# resulting_message = find_word(resulting_message_index, character_list)
# resulting_message = "".join(resulting_message)

# return resulting_message

# # ==== User Interface functions ====

# def encrypt_Vigenre(message: str, key: str, character_list: list = default_characters) -> str:

# encoded_message = viginere_word(message, key, character_list, encode = True)

# return encoded_message

# def decrypt_Vigenre(message: str, key: str, character_list: list = default_characters) -> str:

# decoded_message = viginere_word(message, key, character_list, encode = False)

# return decoded_message


def encrypt_PolyBius(plaintext):
    print("Encrypted PolyBius")
    square = [
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "¶"],
        ["J", "K", "L", "M", "N", "O", "P", "Q", "R", "¥"],
        ["S", "T", "U", "V", "W", "X", "Y", "Z", "a", "~"],
        ["b", "c", "d", "e", "f", "g", "h", "i", "j", "©"],
        ["k", "l", "m", "n", "o", "p", "q", "r", "s", "Å"],
        ["t", "u", "v", "w", "x", "y", "z", "1", "2", "€"],
        ["3", "4", "5", "6", "7", "8", "9", "0", "!", "𐐷"],
        ["@", "#", "$", "%", "^", "&", "*", "(", ")", "œ"],
        ["_", "-", "+", "=", ".", ">", "<", ",", "/", "š"],
        ["?", ":", ";", "]", "[", "{", "}", "|", "`", "Ž"]]

    # Convert the plaintext to uppercase and remove spaces
    plaintext = plaintext  # .upper()#.replace(" ", "")

    # Replace each letter with its corresponding Polybius square coordinates
    ciphertext = ""
    for letter in plaintext:
        for row in range(len(square)):
            if letter in square[row]:
                col = square[row].index(letter)
                ciphertext += str(row + 1) + str(col + 1)
                break
    print(len(ciphertext))
    return ciphertext


def decrypt_PolyBius(ciphertext):
    print("Decrypted PolyBius")
    square = [
        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "¶"],
        ["J", "K", "L", "M", "N", "O", "P", "Q", "R", "¥"],
        ["S", "T", "U", "V", "W", "X", "Y", "Z", "a", "~"],
        ["b", "c", "d", "e", "f", "g", "h", "i", "j", "©"],
        ["k", "l", "m", "n", "o", "p", "q", "r", "s", "Å"],
        ["t", "u", "v", "w", "x", "y", "z", "1", "2", "€"],
        ["3", "4", "5", "6", "7", "8", "9", "0", "!", "𐐷"],
        ["@", "#", "$", "%", "^", "&", "*", "(", ")", "œ"],
        ["_", "-", "+", "=", ".", ">", "<", ",", "/", "š"],
        ["?", ":", ";", "]", "[", "{", "}", "|", "`", "Ž"]]

    # Split the ciphertext into pairs of digits and convert to integers
    digits = [int(ciphertext[i:i + 2]) for i in range(0, len(ciphertext), 2)]

    # Convert each pair of digits to the corresponding letter
    plaintext = ""
    for digit_pair in digits:
        row = 0
        col = 0
        al = list(map(int, str(digit_pair)))
        if (len(al) == 1):
            row = al[0]
            col = 0
        if (len(al) == 2):
            row = al[0]
            col = al[1]

        plaintext += square[row - 1][col - 1]
    return plaintext


def encrypt(EncryptedTextBox, plaintext, key):
    vigenre_cipher = encrypt_Vigenre(str(plaintext), str(key))
    out_cipherText = encrypt_PolyBius(str(vigenre_cipher))
    EncryptedTextBox.insert(tk.END, out_cipherText)
    EncryptedTextBox.configure(state='normal')
    return


def encryptImage(plaintext, key):
    key = len(key) + 128
    plaintext = bytearray(plaintext)
    for index, values in enumerate(plaintext):
        plaintext[index] = values ^ key

    return plaintext


def decrypt(DecryptedTextBox, ciphertext, key):
    polyBius_decrypted = decrypt_PolyBius(ciphertext)
    out_plainText = decrypt_Vigenre(polyBius_decrypted, key)
    DecryptedTextBox.insert(tk.END, out_plainText)


def decryptImage(ciphertext, key):
    decCipher = ciphertext
    key = len(key) + 128
    decCipher = bytearray(decCipher)
    for index, values in enumerate(decCipher):
        decCipher[index] = values ^ key

    return decCipher


def submit(plainTextBox, keyTextBox, EncryptedTextBox, DecryptedTextBox, EncryptedTimeTextBox, DecryptedTimeTextBox):
    EncryptedTimeTextBox.delete("1.0", "end")
    DecryptedTimeTextBox.delete("1.0", "end")
    EncryptedTextBox.delete("1.0", "end")
    DecryptedTextBox.delete("1.0", "end")

    start_encrypt = time.perf_counter()
    encrypt(EncryptedTextBox, plainTextBox.get('1.0', 'end-1c'), keyTextBox.get('1.0', 'end-1c'))
    end_encrypt = time.perf_counter()
    EncryptedTimeTextBox.insert(tk.END, str((end_encrypt - start_encrypt) * 10 ** 6) + " ms")

    start_decrypt = time.perf_counter()
    decrypt(DecryptedTextBox, EncryptedTextBox.get('1.0', 'end-1c'), keyTextBox.get('1.0', 'end-1c'))
    end_decrypt = time.perf_counter()
    DecryptedTimeTextBox.insert(tk.END, str((end_decrypt - start_decrypt) * 10 ** 6) + " ms")


def Text_UI(root):
    InputText = Label(root, text="Input Plain text")
    InputText.pack()
    InputText.place(x=100, y=100)
    KeyText = Label(root, text="Cipher Key text")
    KeyText.pack()
    KeyText.place(x=100, y=250)

    EncryptedText = Label(root, text="Encrypted text")
    EncryptedText.pack()
    EncryptedText.place(x=800, y=100)

    DecryptedText = Label(root, text="Decrypted text")
    DecryptedText.pack()
    DecryptedText.place(x=800, y=250)

    plainTextBox = Text(
        root,
        height=10,
        width=60
    )
    plainTextBox.pack()
    plainTextBox.place(x=200, y=0)

    keyTextBox = Text(
        root,
        height=10,
        width=60
    )
    keyTextBox.pack()
    keyTextBox.place(x=200, y=200)

    EncryptedTextBox = Text(
        root,
        height=10,
        width=60
    )
    EncryptedTextBox.pack()
    EncryptedTextBox.place(x=1000, y=0)
    EncryptedTextBox.configure(state='normal')

    DecryptedTextBox = Text(
        root,
        height=10,
        width=60
    )
    DecryptedTextBox.pack()
    DecryptedTextBox.place(x=1000, y=200)
    DecryptedTextBox.configure(state='normal')

    EncryptedTimeTextBox = Text(
        root,
        height=1,
        width=8
    )

    EncryptedTimeTextBox.pack()
    EncryptedTimeTextBox.place(x=1000, y=550)
    EncryptedTimeTextBox.configure(state='normal')

    DecryptedTimeTextBox = Text(
        root,
        height=1,
        width=8
    )

    DecryptedTimeTextBox.pack()
    DecryptedTimeTextBox.place(x=1000, y=650)
    DecryptedTimeTextBox.configure(state='normal')



    submit_button = ttk.Button(
        root,
        text='Submit',
        command=lambda: submit(plainTextBox, keyTextBox, EncryptedTextBox, DecryptedTextBox, EncryptedTimeTextBox,
                               DecryptedTimeTextBox)
    )

    submit_button.pack(expand=True)
    submit_button.place(x=300, y=400)


def select_file(root):
    filetypes = (
        ('Image files', '*.png'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    renderImage(root, filename)


def Image_UI(root):
    open_button = ttk.Button(
        root,
        text='Select image to Encrypt',
        command=lambda: select_file(root)
    )
    open_button.pack()
    open_button.place(x=0, y=0)


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def Encrypt_Decrypt_Image(im):
    # norm_im = ((im/255)*65)+25
    # norm_im = np.clip(norm_im, 65, 90)
    # ascii_im = norm_im.astype('uint').tobytes().decode("ascii")

    key = "abcd"
    out_cipherText = encryptImage(im, key)
    out_plainText = decryptImage(out_cipherText, key)

    return out_cipherText, out_plainText


def renderImage(root, image_file):
    InputImage = Label(root, text="Image to Encrypt")
    InputImage.pack()
    InputImage.place(x=50, y=30)

    fobj = open(image_file, 'rb')
    image_to_ecDec = fobj.read()
    fobj.close()
    original_img = Image.open(image_file)
    renorm_np_Enc, renorm_np_final = Encrypt_Decrypt_Image(image_to_ecDec)
    fobj = open(r"temp.png", 'wb')
    fobj.write(renorm_np_final)
    fobj.close()
    EncDec_img = Image.open(r"temp.png")

    Enc_img = Image.fromarray(np.array(bytearray(renorm_np_Enc)))

    resize_EncDec_img = EncDec_img.resize((400, 400))
    resize_image_orig = original_img.resize((400, 400))
    resize_Enc_img = Enc_img.resize((400, 400))
    img = ImageTk.PhotoImage(resize_image_orig)
    canvas = Label(root, width=400, height=400, image=img)
    canvas.pack()
    canvas.place(x=50, y=50)

    InputImage = Label(root, text="Decrypted Image")
    InputImage.pack()
    InputImage.place(x=500, y=30)

    img2 = ImageTk.PhotoImage(resize_EncDec_img)
    canvas_3 = Label(root, width=400, height=400, image=img2)
    canvas_3.pack()
    canvas_3.place(x=500, y=50)

    InputImage = Label(root, text="Encrypted Image")
    InputImage.pack()
    InputImage.place(x=50, y=480)

    img5 = ImageTk.PhotoImage(resize_Enc_img)
    canvas_2 = Label(root, width=400, height=400, image=img5)
    canvas_2.pack()
    canvas_2.place(x=50, y=500)

    ImageMSE = mse(np.array(original_img), np.array(EncDec_img))
    ImagePSNR = PSNR(np.array(original_img), np.array(EncDec_img))
    ImageSSIM = compare_ssim(original_img, EncDec_img)
    Image_CORCOEFF = np.corrcoef(np.array(original_img).flat,  np.array(EncDec_img).flat)

    MSEText = Label(root, text="MSE")
    MSEText.pack()
    MSEText.place(x=800, y=500)

    PSNRText = Label(root, text="PSNR")
    PSNRText.pack()
    PSNRText.place(x=800, y=570)

    CCOEFFText = Label(root, text="Correlation Coefficient")
    CCOEFFText.pack()
    CCOEFFText.place(x=800, y=650)

    SSIMText = Label(root, text="SSIM")
    SSIMText.pack()
    SSIMText.place(x=800, y=750)

    MSETextBox = Text(
        root,
        height=1,
        width=8
    )
    MSETextBox.pack()
    MSETextBox.place(x=1000, y=500)
    MSETextBox.configure(state='normal')
    MSETextBox.insert(tk.END,ImageMSE)

    PSNRTextBox = Text(
        root,
        height=1,
        width=8
    )

    PSNRTextBox.pack()
    PSNRTextBox.place(x=1000, y=570)
    PSNRTextBox.configure(state='normal')
    PSNRTextBox.insert(tk.END, ImagePSNR)

    CCOEFFTextBox = Text(
        root,
        height=3,
        width=8
    )

    CCOEFFTextBox.pack()
    CCOEFFTextBox.place(x=1000, y=650)
    CCOEFFTextBox.configure(state='normal')
    CCOEFFTextBox.insert(tk.END, Image_CORCOEFF)

    SSIMTextBox = Text(
        root,
        height=1,
        width=8
    )

    SSIMTextBox.pack()
    SSIMTextBox.place(x=1000, y=750)
    SSIMTextBox.configure(state='normal')
    SSIMTextBox.insert(tk.END, ImageSSIM)





    root.mainloop()


def month_changed(event):
    """ handle the month changed event """
    if (selected_month.get() == "Text"):
        Text_UI(root)
    elif selected_month.get() == "Image":
        Image_UI(root)


root = Tk()
root.title('PolyBius-Vigenre Encryption Analysis')
root.geometry('1050x1000')
selected_month = tk.StringVar()
month_cb = ttk.Combobox(root, textvariable=selected_month, width=3, height=5)
month_cb.pack()
month_cb.place(x=0, y=100)
month_cb['values'] = ["Text", "Image"]
month_cb.bind('<<ComboboxSelected>>', month_changed)

root.mainloop()
