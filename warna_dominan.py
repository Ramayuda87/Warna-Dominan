import numpy as np
import cv2
import matplotlib.pyplot as plt

# Fungsi untuk mendapatkan warna dominan dari patch
def get_dominant_color(image):
    data = np.reshape(image, (-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, _, centers = cv2.kmeans(data, 1, None, criteria, 10, flags)

    return centers[0]

# Fungsi untuk mengonversi warna BGR ke RGB
def bgr_to_rgb(bgr_color):
    return bgr_color[::-1]

# Fungsi untuk mengonversi warna RBG ke HEX
def rgb_to_hex(rgb_color):
    # Mengkonversi nilai RGB yang dinormalisasi ke HEX
    hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))
    return hex_color

# Fungsi untuk mengonversi RGB ke CMYK
def rgb_to_cmyk(rgb_color):
    # Normalisasi nilai RGB ke rentang 0-1
    r, g, b = rgb_color / 255.0

    k = 1 - max(r, g, b)
    if k == 1:
        c = m = y = 0
    else:
        c = (1 - r - k) / (1 - k)
        m = (1 - g - k) / (1 - k)
        y = (1 - b - k) / (1 - k)

    return np.array([c, m, y, k])

# Fungsi untuk mengonversi warna BGR ke LAB dan mendapatkan nilai L, a, b
def bgr_to_lab(bgr_color):
    bgr_color = np.divide(bgr_color, 255.0)  # Normalisasi nilai BGR
    lab_color = cv2.cvtColor(np.array([[bgr_color]], dtype=np.float32), cv2.COLOR_BGR2LAB)[0][0]
    return lab_color

# Fungsi untuk mengonversi warna LAB ke BGR untuk visualisasi
def lab_to_bgr(lab_color):
    lab_color = np.array([[lab_color]], dtype=np.float32)
    bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)[0][0]
    bgr_color = np.clip(bgr_color * 255, 0, 255).astype(np.uint8)  # Konversi kembali ke skala 0-255
    return bgr_color

# Fungsi untuk menghitung Delta E (CIE76)
def delta_e_cie76(lab1, lab2):
    delta_l = lab2[0] - lab1[0]
    delta_a = lab2[1] - lab1[1]
    delta_b = lab2[2] - lab1[2]
    return np.sqrt(delta_l ** 2 + delta_a ** 2 + delta_b ** 2)

# Fungsi untuk menghitung Delta E (CIE2000)
def delta_e_cie2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1 ** 2 + b1 ** 2)
    C2 = np.sqrt(a2 ** 2 + b2 ** 2)
    avg_C = (C1 + C2) / 2.0

    G = 0.5 * (1 - np.sqrt(avg_C ** 7 / (avg_C ** 7 + 25 ** 7)))

    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    C1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime))
    h2_prime = np.degrees(np.arctan2(b2, a2_prime))
    if h1_prime < 0:
        h1_prime += 360
    if h2_prime < 0:
        h2_prime += 360

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians((h2_prime - h1_prime) / 2))

    avg_L_prime = (L1 + L2) / 2.0
    avg_C_prime = (C1_prime + C2_prime) / 2.0

    avg_h_prime = (h1_prime + h2_prime) / 2.0
    if abs(h1_prime - h2_prime) > 180:
        avg_h_prime += 180
    if avg_h_prime >= 360:
        avg_h_prime -= 360

    T = 1 - 0.17 * np.cos(np.radians(avg_h_prime - 30)) + 0.24 * np.cos(np.radians(2 * avg_h_prime)) + 0.32 * np.cos(np.radians(3 * avg_h_prime + 6)) - 0.20 * np.cos(np.radians(4 * avg_h_prime - 63))

    delta_theta = 30 * np.exp(-((avg_h_prime - 275) / 25) ** 2)
    R_C = 2 * np.sqrt(avg_C_prime ** 7 / (avg_C_prime ** 7 + 25 ** 7))
    S_L = 1 + (0.015 * (avg_L_prime - 50) ** 2) / np.sqrt(20 + (avg_L_prime - 50) ** 2)
    S_C = 1 + 0.045 * avg_C_prime
    S_H = 1 + 0.015 * avg_C_prime * T
    R_T = -np.sin(np.radians(2 * delta_theta)) * R_C

    delta_E_2000 = np.sqrt((delta_L_prime / S_L) ** 2 + (delta_C_prime / S_C) ** 2 + (delta_H_prime / S_H) ** 2 + R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H))

    return delta_E_2000, delta_L_prime, delta_C_prime, delta_H_prime

# Fungsi untuk menghitung L*, C*, h*
def lab_to_lch(lab_color):
    L = lab_color[0]
    C = np.sqrt(lab_color[1]**2 + lab_color[2]**2)
    h = np.arctan2(lab_color[2], lab_color[1])
    h = np.degrees(h)
    if h < 0:
        h += 360
    return L, C, h

# Contoh penggunaan fungsi untuk gambar pertama (seperti sebelumnya)
img1_path = 'D:/Back Up Lenovo/Latihan/Warna/sampel/payung std.jpg'
img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
dominant_color1 = get_dominant_color(img1)
dominant_color_rgb1 = bgr_to_rgb(dominant_color1)
hex_color1 = rgb_to_hex(dominant_color_rgb1)
cmyk_color1 = rgb_to_cmyk(dominant_color_rgb1)
lab_values1 = bgr_to_lab(dominant_color1)
L1, C1, h1 = lab_to_lch(lab_values1)

print('Dominant color (BGR) (Image 1):', dominant_color1)
print('Dominant color (RBG) (Image 1):', dominant_color_rgb1)
print('Dominant color (Hex) (Image 1):', hex_color1)
print('CMYK value (Image 1):', cmyk_color1)
print('L value (Image 1):', lab_values1[0])
print('a value (Image 1):', lab_values1[1])
print('b value (Image 1):', lab_values1[2])
print('L* value (Image 1):', L1)
print('C* value (Image 1):', C1)
print('h* value (Image 1):', h1)

# Contoh untuk gambar kedua
img2_path = 'D:/Back Up Lenovo/Latihan/Warna/sampel/payung spl.jpg'
img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
dominant_color2 = get_dominant_color(img2)
dominant_color_rgb2 = bgr_to_rgb(dominant_color2)
hex_color2 = rgb_to_hex(dominant_color_rgb2)
cmyk_color2 = rgb_to_cmyk(dominant_color_rgb2)
lab_values2 = bgr_to_lab(dominant_color2)
L2, C2, h2 = lab_to_lch(lab_values2)

print('Dominant color (BGR) (Image 2):', dominant_color2)
print('Dominant color (RBG) (Image 2):', dominant_color_rgb2)
print('Dominant color (Hex) (Image 2):', hex_color2)
print('CMYK value (Image 2):', cmyk_color2)
print('L value (Image 2):', lab_values2[0])
print('a value (Image 2):', lab_values2[1])
print('b value (Image 2):', lab_values2[2])
print('L* value (Image 2):', L2)
print('C* value (Image 2):', C2)
print('h* value (Image 2):', h2)

# Hitung Delta E antara dua warna dominan
delta_e_76 = delta_e_cie76(lab_values1, lab_values2)
delta_e_2000, delta_L_prime, delta_C_prime, delta_H_prime = delta_e_cie2000(lab_values1, lab_values2)

print('Delta L:', lab_values2[0] - lab_values1[0])
print('Delta a:', lab_values2[1] - lab_values1[1])
print('Delta b:', lab_values2[2] - lab_values1[2])
print('Delta L*:', delta_L_prime)
print('Delta C*:', delta_C_prime)
print('Delta H*:', delta_H_prime)
print('Delta E (CIE76):', delta_e_76)
print('Delta E (CIE2000):', delta_e_2000)

# Visualisasi menggunakan matplotlib
warna_lab_values1 = [lab_values1[0], lab_values1[1], lab_values1[2]]
warna_bgr_for_display1 = lab_to_bgr(warna_lab_values1)

warna_lab_values2 = [lab_values2[0], lab_values2[1], lab_values2[2]]
warna_bgr_for_display2 = lab_to_bgr(warna_lab_values2)

# Buat gambar warna dengan warna yang ditentukan untuk masing-masing gambar
color_patch1 = np.ones((100, 100, 3), dtype=np.uint8) * warna_bgr_for_display1
color_patch2 = np.ones((100, 100, 3), dtype=np.uint8) * warna_bgr_for_display2

plt.figure(figsize=(12, 4))

# Plot gambar pertama
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(color_patch1, cv2.COLOR_BGR2RGB))
plt.title(f'Image 1\nLAB: L={lab_values1[0]:.2f}, a={lab_values1[1]:.2f}, b={lab_values1[2]:.2f}\nLCH: L*={L1:.2f}, C*={C1:.2f}, h*={h1:.2f}')
plt.axis('off')

# Plot informasi delta di kolom ketiga
plt.subplot(1, 3, 3)
plt.axis('off')
plt.text(0.0, 0.9, f'Δ L    : {lab_values1[0] - lab_values2[0]:.2f}', color='black', fontsize=12, ha='left', transform=plt.gca().transAxes)
plt.text(0.0, 0.8, f'Δ a    : {lab_values1[1] - lab_values2[1]:.2f}', color='black', fontsize=12, ha='left', transform=plt.gca().transAxes)
plt.text(0.0, 0.7, f'Δ b    : {lab_values1[2] - lab_values2[2]:.2f}', color='black', fontsize=12, ha='left', transform=plt.gca().transAxes)
plt.text(0.0, 0.6, f'Δ L*   : {delta_L_prime:.2f}', color='black', fontsize=12, ha='left', transform=plt.gca().transAxes)
plt.text(0.0, 0.5, f'Δ C*   : {delta_C_prime:.2f}', color='black', fontsize=12, ha='left', transform=plt.gca().transAxes)
plt.text(0.0, 0.4, f'Δ H*   : {delta_H_prime:.2f}', color='black', fontsize=12, ha='left', transform=plt.gca().transAxes)
plt.text(0.0, 0.3, f'Δ E76  : {delta_e_76:.2f}', color='black', fontsize=12, ha='left', transform=plt.gca().transAxes)
plt.text(0.0, 0.2, f'Δ E2000: {delta_e_2000:.2f}', color='black', fontsize=12, ha='left', transform=plt.gca().transAxes)

# Plot gambar kedua
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(color_patch2, cv2.COLOR_BGR2RGB))
plt.title(f'Image 2\nLAB: L={lab_values2[0]:.2f}, a={lab_values2[1]:.2f}, b={lab_values2[2]:.2f}\nLCH: L*={L2:.2f}, C*={C2:.2f}, h*={h2:.2f}')
plt.axis('off')

plt.tight_layout()
plt.show()

# Fungsi untuk menampilkan nilai di atas bar
def autolabel(rects, ax, values):
    for rect, value in zip(rects, values):
        height = rect.get_height()
        ax.annotate(f'{value:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Membuat plot untuk menampilkan perbandingan warna RGB dan CMYK
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Menampilkan nilai RGB untuk kedua gambar
x = np.arange(len(dominant_color_rgb1))
width = 0.35

rects1 = axes[0].bar(x - width/2, dominant_color_rgb1, width, label='Image 1', color=['red', 'green', 'blue'])
rects2 = axes[0].bar(x + width/2, dominant_color_rgb2, width, label='Image 2', color=['darkred', 'darkgreen', 'darkblue'])

axes[0].set_ylabel('Nilai')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['R', 'G', 'B'])
axes[0].legend()

# Menampilkan nilai CMYK untuk kedua gambar
x = np.arange(len(cmyk_color1))
width = 0.35

rects3 = axes[1].bar(x - width/2, cmyk_color1, width, label='Image 1', color=['cyan', 'magenta', 'yellow', 'black'])
rects4 = axes[1].bar(x + width/2, cmyk_color2, width, label='Image 2', color=['darkcyan', 'darkmagenta', 'gold', 'dimgray'])

axes[1].set_ylabel('Nilai')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['C', 'M', 'Y', 'K'])
axes[1].legend()

# Menambahkan nilai di atas bar
autolabel(rects1, axes[0], dominant_color_rgb1)
autolabel(rects2, axes[0], dominant_color_rgb2)
autolabel(rects3, axes[1], cmyk_color1)
autolabel(rects4, axes[1], cmyk_color2)

# Menampilkan plot
plt.tight_layout()
plt.show()