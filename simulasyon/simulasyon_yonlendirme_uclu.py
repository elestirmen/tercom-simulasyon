import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

import cv2
import matplotlib.pyplot as plt
#from osgeo import gdal
import random


# spatial cozunurluk elde etme
camera_sensor_genislik = 6
camera_focal_lenght = 4
ucus_yuksekligi = 75
goruntu_piksel_genisligi = 4000
goruntu_piksel_yuksekligi = 3000
mekansal_cozunurluk = (
    camera_sensor_genislik * ucus_yuksekligi * 100
) / (camera_focal_lenght * goruntu_piksel_genisligi)
goruntunun_gercek_uzunlugu = (
    mekansal_cozunurluk * goruntu_piksel_genisligi
) / 100


# iki kare arasinda kesismeyi bulur
def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return ()
    return (x, y, w, h)


# anlik_goruntu="parcalar/adana_anlik.jpg"
# harita="haritalar/adana_harita.jpg"

anlik_goruntu = "parcalar/urgup_facemap_level_18.tif"
harita = "haritalar/urgup_gmap.jpg"

img = cv2.imread(harita, 0)
print(img.shape)

kenarx = int(img.shape[0] / 512)

kx = (112 % kenarx) * 512
ky = (int(112 / kenarx)) * 512


# gdal.Warp('anlik_goruntu_warped.tif', anlik_goruntu, xRes=0.09, yRes=0.09)
# raster = gdal.Open('anlik_goruntu_warped.tif')
# gt =raster.GetGeoTransform()
#
# print (gt)
# pixelSizeX = gt[1]
# pixelSizeY = -gt[5]
# print ("x = ",pixelSizeX)
# print ("y = ",pixelSizeY)


anlik_harita = cv2.imread(anlik_goruntu, 0)

dikey = 2500
yatay = 2500

dikey = random.randint(0, 8704)
yatay = random.randint(0, 8704)

konum = (0, 0)
kare = ()

img_temp = img


#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
methods = ["cv2.TM_CCOEFF"]

fark = 170

while True:
    template1 = anlik_harita[dikey - 512 - fark : dikey - fark, yatay - 512 - fark : yatay - fark]
    template2 = anlik_harita[dikey - 512 : dikey, yatay - 512 : yatay]
    template3 = anlik_harita[dikey - 512 + fark : dikey + fark, yatay - 512 + fark : yatay + fark]

    print(template1.shape)
    h, w = template1.shape
    for meth in methods:
        method = eval(meth)

        res1 = cv2.matchTemplate(img, template1, method, None)
        res2 = cv2.matchTemplate(img, template2, method, None)
        res3 = cv2.matchTemplate(img, template3, method, None)

        print(res1.shape)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
        min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)

        print("konum_1: ", max_val1, max_loc1)
        print("konum_2: ", max_val2, max_loc2)
        print("konum_3: ", max_val3, max_loc3)

        top_left1 = max_loc1
        top_left2 = max_loc2
        top_left3 = max_loc3

        bottom_right1 = (top_left1[0] + w, top_left1[1] + h)
        bottom_right2 = (top_left2[0] + w, top_left2[1] + h)
        bottom_right3 = (top_left3[0] + w, top_left3[1] + h)

        global_x = max_loc2[0] + int(w / 2)
        global_y = max_loc2[1] + int(h / 2)

        a = (top_left1[0], top_left1[1], w, h)
        b = (top_left2[0], top_left2[1], w, h)
        c = (top_left3[0], top_left3[1], w, h)

        # konum bulmak icin kesisimler ve kesisim karelerinin koordinatlari bulunuyor
        kesisim_ab = intersection(a, b)
        kesisim_bc = intersection(b, c)
        kesisim_ac = intersection(a, c)

        if kesisim_ab != () and kesisim_bc != ():
            kesisim_abc = intersection(kesisim_ab, kesisim_bc)
            kare = (kesisim_abc[0], kesisim_abc[1], int(kesisim_abc[2]), int(kesisim_abc[3]))
            print("konum: ", kare)
        elif kesisim_ab != ():
            kare = (kesisim_ab[0], kesisim_ab[1], int(kesisim_ab[2]), int(kesisim_ab[3]))
            print("konum: ", kare)
        elif kesisim_bc != ():
            konum = (kesisim_bc[0], kesisim_bc[1], int(kesisim_bc[2]), int(kesisim_bc[3]))
            print("konum: ", kare)
        else:
            print("kesisim yok")

        konum = (kare[0] + int(kare[2] / 2), kare[1] + int(kare[3] / 2))

        centerOfCircle = konum

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.rectangle(img, top_left1, bottom_right1, (0, 0, 255), 25)
        cv2.rectangle(img, top_left2, bottom_right2, (0, 255, 0), 25)
        cv2.rectangle(img, top_left3, bottom_right3, (255, 0, 0), 25)
        radius = 10
        cv2.circle(img, centerOfCircle, radius, (0, 255, 255), 25)

        res = cv2.resize(img, dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)
        cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
        cv2.imshow("Resized", res)

        img = img_temp

        k = cv2.waitKey(0)

        adim = 250
        if cv2.waitKey(0) & 0xFF == ord("s"):
            yatay += 0
            dikey += adim
        elif cv2.waitKey(0) & 0xFF == ord("d"):
            yatay += adim
            dikey += 0
        elif cv2.waitKey(0) & 0xFF == ord("a"):
            yatay -= adim
            dikey += 0
        elif cv2.waitKey(0) & 0xFF == ord("w"):
            yatay += 0
            dikey -= adim
