import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal
import random


#spatial çözünürlük elde etme
camera_sensor_genislik=6 #mavic2zoom için 6  milimetre sensör genişliği
camera_focal_lenght=4 #mavic2zoom için 4 milimetre
ucus_yuksekligi=75  #metre olarak uçuş yüksekliği
goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği
mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak



anlik_goruntu="parcalar/bern_swistopo.jpg"

#template matching
harita="haritalar/bern_gmap.jpg"
#harita="harita_gmap.jpg"

img = cv2.imread(harita,0)
print(img.shape)

kenarx=int(img.shape[0]/512)


kx = (112 % kenarx)*512
ky = (int(112/kenarx))*512




# gdal.Warp('anlik_goruntu_warped.tif', anlik_goruntu, xRes=0.09, yRes=0.09) 
# raster = gdal.Open('anlik_goruntu_warped.tif')
# gt =raster.GetGeoTransform()

# print (gt)
# pixelSizeX = gt[1]
# pixelSizeY = -gt[5]
# print ("x = ",pixelSizeX)
# print ("y = ",pixelSizeY)


anlik_harita= cv2.imread(anlik_goruntu,0)

dikey=2500
yatay=2500

dikey = random.randint(0, 8704)
yatay= random.randint(0, 8704)

img_temp=img


#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
methods =['cv2.TM_CCOEFF']



while(True):
    
    template=anlik_harita[dikey-512:dikey,yatay-512:yatay]
    #plt.imshow(template, cmap = "gray")
    

   
    
    print(template.shape)
    h,w =template.shape
    for meth in methods:
        method  = eval(meth)    #stringleri fonksiyona çeviren fonksiyona
        
        res= cv2.matchTemplate(img, template, method, None)
        
        print(res.shape)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        print("konum: ",max_val, max_loc)
        
        
        top_left = max_loc
                
        bottom_right = (top_left[0] + w,top_left[1] +h)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        cv2.rectangle(img, top_left, bottom_right,(255,0,0),35)
        #plt.figure()
        
        # plt.imshow(img)
        # plt.title("Tespit edilen Sonuç"), plt.axis("on")
        # plt.suptitle(meth)
        # plt.pause(0.0001)
        res = cv2.resize(img, dsize=(1000,1000), interpolation=cv2.INTER_CUBIC)
        cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
        cv2.imshow("Resized", res)
        cv2.waitKey(100)
        #cv2.destroyAllWindows()
       
        img = img_temp
        
        yon = random.randint(0, 8)
        adim =random.randint(50, 150)
        if yon==0:
            yatay+=0
            dikey+=adim
        elif yon==1:
            yatay+=adim
            dikey+=0
        elif yon==2:
            yatay-=adim
            dikey+=0
        elif yon==3:
            yatay+=0
            dikey-=adim
        elif yon==4:
            yatay+=adim
            dikey+=adim
        elif yon==5:
            yatay+=adim
            dikey-=adim
        elif yon==6:
            yatay-=adim
            dikey+=adim
        elif yon==7:
            yatay-=adim
            dikey-=adim
        
        
        
    
    

    