import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import matplotlib.pyplot as plt
#from osgeo import gdal
import random


#spatial çözünürlük elde etme
camera_sensor_genislik=6 #mavic2zoom için 6  milimetre sensör genişliği
camera_focal_lenght=4 #mavic2zoom için 4 milimetre
ucus_yuksekligi=75  #metre olarak uçuş yüksekliği
goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği
mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak



anlik_goruntu="parcalar/adana_anlik.jpg"

#template matching
harita="haritalar/adana_harita.jpg"
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



hedef_konumlar = ((4000, 5000), (4000, 2000), (6000, 1000), (6000, 6000), (1000, 6000),
                  (2000, 4000), (7000, 4000), (3000, 7000), (5000, 7000), (7000, 2000),
                  (2000, 7000), (5000, 3000), (3000, 5000), (7500, 7500), (2500, 2500),
                  (6500, 3500), (3500, 6500), (4500, 4500), (5500, 5500), (1500, 1500))

anlik_harita= cv2.imread(anlik_goruntu,0)

dikey=2500+512
yatay=2500+512



# dikey = random.randint(0, 8704)
# yatay= random.randint(0, 8704)



img_temp=img


#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#methods =['cv2.TM_CCOEFF']


i=0  #hedef_konumlarin sirasini belirtir
while(True):    
    
    
    template=anlik_harita[dikey-512:dikey,yatay-512:yatay]
    #plt.imshow(template, cmap = "gray")
    

    hedef_konum=hedef_konumlar[i]
    
    print(template.shape)
    h,w =template.shape
    
    res= cv2.matchTemplate(img, template, cv2.TM_CCOEFF, None)
        
    print(res.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
       
    print("konum: ",max_val, max_loc)
        
        
    top_left = max_loc
    konum = max_loc
                
    bottom_right = (top_left[0] + w,top_left[1] +h)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    cv2.rectangle(img, top_left, bottom_right,(255,0,0),35)
    
    
    for j, hedef_konum_daire in enumerate(hedef_konumlar, start=0):
      x, y = hedef_konum_daire
    
      # Nokta yerine yazı yazmak için fonksiyon
      cv2.putText(img, str(j+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 150, 255), 25)
    
    
    
        #plt.figure()
        
        # plt.imshow(img)
        # plt.title("Tespit edilen Sonuç"), plt.axis("on")
        # plt.suptitle(meth)
        # plt.pause(0.0001)
    res = cv2.resize(img, dsize=(1000,1000), interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
    cv2.imshow("Resized", res)
        #cv2.destroyAllWindows()
       
    img = img_temp
        
    k = cv2.waitKey(10)     
        
    
    
    print((konum[0]-hedef_konum[0]),(konum[1]-hedef_konum[1]))
    
    adim =100
    
    if abs(konum[0]-hedef_konum[0])<100:
        
        dikey+=0
    elif konum[0]-hedef_konum[0]>0:
        yatay-=adim
        dikey+=0
    else:        
        yatay+=adim
        dikey+=0
        
    if abs(konum[1]-hedef_konum[1])<100:     
        
        dikey+=0
        
    elif konum[1]-hedef_konum[1]>0:
        yatay+=0
        dikey-=adim
    else:        
        yatay+=0
        dikey+=adim
    print("i= ",i)
            
    if abs(konum[0]-hedef_konum[0])<250 and abs(konum[1]-hedef_konum[1])<250:
        
        i=i+1 #hedef konumda bir sonrakine gider
        
        if len(hedef_konumlar)==i:
            print("hedefe ulaşıldı")
            break
        
        print("ara hedefe ulaştı\n\n\n**********")
        
        
        # adim =250
        # if cv2.waitKey(0) & 0xFF == ord('s'):
        #     yatay+=0
        #     dikey+=adim
        # elif cv2.waitKey(0) & 0xFF == ord('d'):
        #     yatay+=adim
        #     dikey+=0
        # elif cv2.waitKey(0) & 0xFF == ord('a'):
        #     yatay-=adim
        #     dikey+=0
        # elif cv2.waitKey(0) & 0xFF == ord('w'):
        #     yatay+=0
        #     dikey-=adim