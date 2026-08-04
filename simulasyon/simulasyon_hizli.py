import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import matplotlib.pyplot as plt
#from osgeo import gdal


# #spatial çözünürlük elde etme
# camera_sensor_genislik=6 #mavic2zoom için 6  milimetre sensör genişliği
# camera_focal_lenght=4 #mavic2zoom için 4 milimetre
# ucus_yuksekligi=75  #metre olarak uçuş yüksekliği
# goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
# goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği
# mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
# goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak



anlik_goruntu="parcalar/adana_anlik.jpg"

#template matching
harita="haritalar/adana_harita.jpg"
#harita="harita_gmap.jpg"

ana_harita = cv2.imread(harita,0)
print(ana_harita.shape)

# kenarx=int(img.shape[0]/512)


# kx = (112 % kenarx)*512
# ky = (int(112/kenarx))*512




# gdal.Warp('anlik_goruntu_warped.tif', anlik_goruntu, xRes=0.09, yRes=0.09) 
# raster = gdal.Open('anlik_goruntu_warped.tif')
# gt =raster.GetGeoTransform()

# print (gt)
# pixelSizeX = gt[1]
# pixelSizeY = -gt[5]
# print ("x = ",pixelSizeX)
# print ("y = ",pixelSizeY)


anlik_pozisyon= cv2.imread(anlik_goruntu,0)

dikey=4000
yatay=2000

ana_harita_temp=ana_harita
ana_harita_tahmin=ana_harita

#methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# methods =['cv2.TM_CCOEFF']

x=0
y=0


i=0
while(True):
    
    if yatay<512:
        yatay=512
        continue
    if dikey<512:
        dikey=512
        continue
    
    
    template=anlik_pozisyon[dikey-512:dikey,yatay-512:yatay]   #yatay dikey yer değştirince doğru çalıştı sebebini anlayamadım
    #plt.imshow(template, cmap = "gray")
    #cv2.waitKey(0)

   

    #print(template.shape)
    h,w =template.shape
    # for meth in methods:
    #     method  = eval(meth)    #stringleri fonksiyona çeviren fonksiyona
        
        
    res= cv2.matchTemplate(ana_harita_tahmin, template, cv2.TM_CCOEFF, None)    
                
        
    print(ana_harita_tahmin.shape)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
    if(i==0):
            
        x0=max_loc[0]
        y0=max_loc[1]
        max_loc=(0,0)
        x=max_loc[0]+x0
        y=max_loc[1]+y0
    else:            
        
        x=max_loc[0]+x0-1024
        y=max_loc[1]+y0-1024
        x0=x
        y0=y
        if x0<0:
            x0=0
            x=0
        elif y0<0:
            y0=0
            y=0
        
        
    
        
    top_left = [x,y]
    
    #x=1030
         
    if x>=1024:
        ana_harita_tahmin=ana_harita[y-1024:y+1024,x-1024:x+1024]
        if y>=1024:
            ana_harita_tahmin=ana_harita[y-1024:y+1024,x-1024:x+1024]
        elif y>=0:
            ana_harita_tahmin=ana_harita[0:2048,x-1024:x+1024]
        else: 
            y=0
     
            
    elif x>=0:
        ana_harita_tahmin=ana_harita[y-1024:y+1024,0:2048]
        if y>=1024:
            ana_harita_tahmin=ana_harita[y-1024:y+1024,0:2048]
        elif y>=0:
            ana_harita_tahmin=ana_harita[0:2048,0:2048]
        else:
            y=0
    else:
        x=0
    
    
   
        
   
    print("x: ",x," y: ",y)
    
    print("konum: ",max_val, top_left)
        
        
        
    bottom_right = (top_left[0] + w,top_left[1] +h)
    img = cv2.cvtColor(ana_harita, cv2.COLOR_GRAY2BGR)
        
    cv2.rectangle(img, top_left, bottom_right,(255,0,0),35)
    #plt.figure()
        
    res = cv2.resize(img, dsize=(1000,1000), interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
    cv2.imshow("Resized", res)
    #cv2.waitKey(100)
        
    # plt.imshow(img)
    # plt.title("Tespit edilen Sonuç"), plt.axis("on")
    # plt.suptitle(meth)
    # #plt.pause(0.0001)
    ana_harita = ana_harita_temp
        
    
    i+=1
    k = cv2.waitKey(0)     
    
    
    
    
    adim=100    
   
    
    if cv2.waitKey(0) & 0xFF == ord('s'):
        yatay+=0
        dikey+=adim
    elif cv2.waitKey(0) & 0xFF == ord('d'):
        yatay+=adim
        dikey+=0
    elif cv2.waitKey(0) & 0xFF == ord('a'):
        yatay-=adim
        dikey+=0
    elif cv2.waitKey(0) & 0xFF == ord('w'):
        yatay+=0
        dikey-=adim    
        #cv2.waitKey(1000)
        
    
    

    