import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import matplotlib.pyplot as plt
#from osgeo import gdal
import numpy as np
import pandas as pd

"""
#spatial çözünürlük elde etme
camera_sensor_genislik=6 #mavic2zoom için 6  milimetre sensör genişliği
camera_focal_lenght=4 #mavic2zoom için 4 milimetre
ucus_yuksekligi=75  #metre olarak uçuş yüksekliği
goruntu_piksel_genisligi = 4000 #pipksel olarak resmin genişliği
goruntu_piksel_yuksekligi = 3000 #pipksel olarak resmin genişliği
mekansal_cozunurluk = (camera_sensor_genislik*ucus_yuksekligi*100)/(camera_focal_lenght*goruntu_piksel_genisligi)  #mekansal çözünürlük cantimeter/pixel olarak
goruntunun_gercek_uzunlugu=(mekansal_cozunurluk*goruntu_piksel_genisligi)/100 #metre olarak

"""

def dosyaya_yaz(model_name,epochs,sonuclar_dogru,sonuclar_yanlis):    
    
    model_name="sonuclar_"+model_name
    sonuclar_dosya = open(model_name+"txt", "w")
    sonuclar = np.vstack((epochs,sonuclar_dogru, sonuclar_yanlis)).T
    print(sonuclar)
    
    df = pd.DataFrame(sonuclar, columns = ['epochs','dogru_tahmin','yanlis_tahmin'])
    
    sonuclar_dosya.write(str(df))
    
    df.to_csv(model_name+"csv", index=False)
    

DATADIR = r"parcalar"

DATADIR_harita = r"haritalar"

sonuclar_dogru = np.array([])
sonuclar_yanlis = np.array([])
epochs = np.array([])


from natsort import natsorted   #dosyaları doğru sıralamak için eklendi

model_klasoru =os.listdir(DATADIR)

model_name = model_klasoru[0][0:9]
model_name += model_klasoru[0][21:-2]

harita_klasoru=os.listdir(DATADIR_harita)

for k, ana_harita in enumerate(harita_klasoru):
     ana_harita = os.path.join(DATADIR_harita,ana_harita)
    
    #harita="harita_gmap.jpg"
     harita = cv2.imread(ana_harita,0)
     print(harita.shape)



     for i, model in enumerate(model_klasoru):
        if k==i:
            
            path = os.path.join(DATADIR,model)
            
            liste = natsorted(os.listdir(path))
            
            konum_dogru=0
            konum_yanlis=0
            for j, img in enumerate(liste):
                anlik_goruntu =os.path.join(path,img)
                print(ana_harita+"\n"+anlik_goruntu)
            
               
                
                sira = j
                kenarx=int(harita.shape[0]/512)
                kx = (sira % kenarx)*512
                ky = (int(sira/kenarx))*512
                
                #anlik_goruntu="anlik112.jpg"
                
                #template matching
               
                
                
                
                # gdal.Warp('anlik_goruntu_warped.tif', anlik_goruntu, xRes=0.09, yRes=0.09) 
                # raster = gdal.Open('anlik_goruntu_warped.tif')
                # gt =raster.GetGeoTransform()
                
                # print (gt)
                # pixelSizeX = gt[1]
                # pixelSizeY = -gt[5]
                # print ("x = ",pixelSizeX)
                # print ("y = ",pixelSizeY)
                
                
                template= cv2.imread(anlik_goruntu,0)
                
                # Cropping an image to 512 X 512
        
                cropped_image = template[16:528, 16:528]
                
                template = cropped_image
                
                
                #plt.figure()
                #plt.imshow(template, cmap = "gray")
                
                #print(template.shape)
                h,w =template.shape
                
                
                #methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
                #           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
                
                methods = [cv2.TM_CCOEFF]
                for method in methods:
                    res = cv2.matchTemplate(harita, template, method, None, template)
                    print(res.shape)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    print("skor = __", round(max_val/1000000, 2), "__ konum = ", max_loc)
                    top_left = max_loc
                            
                    bottom_right = (top_left[0] + w,top_left[1] + h)
                    
                    
                    konum=""
                    if abs(kx-top_left[0]) < 512 and abs(ky-top_left[1])<512:
                        print("konum dogru")
                        konum="dogru"
                        konum_dogru+=1 
                    else:
                        print("yanlis kounm")
                        konum_yanlis+=1
                        konum="yanlis"
                    #harita = cv2.cvtColor(harita, cv2.COLOR_GRAY2BGR)
                    # cv2.rectangle(harita, top_left, bottom_right,(255,0,0),35)
                    # plt.figure()
                    # plt.subplot(121), plt.imshow(res, cmap = "gray")
                    # plt.title("Eşleşen Sonuç"), plt.axis("on")
                    # plt.subplot(122), plt.imshow(harita)
                    # plt.title("Tespit edilen Sonuç"), plt.axis("on")
                    # plt.suptitle(meth+"__"+konum)
                    #harita = cv2.imread(ana_harita,0)
                
                
            sonuclar_dogru = np.append(sonuclar_dogru, konum_dogru)
            sonuclar_yanlis = np.append(sonuclar_yanlis, konum_yanlis)
            epochs = np.append(epochs,(i+1)*100)
            dosyaya_yaz(model_name,epochs,sonuclar_dogru,sonuclar_yanlis)      
     
     
     
     

    
#dosyaya_yaz(sonuclar_dogru,sonuclar_yanlis)
