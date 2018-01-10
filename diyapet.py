# -*- coding: utf-8 -*-
"""
Created on Tue Jan 09 18:33:10 2018

@author: Berceste
"""

#Kullanacağımız ysa modeli için.
from keras.models import Sequential
 
#YSA modelimizde katmanlar oluşturabilmek için.
from keras.layers import Dense
 
#Çıktımızı terminalden aldığımızda sonuçları renklendiriyoruz. Yanlışlar kırmızı, doğrular yeşil.
##from termcolor import cprint
from termcolor import cprint
 
#YSA matrislerle çalıştığı için numpy olmazsa olmaz.
import numpy

#Datasetimizi yüklüyoruz.
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
#numpy kütüphanesindeki loadtxt() metodumuz verilen ilk parametredeki dosya/string I delimiter(ayırıcı) ile split ederek bir matrise atamaktadır. 

#Datesetimizi ayrıştırıyoruz. X, 8 adet girdimiz. Yani 600 satır ve 8 sutun. Y ise çıkışımız 600 satır 1 sutundan oluşmaktadır.
X = dataset[:600,0:8]
Y = dataset[:600,8]


#Şimdi modelimizi oluşturup katmanlarımızı yerleştirelim. Modelimizden model adında bir nesne türettik.
model = Sequential()


#Parametrelerden 12 olanı ilk katmandaki perceptron sayısı. 
#İnput dim bizim girdi değer sayımız. Hatırlarsanız 8 değerimiz vardı. 
#Init parametremiz yapay sinir hücrelerimiz arasındaki ağırlıklarımızın rastgele verilmesi yerine belli bir algoritmaya göre vermektedir. 
#Aktivation ise o katmanda kullandığımız aktivasyon fonksiyonumuz.

model.add(Dense(20, input_dim=8, init='uniform', activation='relu'))

#İkinci katmanımızda 12 yapay sinir hücresi. 
model.add(Dense(20, init='uniform', activation='relu'))
 
#Üçüncü katmanımızda 8 yapay hücremiz var.
model.add(Dense(8, init='uniform', activation='sigmoid'))
 
#Dördüncü katmanımızda 1 yapay hücremiz var. Yani çıkışımız.
model.add(Dense(1, init='uniform', activation='sigmoid')) 
 
#Modelimizi derliyoruz.
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#İlk parametre girdilerimiz, ikinci parametre çıktığımız. 3. paramtre epoch değerimiz. 
#Yani ağırlıkların yeniden düzenleyerek çözüme en yakın değeri buluncaya kadar yapacağı deneme sayısı. 
#Batch size tek seferde alınacak veri sayısını ifade etmektedir. 
#Verbose ise herhangi bir hata oluşması durumunda göstermemeisini istedik.
model.fit(X, Y, nb_epoch=200, batch_size=10,  verbose=2)


#Modelimizin başarı yüzdesini hesaplıyoruz.
scores = model.evaluate(X, Y)


#son olarak modelimizin sonucunu ekrana basıyoruz.
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#96 adet test verimizin sadece 8 girdisini sisteme veriyoruz.
test_verisi = dataset[600:696, 0:8]
 
#Sistemin verdiğimiz değerlerden yola çıkarak kişinin diyabet hastası olup olmadığını tahmin ediyor.
predictions = model.predict(test_verisi)

dogru = 0
yanlis = 0
toplam_veri = len(dataset[600:696,8])
 
for x, y in zip(predictions, dataset[600:696,8]):    
    x = int(numpy.round(x[0]))
    if int(x) == y:
        cprint("Tahmin: "+str(x)+" - Gerçek Değer: "+str(int(y)), "white", "on_green", attrs=['bold'])
        dogru += 1
    else:
        cprint("Tahmin: "+str(x)+" - Gerçek Değer: "+str(int(y)), "white", "on_red", attrs=['bold'])
        yanlis += 1
 
print("\n", "-"*150, "\nISTATISTIK:\nToplam ", toplam_veri, " Veri içersinde;\nDoğru Bilme Sayısı: ", dogru, "\nYanlış Bilme Sayısı: ",yanlis,
      "\nBaşarı Yüzdesi: ", str(int(100*dogru/toplam_veri))+"%", sep="") 