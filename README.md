# Genel Bakış

6 farklı makine öğrenmesi algoritması kullanılarak sınıflandırma yapılmıştır. Sınıflandırmalar 5 kat çapraz doğrulama (5-fold Cross Validation) ile yapılmıştır. Kullanılan algoritmalar SVM, kNN, Naive Bayes, Logistic Regression, Random Forest, Linear Regression şeklindedir. Yapılan sınıflandırmanın hedefi %81 tahmin oranını aşmaktır. SVM algoritması %92,75 ile en yüksek ortalama doğruluk oranını verirken Random Forest algoritması %64 ortalama doğruluk oranı ile en düşük ortalamaya sahiptir. 
Sınıflandırma yapmak için kullanılan veri 40.000 satır 14 sütundan oluşmaktadır. Her bir satırda ayrı bir kullanıcıya ait bilgiler bulunmaktadır. Sütunlarda meslek, eğitim durumu, son görüşme tarihi ve görüşmenin süresi, bakiye gibi bilgiler bulunmaktadır. 

# Kod

Kod Python dilinde yazılmıştır. Bölümler (section), işlemlerin ve sonuçların daha anlaşılabilmesi nedeniyle oluşturulmuştur. *Veri* değişkeni içerisine atılan veri işlem yapılmadan önce incelenmektedir. 

    1	veri.info()

x (girişler) ve y (etiketlere) değişkenlerine ayrılan veri nümerik hale getirilmektedir. Örnek olarak iki sütuna uygulanan kod gösterilmektedir.

    1	x.loan= [1 if each == "yes" else 0 for each in x.loan]
    2	y=[1 if each=="yes" else 0 for each in y]

Verinin, makine öğrenmesi algoritmalarında kullanmak üzere çapraz doğrulama için, önişlemi yapıldı. SVM algoritması *svm* değişkenine atanıp cv (cross validation) değeri 5 olarak belirlenip x ve y üzerine uygulanmaktadır. 

    1	# Support Vector Machine
    2	svm=SVC()
    3	k=5
    4	svm_result = cross_val_score(svm,x,y,cv=k)
    5	print('CV Değeri: ',svm_result)
    6	print('CV Ortalamsı: ',np.sum(svm_result)/k)

Program çıktısı aşağıdaki gibi gelmektedir.

    1	CV Değeri:  [0.927625 0.92775  0.928625 0.925875 0.927625]
    2	CV Ortalamsı:  0.9275

Benzer şekilde diğer algoritmalar uygulanmış ve sonuçlar aşağıdaki alınmıştır. Naive Bayes algoritması:

    1	CV Değeri:  [0.939375 0.936875 0.917375 0.277125 0.924625]
    2	CV Ortalamsı:  0.799075

Logistic Regression:

    1	CV Değeri:  [0.93625 0.93    0.93525 0.91775 0.926  ]
    2	CV Ortalamsı:  0.9290499999999999
 
 kNN:

    1	CV Değeri:  [0.9315   0.933    0.92475  0.916375 0.921125]
    2	CV Ortalamsı:  0.9253499999999999

Random Forest:

    1	CV Değeri:  [0.9285   0.7      0.834    0.095875 0.644875]
    2	CV Ortalamsı:  0.6406499999999999

Linear Regression:

    1	CV Değeri:  [0.939375 0.936875 0.917375 0.277125 0.924625]
    2	CV Ortalamsı:  0.799075

Çıkan sonuçlar doğrultusunda SVM, Logistic Regression ve kNN algoritmaları sırasıyla %92.75, %92.90, %92.53 tahmin ortalamaları ile istenen sonuca ulaşmıştır. 

Sunulan kampanyanın kabul durumunun verilen özelliklerle bağlantısı korelasyon ile bulunmaktadır. Bunun için *corr* metodu kullanılmaktadır. Bu metodu kullanmak için *veri* değişkeni içerisinde bulunan bütün veriler nümerik hale getirilmektedir. Örnek kod satırı:

    1	x.default= [1 if each == "yes" else 0 for each in x.default]

Korelasyon sonucunda y sütununun (kampanya kabul durumu) en fazla duration (son görüşmenin saniye cinsinden süresi ) sütunu ile ilişkili olduğu görülmektedir. Yapılan görüşmelerin uzun tutulması kampanyanın kabul edilmesi ihtimalini diğer özelliklere göre daha çok mümkün kılmaktadır.

![enter image description here](https://i.hizliresim.com/fyJcHO.png)
