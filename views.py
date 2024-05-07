from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from users.models import *
from users.forms import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from django_pandas.io import read_frame

# Create your views here.

def index(request):
    return render(request,'index.html')

def logout(request):
    return render(request, "index.html")

def userlogin(request):
    return render(request,'user/userlogin.html')

def userregister(request):
    if request.method=='POST':
        form1=userForm(request.POST)
        if form1.is_valid():
            form1.save()
            print("succesfully saved the data")
            return render(request, "user/userlogin.html")
            #return HttpResponse("registreration succesfully completed")
        else:
            print("form not valied")
            return HttpResponse("form not valied")
    else:
        form=userForm()
        return render(request,"user/userregister.html",{"form":form})


def userlogincheck(request):
    if request.method == 'POST':
        sname = request.POST.get('email')
        print(sname)
        spasswd = request.POST.get('upasswd')
        print(spasswd)
        try:
            check = userModel.objects.get(email=sname,passwd=spasswd)
            # print('usid',usid,'pswd',pswd)
            print(check)
            # request.session['name'] = check.name
            # print("name",check.name)
            status = check.status
            print('status',status)
            if status == "Activated":
                request.session['email'] = check.email
                return render(request, 'user/userpage.html')
            else:
                messages.success(request, 'user is not activated')
                return render(request, 'user/userlogin.html')
        except Exception as e:
            print('Exception is ',str(e))
            pass
        messages.success(request,'Invalid name and password')
        return render(request,'user/userlogin.html')

def randomforest(request):
    train_dataset = pd.read_csv('train_set.csv')
    X_train = train_dataset.iloc[:, [2, 3, 4, 5]].values
    y_train = train_dataset.iloc[:, 6].values

    # Importing the testing dataset
    test_dataset = pd.read_csv('test_set.csv')
    X_test = test_dataset.iloc[:, [2, 3, 4, 5]].values
    y_test = test_dataset.iloc[:, 6].values

    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    rand_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    rand_classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = rand_classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import accuracy_score
    ac1 = accuracy_score(y_test, y_pred)

    print('Accuracy =', ac1)

    #
    #
    #
    #
    #
    #
    #
    #
    #
    # from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # labelencoder_X = LabelEncoder()
    # X_train[:, 2] = labelencoder_X.fit_transform(X_train[:, 2])
    # onehotencoder = OneHotEncoder(categorical_features = [2])
    # X_train = onehotencoder.fit_transform(X_train).toarray()
    #
    # X_train[:, 3] = labelencoder_X.fit_transform(X_train[:, 3])
    # onehotencoder = OneHotEncoder(categorical_features = [3])
    # X_train = onehotencoder.fit_transform(X_train).toarray()
    #
    # X_train[:, 4] = labelencoder_X.fit_transform(X_train[:, 4])
    # onehotencoder = OneHotEncoder(categorical_features = [4])
    # X_train = onehotencoder.fit_transform(X_train).toarray()
    #
    # X_train[:, 5] = labelencoder_X.fit_transform(X_train[:, 5])
    # onehotencoder = OneHotEncoder(categorical_features = [5])
    # X_train = onehotencoder.fit_transform(X_train).toarray()
    #
    #
    # # Encoding categorical data
    # # Encoding the Independent Variable
    # labelencoder_XY = LabelEncoder()
    # X_test[:, 2] = labelencoder_XY.fit_transform(X_test[:, 2])
    # onehotencoder = OneHotEncoder(categorical_features = [2])
    # X_test = onehotencoder.fit_transform(X_test).toarray()
    #
    # X_test[:, 3] = labelencoder_XY.fit_transform(X_test[:, 3])
    # onehotencoder = OneHotEncoder(categorical_features = [3])
    # X_test = onehotencoder.fit_transform(X_test).toarray()
    #
    # X_test[:, 4] = labelencoder_XY.fit_transform(X_test[:, 4])
    # onehotencoder = OneHotEncoder(categorical_features = [4])
    # X_test = onehotencoder.fit_transform(X_test).toarray()
    #
    # X_test[:, 5] = labelencoder_XY.fit_transform(X_test[:, 5])
    # onehotencoder = OneHotEncoder(categorical_features = [5])
    # X_test = onehotencoder.fit_transform(X_test).toarray()
    #
    #
    #
    #
    # # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Random Forest Classification to the Training set
    from sklearn.ensemble import RandomForestClassifier
    rand_classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    rand_classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred1 = rand_classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import accuracy_score
    ac1 = accuracy_score(y_test, y_pred)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm1 = confusion_matrix(y_test, y_pred)

    from sklearn.metrics import f1_score
    f1_score = f1_score(y_test, y_pred, average='micro')
    print("f1_score:", f1_score)
    import matplotlib.pyplot as plt
    # print(X_test, len(X_test))
    # print(y_test, len(y_test))
    # # plt.scatter(X_test, y_test, color = 'red')
    # plt.plot(X_train, rand_classifier.predict(X_train), color='blue')
    # plt.show()
    qs={"accuracy":ac1,"f1score":f1_score}
    return render(request,'admin/randomforestscore.html',{"object":qs})




def adddata(request):
    if request.method == 'POST':
        form = storetrafficdataForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'Data Added Successfull')
            form = storetrafficdataForm()
            return render(request, 'admin/addtrafficdata.html',{'form':form})
        else:
            print("Invalid form")
    else:
        form = storetrafficdataForm()
    return render(request, 'admin/addtrafficdata.html', {'form':form})






def svm(request):
        qs = storetrafficdata.objects.all()
        data = read_frame(qs)
        data = data.fillna(data.mean())
        # data[0:label]
        data.info()
        print(data.head())
        print(data.describe())
        # print("data-label:",data.label)
        dataset = data.iloc[:, [4, 5]].values
        print("x", dataset)
        dataset1 = data.iloc[:, -1].values
        print("y", dataset1)
        print("shape", dataset.shape)
        X = dataset
        y = dataset1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train)
        # print(svclassifier.predict([0.58, 0.76]))
        y_pred = svclassifier.predict(X_test)
        m = confusion_matrix(y_test, y_pred)
        accurancy = classification_report(y_test, y_pred)
        print(m)
        print(accurancy)
        x = accurancy.split()
        print("Toctal splits ", len(x))
        dict = {

            "m": m,
            "accurancy": accurancy,
            'len0': x[0],
            'len1': x[1],
            'len2': x[2],
            'len3': x[3],
            'len4': x[4],
            'len5': x[5],
            'len6': x[6],
            'len7': x[7],
            'len8': x[8],
            'len9': x[9],
            'len10': x[10],
            'len11': x[11],
            'len12': x[12],
            'len13': x[13],
            'len14': x[14],
            'len15': x[15],
            'len16': x[16],
            'len17': x[17],
            'len18': x[18],
            'len19': x[19],
            'len20': x[20],
            'len21': x[21],
            'len22': x[22],
            'len23': x[23],
            'len24': x[24],
            'len25': x[25],
            'len26': x[26],
            'len27': x[27],
            'len28': x[28],
            'len29': x[29],
            'len30': x[30],
            'len31': x[31],
            'len32': x[32],
            'len33': x[33],
            'len34': x[34],
            'len35': x[35],
            'len36': x[36],
            'len37': x[37],
            'len38': x[38],
            'len39': x[39],
            'len40': x[40],
            'len41': x[41],
            'len42': x[42],
            'len43': x[43]

        }
        return render(request, 'admin/svm.html', dict)