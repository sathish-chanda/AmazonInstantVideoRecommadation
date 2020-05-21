

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import xgboost as xg
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
from numpy import *
from scipy.sparse.linalg import svds
from numpy import linalg as la
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

def opening(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def initial_data_from_dict(path):
  i = 0
  initial_data = {}
  for d in opening(path):
    initial_data[i] = d
    i += 1
  return pd.DataFrame.from_dict(initial_data, orient='index')

def more_clean(initial_data, feature, m):
    count = initial_data[feature].value_counts()
    initial_data = initial_data[initial_data[feature].isin(count[count > m].index)]
    return initial_data

def cleaning_data(initial_data,features,m):
    fil = initial_data.asin.value_counts()
    fil2 = initial_data.reviewerID.value_counts()
    initial_data['no_of_products'] = initial_data.asin.apply(lambda x: fil[x])
    initial_data['no_of_users'] = initial_data.reviewerID.apply(lambda x: fil2[x])
    while (initial_data.asin.value_counts(ascending=True)[0]) < m or  (initial_data.reviewerID.value_counts(ascending=True)[0] < m):
        initial_data = more_clean(initial_data,features[0],m)
        initial_data = more_clean(initial_data,features[1],m)
    return initial_data

def data():
    print('loading data...')
    initial_data = initial_data_from_dict('reviews_Amazon_Instant_Video_5.json.gz')
    initial_data['reviewTime'] =initial_data['reviewTime'].apply(lambda x: datetime.strptime(x, '%m %d, %Y'))
    initial_data['datetime'] = pd.to_datetime(initial_data.reviewTime, unit='s')
    raw_data = cleaning_data(initial_data, ['asin', 'reviewerID'], 2)
    raw_data['userid'] = pd.factorize(raw_data['reviewerID'])[0]
    raw_data['videoid'] = pd.factorize(raw_data['asin'])[0]
    sc = MinMaxScaler()
    raw_data['time']=sc.fit_transform(raw_data['reviewTime'].values.reshape(-1,1))
    raw_data['numberuser']=sc.fit_transform(raw_data['no_of_users'].values.reshape(-1,1))
    raw_data['numberprod']=sc.fit_transform(raw_data['no_of_products'].values.reshape(-1,1))
    raw_data['reviewTime'] =  pd.to_datetime(raw_data['reviewTime'], format='%Y-%b-%d:%H:%M:%S.%f')    
    raw_data['weekend']=raw_data['reviewTime'].dt.dayofweek>=5
    raw_data['weekend']=raw_data['weekend'].astype(int)
    First = raw_data.loc[:,['userid','videoid']]
    Second = raw_data.loc[:,['userid','videoid','time']]
    Third = raw_data.loc[:,['userid','videoid','time','numberuser','numberprod']]
    y = raw_data.overall
    # train_test split
    train_1,test_1,y_train,y_test = train_test_split(First,y,test_size=0.3,random_state=2017)
    train_2,test_2,y_train,y_test = train_test_split(Second,y,test_size=0.3,random_state=2017)
    train_3,test_3,y_train,y_test = train_test_split(Third,y,test_size=0.3,random_state=2017)
    train = np.array(train_1.join(y_train))
    test = np.array(test_1.join(y_test))
    videoid2videoid = raw_data.asin.unique()
    data_mixed = First.join(y)
    total_p = data_mixed['videoid'].unique().shape[0]
    total_u = data_mixed['userid'].unique().shape[0]
    # make the user-item uv_table
    uv_table = np.zeros([total_u,total_p])
    z = np.array(data_mixed)
    for line in z:
        u,p,s = line
        if uv_table[int(u)][int(p)] < int(s):
            uv_table[int(u)][int(p)] = int(s) 
    print('the uv_table\'s shape is:' )
    print(uv_table.shape)
    return z, total_u,total_p,videoid2videoid,train,test,uv_table,raw_data

z, total_u,total_p,videoid2videoid,train,test,uv_table,raw_data = data()


def ET_XG(a):

    raw_data['userid'] = pd.factorize(raw_data['reviewerID'])[0]
    raw_data['videoid'] = pd.factorize(raw_data['asin'])[0]
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    raw_data['time']=sc.fit_transform(raw_data['reviewTime'].values.reshape(-1,1))
    raw_data['numberuser']=sc.fit_transform(raw_data['no_of_users'].values.reshape(-1,1))
    raw_data['numberprod']=sc.fit_transform(raw_data['no_of_products'].values.reshape(-1,1))
    raw_data['reviewTime'] =  pd.to_datetime(raw_data['reviewTime'], format='%Y-%b-%d:%H:%M:%S.%f')    
    raw_data['weekend']=raw_data['reviewTime'].dt.dayofweek>=5
    raw_data['weekend']=raw_data['weekend'].astype(int)
    First = raw_data.loc[:,['userid','videoid']]
    Second = raw_data.loc[:,['userid','videoid','time']]
    Third = raw_data.loc[:,['userid','videoid','time','numberuser','numberprod']]
    y = raw_data.overall

    from sklearn.model_selection import train_test_split
    train_1,test_1,y_train,y_test = train_test_split(First,y,test_size=0.3,random_state=2017)
    train_2,test_2,y_train,y_test = train_test_split(Second,y,test_size=0.3,random_state=2017)
    train_3,test_3,y_train,y_test = train_test_split(Third,y,test_size=0.3,random_state=2017)
  #  a=ExtraTreesRegressor()
    a.fit(train_3,y_train)
    y3 = a.predict(test_3)
    sc = MinMaxScaler(feature_range=(1,5))
    c = mean_squared_error(y_train,a.predict(train_3)), mean_squared_error(y_test,sc.fit_transform(y3.reshape(-1,1)))
    b = mean_squared_error(y_test,y3)
    print('train MSE is {}, test MSE is {}'.format(c,b))

    c3 = y3>=4
    t = y_test>=4
    print('Recommendation_Accuracy:')
    print(accuracy_score(t,c3))
    c31 = y3<=1
    t1 = y_test<=1
    print('Recommendation_Accuracy:')
    print(accuracy_score(t1,c31))
    y_pred3 = []
    y_test3 = []
    for i in range(y3.shape[0]):
        if y3[i]>=4:
            y_pred3.append(1)
        elif y3[i]<4:
            y_pred3.append(0)

    for j in range(y3.shape[0]):
        if np.array(y_test)[j]>=4:
            y_test3.append(1)
        elif np.array(y_test)[j]<4:
            y_test3.append(0)

    import itertools
    import matplotlib.pyplot as plt
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    class_names = ['not recommand','recommand']
    cnf_matrix = confusion_matrix(y_test3,y_pred3)
    tn,fp,fn,tp=confusion_matrix(y_test3,y_pred3).ravel()
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1_score=2*precision*recall/(precision+recall)
    print("precision: " +str(precision)+"\nrecall: "+ str(recall)+"\n f1_score: "+str(f1_score))        
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='rf')


    plt.show()
    return a

reg1=ExtraTreesRegressor()
reg2=xg.XGBRegressor()
ET_XG(reg1)
ET_XG(reg2)

def conf_mat_plot(y_pred,y_test =test ,title=''):
    print('caculating cm..')
    y1=[]
    y2=[]
    for line in y_test:
        u,p,s = line
        u=int(u)
        p=int(p)
        s=int(s)
        y1.append(s)
        y2.append(y_pred[u,p])
    temp1 = []
    temp2 = []
    for i in range(len(y1)):
        if np.array(y1)[i] >= 4:
            temp1.append(1)
        elif np.array(y1)[i] <= 2:
            temp1.append(0)
        else:
            temp1.append(0)
        if y2[i] >= 4:
            temp2.append(1)
        elif y2[i] <= 2:
            temp2.append(0)
        else:
            temp2.append(0)
    cm = confusion_matrix(temp1,temp2)
    tn,fp,fn,tp=confusion_matrix(temp1,temp2).ravel()
    precision=tp/(tp+fp)
    recall=tp/(tp+fn)
    f1_score=2*precision*recall/(precision+recall)
    print("precision: " +str(precision)+"\nrecall: "+ str(recall)+"\n f1_score: "+str(f1_score))    
    plt.figure()
    plot_confusion_matrix(cm, classes=['not','recommand'], normalize=True,
                          title=title)
    plt.show()
    
    
def recommendation(final_table, userid,n,rawId= False):
    if userid in range(total_u):
        top_N = np.argpartition(final_table[userid],-n)[-n:]
        print('the top{} recommanded products for user {} is {}'.format(n,userid,top_N))
        if rawId == True:
            print('the real ID is {}'.format(videoid2videoid[top_N]))
    return top_N
    
from sklearn.metrics.pairwise import pairwise_distances
def cf(uv_table = uv_table,distance = 'cosine'):
    user_similarity = pairwise_distances(uv_table, metric=distance)
    item_similarity = pairwise_distances(uv_table.T, metric=distance)
    sc = MinMaxScaler(feature_range=(1,5))
    a = sc.fit_transform(np.dot(user_similarity,uv_table).dot(item_similarity))
    return a
final_table =cf()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




def caculate_mse(x):
    MSE1=[]
    MSE2=[]
    for line in train:
        u,p,s = line
        MSE1.append(s)
        MSE2.append(x[int(u),int(p)])
    MSE_in_sample = mean_squared_error(MSE1,MSE2)
    MSE3=[]
    MSE4 = []
    for line in test:
        u,p,s = line
        MSE3.append(s)
        MSE4.append(x[int(u),int(p)])
    MSE_out_sample = mean_squared_error(MSE3,MSE4)
    print('the in sample MSE = {} \nthe out sample MSE = {}'.format(MSE_in_sample,MSE_out_sample))
    return MSE_in_sample,MSE_out_sample

caculate_mse(final_table)
conf_mat_plot(final_table,title='MF')
recommendation(final_table, 10,10,rawId= True)

def svdrecommendation(uv_table = uv_table, factors= 150):
    UI = matrix(uv_table)
    user_ratings_mean=mean(UI,axis=0)
    user_ratings_mean=user_ratings_mean.reshape(1,-1)
    UI_demeaned=UI-user_ratings_mean
    U,sigma,Vt=svds(UI_demeaned,factors)
    sigma=diag(sigma)
    pred_mat=dot(dot(U,sigma),Vt) + user_ratings_mean
    sc=MinMaxScaler(feature_range = (1,5))
    pred_mat = sc.fit_transform(pred_mat)
    return pred_mat

final_table1 =svdrecommendation(factors=150)
caculate_mse(final_table1)
conf_mat_plot(final_table1,title='SVD')
recommendation(final_table1, 10,10,rawId= True)



def Matrix_Factorization(data=z, factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, plot=False):

    P = np.random.rand(total_u, factors) / 3
    Q = np.random.rand(total_p, factors) / 3
    y = []
    iteration = 0
    last_loss = 0
    while iteration < maxIter:
        loss = 0
        for i in range(data.shape[0]):
            u, p, s = data[i]
            u=int(u)
            p=int(p)
            s=int(s)
            error = s - np.dot(P[int(u)], Q[int(p)])
            loss += error ** 2 / 50
            pp = P[u]
            qq = Q[p]
            P[u] += LRate * error * qq
            Q[p] += LRate * error * pp
        iteration += 1
        y.append(loss)
        delta_loss = last_loss - loss
        print('iter = {}, loss = {}, delta_loss = {}, LR = {}'.format(iteration, loss, delta_loss, LRate))

        if abs(last_loss) > abs(loss):
            LRate *= 1.05
        else:
            LRate *= 0.5
        if abs(delta_loss) < abs(GD_end):
            print('Difference in loss is {}, so the GD stops'.format(delta_loss))
            break
        last_loss = loss
    if plot:
        plt.plot(y)
        plt.show()
    return P.dot(Q.T)

final_table_MF =Matrix_Factorization( factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, plot=1)
caculate_mse(final_table_MF)
conf_mat_plot(final_table_MF,title='Matrix_Factorization')
recommendation(final_table_MF, 10,10,rawId= True)



def Probabilistic_Matrix_Factorization(data=z, factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, regU = 0.01 ,regI = 0.01 ,plot=False):
    P = np.random.rand(total_u, factors) / 3
    Q = np.random.rand(total_p, factors) / 3
    y = []
    iteration = 0
    last_loss = 100
    while iteration < maxIter:
        loss = 0
        for i in range(data.shape[0]):
            u, p, s = data[i]
            u=int(u)
            p=int(p)
            s=int(s)
            error = s - np.dot(P[u], Q[p])
            loss += error ** 2/50
            pp = P[u]
            qq = Q[p]
            P[u] += LRate *  (error * qq - regU*pp)
            Q[p] += LRate * (error * pp - regI * qq)
        loss += regU*(P*P).sum() +regI*(Q*Q).sum()
        iteration += 1
        y.append(loss)
        delta_loss = last_loss - loss
        print('iter = {}, loss = {}, delta_loss = {}, LR = {}'.format(iteration, loss, delta_loss, LRate))
        if abs(last_loss) > abs(loss):
            LRate *= 1.05
        else:
            LRate *= 0.5

        if abs(delta_loss) < abs(GD_end):
            print('Difference in loss is {}, so the GD stops'.format(delta_loss))
            break
        last_loss = loss
    if plot:
        plt.plot(y)
        plt.show()
    return P.dot(Q.T)
final_table_PMF =Probabilistic_Matrix_Factorization( factors=30, maxIter=100, LRate=0.02, GD_end=1e-3, plot=1)
caculate_mse(final_table_PMF)
conf_mat_plot(final_table_PMF,title='Probabilistic_Matrix_Factorization')
recommendation(final_table_PMF, 10,10,rawId= True)


final_table = (final_table_MF + final_table_PMF)/2
caculate_mse(final_table)
conf_mat_plot(final_table,title='Avg_of_MF_and_PMF')
recommendation(final_table, 10,10,rawId= True)
