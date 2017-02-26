from numpy import *
import math
import matplotlib.pyplot as plt

data = load("/Users/deepaliKamat/Downloads/data.npy")

#  QUESTION 2(a)
#  Function for Least squares Linear Classifier
def least_Square(data):

    X = data[:,[0,1]]
    X1 = data[:,0]
    X2 = data[:,1]
    X_new = c_[ones(len(X)),X]
    Y = data[:,2]

    #Defining confusion matrix variables
    tp=0    #True Positive
    tn=0    #True Negetive
    fp=0    #False Positive
    fn=0    #False Negetive

    #Computating the beta matrix
    X_T = X_new.conj().transpose()
    product1 = dot(X_T,X_new)
    pro_inv = linalg.inv(product1)
    product2 = dot(X_T,Y)
    beta = dot(pro_inv,product2)

    #Predicted output computation
    Y_pred = dot(X_new, beta)
    Y_hat = []
    print(Y_pred)
    for i in Y_pred:
        if i > 0.5:
            Y_hat.append(1)
        else:
            Y_hat.append(0)

    # Confusion Matrix Values
    for i in range(200):
        if (Y_hat[i]==0) and (Y[i]==0):
            tp=tp+1
        elif (Y_hat[i]==0) and (Y[i]==1):
            fp=fp+1
        elif (Y_hat[i]==1) and (Y[i]== 0):
            fn=fn+1
        else:
            tn=tn+1



    x_min = min(data[:, 0])
    x_max = max(data[:, 0])
    x2_min = min(data[:,1])
    x2_max = max(data[:,1])
    x2_1 = (0.5- beta[0]-(beta[1]*x_min))/beta[2]
    x2_2 = (0.5- beta[0]-(beta[1]*x_max))/beta[2]
    Y_new = ["blue" if i==0 else "orange" for i in Y]

    Y_new = ["blue" if i==0 else "orange" for i in Y_hat]

    range_finalX1 = linspace(x_min,x_max,20)
    print len(range_finalX1)

    range_finalX2 = linspace(x2_min,x2_max,20)
    for i in range(0,len(range_finalX1)):
        for j in range(0,len(range_finalX2)):
            y_mesh = beta[0] + beta[1]*range_finalX1[i] + beta[2]*range_finalX2[j]
            if y_mesh > 0.5:
                plt.scatter(range_finalX1[i],range_finalX2[j], color='orange', marker ='.')
            else:
                plt.scatter(range_finalX1[i],range_finalX2[j], color='blue', marker='.')



    plt.scatter(data[:,0],data[:,1],facecolor ='none', edgecolor=Y_new)
    plt.plot([x_min,x_max],[x2_1, x2_2])
    plt.title('Linear Regression of 0/1 Response')
    plt.show()

    #Accuracy
    print "\nCONFUSION MATRIX FOR LEAST SQUARES CLASSIFIER\n"
    print "\t\t\t\t\tPredicted Class"
    print "\t\t\t\t\tYes\t\t \tNo"
    print "\n(Actual Class)Yes\t"+str(tp)+"\t\t"+str(fn)
    print "\n(Actual Class)No\t"+str(fp)+"\t\t"+str(tn)

    #Accuracy
    accuracy = (float(tp+tn)/(tp+tn+fp+fn)*100)
    print "Accuracy: ",(accuracy)," percent"

    return 0



#  QUESTION 2(b)
#  Function for Nearest Neighbor Classifier
def KNN(data,k):

    x1= data[:,0]
    x2=data[:,1]
    Y = data[:, 2]
    Y_pred = []
    X_tot = data[:,[0,1]]

    #Defining confusion matrix variables
    tp=0    #True Positive
    tn=0    #True Negetive
    fp=0    #False Positive
    fn=0    #False Negetive

    #Distance calculation between Nearest Neigbors
    for i in range(len(Y)):
        dist=[]
        for j in range(len(Y)):
            #Euclidean Distance considered
            dist.append(math.sqrt(((x1[i]-x1[j])**2)+ (x2[i]-x2[j])**2))

        argDist = argsort(dist) #returns sorted indices
        arg2 = argDist[: k]     #slice k values

        sum1 = 0
        for i in arg2:
            sum1 = sum1+ Y[i]

        Y1_pre = sum1/k

        #Class allocation for sample
        if Y1_pre > 0.5:
            Y_pred.append(1)
        else:
            Y_pred.append(0)

    #Confusion Matrix calculation
    for i in range(200):
        if (Y_pred[i]==0) and (Y[i]==0):
            tp=tp+1
        elif (Y_pred[i]==0) and (Y[i]==1):
            fp=fp+1
        elif (Y_pred[i]==1) and (Y[i]==0):
            fn=fn+1
        else:
            tn=tn+1


    x_min = min(data[:, 0])
    x_max = max(data[:, 0])
    x2_min = min(data[:,1])
    x2_max = max(data[:,1])


    Y_pred2=[]

    range_fX1 = linspace(x_min,x_max,20)
    range_fX2 = linspace(x2_min,x2_max,20)
    Y2_prek = ones((len(range_fX2),len(range_fX1)))
    for i in range(0,len(range_fX1)):

        for j in range(0,len(range_fX2)):
            dist1=[]
            for l in range(0,len(data[:,0])):
                dist1.append((((range_fX1[i]-x1[l])**2)+ (range_fX2[j]-x2[l])**2))

            argDist = argsort(dist1) #returns sorted indices
            arg2 = argDist[: k]     #slice k values

            sum2 = 0
            for i2 in arg2:
                sum2 = sum2 + Y[i2]

            Y1_prek = sum2/k


                #Class allocation for sample
            if Y1_prek > 0.5:
                plt.scatter(range_fX1[i],range_fX2[j], color='orange', marker ='.')
            else:
                plt.scatter(range_fX1[i],range_fX2[j], color='blue', marker='.')

    range_fX1 = linspace(x_min,x_max,200)
    range_fX2 = linspace(x_min,x_max,200)
    Y2_prek = ones((len(range_fX1),len(range_fX2)))
    for i in range(0,len(range_fX1)):

        for j in range(0,len(range_fX2)):
            dist1=[]

            for l in range(0,len(data[:,0])):
                dist1.append((((range_fX1[i]-x1[l])**2)+ (range_fX2[j]-x2[l])**2))

            argDist = argsort(dist1) #returns sorted indices
            arg2 = argDist[: k]     #slice k values

            sum2 = 0
            for i2 in arg2:
                sum2 = sum2 + Y[i2]

            Y1_prek = sum2/k


                #Class allocation for sample
            if Y1_prek > 0.5:
                Y2_prek[i][j] = 1
            else:
                Y2_prek[i][j] = 0

    for i in range(1,len(range_fX1)-1):
        for j in range(1,len(range_fX2)-1):
            '''
            if(Y2_prek[i][j]!=Y2_prek[i+1][j]):
                plt.scatter((range_fX1[j]),(range_fX2[i]+range_fX2[i+1])/2, color='black', marker='.')
            if(Y2_prek[i][j]!=Y2_prek[i][j+1]):
                plt.scatter((range_fX1[j]+range_fX2[j+1])/2,range_fX2[i], color='black', marker='.')
            '''
            if(Y2_prek[i][j]!=Y2_prek[i][j+1]):
                plt.scatter((range_fX1[i]+range_fX2[i+1])/2,range_fX2[j], color='black', marker='.')

    for i in range(1,len(range_fX1)-1):
        for j in range(1,len(range_fX2)-1):
            '''
            if(Y2_prek[i][j]!=Y2_prek[i+1][j]):
                plt.scatter((range_fX1[j]),(range_fX2[i]+range_fX2[i+1])/2, color='black', marker='.')
            if(Y2_prek[i][j]!=Y2_prek[i][j+1]):
                plt.scatter((range_fX1[j]+range_fX2[j+1])/2,range_fX2[i], color='black', marker='.')
            '''
            if(Y2_prek[j][i]!=Y2_prek[j+1][i]):
                plt.scatter((range_fX1[j]+range_fX2[j+1])/2,range_fX2[i], color='black', marker='.')


    Y_new = ["blue" if i==0 else "orange" for i in Y]
    plt.scatter(data[:,0],data[:,1],facecolor ='none', edgecolor=Y_new)
#    plt.plot(y_final)





    if k == 1:
        plt.title('1-Nearest Neighbor Classifier')
        plt.show()
    else:
        plt.title('15-Nearest Neighbor Classifier')
        plt.show()



    #Confusion Matrix
    print "\nCONFUSION MATRIX FOR LEAST SQUARES CLASSIFIER\n"
    print "\t\t\t\t\tPredicted Class"
    print "\t\t\t\t\tYes\t\t \tNo"
    print "\n(Actual Class)Yes\t"+str(tp)+"\t\t"+str(fn)
    print "\n(Actual Class)No\t"+str(fp)+"\t\t"+str(tn)

    #Accuracy
    accuracy = (float(tp+tn)/(tp+tn+fp+fn)*100)
    print "Accuracy: ",(accuracy)," percent"

    return 0


# Least Squares Linear Classifier
least_Square(data)
# Nearest Neighbor Classifier
KNN(data,1)
 #15 - Nearest Neighbor Classifier
#KNN(data,15)




