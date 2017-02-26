from numpy import *
import math
import matplotlib.pyplot as plt

#loading data
data = load("/Users/deepaliKamat/Downloads/data.npy")

def covk(data):
    x1 = data[:,0]  #200x1
    x2 = data[:,1]  #200x1
    y = data[:,2]   #200x1
    n = len(data)   #200


    x10 = x1[y == 0]
    x11 = x1[y == 1]
    x20 = x2[y == 0]
    x21 = x2[y == 1]

    y_0 = y[y==0]
    y_1 = y[y==1]

    #class probabilities
    prob_0 = float(len(y_0))/len(y)
    prob_1 = float(len(y_1))/len(y)

    #means of data points with respect to class
    x10_mu = x10.mean(axis = 0)
    x11_mu = x11.mean(axis = 0)
    x20_mu = x20.mean(axis = 0)
    x21_mu = x21.mean(axis = 0)

    mu_0 = [x10_mu, x20_mu]
    mu_1 = [x11_mu, x21_mu]


    # variance and covariance calculations
    sum1 = 0
    for i in range(len(x10)):
        sum1 =sum1+ (x10[i]-x10_mu)**2


    x10_var = sum1/((len(x10))-1)

    sum2 = 0
    for i in range(len(x20)):
        sum2 =sum2+ (x20[i]-x20_mu)**2


    x20_var = sum2/(len(x20)-1)

    sum1 = 0
    for i in range(len(x11)):
        sum1 =sum1+ (x11[i]-x11_mu)**2


    x11_var = sum1/(len(x11)-1)

    sum2 = 0
    for i in range(len(x21)):
        sum2 =sum2+ (x21[i]-x21_mu)**2


    x21_var = sum2/(len(x21)-1)


    sum3 = 0
    for i in range(len(x10)):
            sum3 = sum3+ ((x10[i]-x10_mu)*(x20[i]-x20_mu))

    cov0 = sum3/(len(x10)-1)

    sum3 = 0
    for i in range(len(x11)):
            sum3 = sum3+ ((x11[i]-x11_mu)*(x21[i]-x21_mu))


    cov1 = sum3/(len(x11)-1)

    #covariance matrices
    cov_mat0 = matrix([[x10_var, cov0], [cov0, x20_var]])
    cov_mat1 = matrix([[x11_var, cov1] , [cov1, x21_var]])


    y_hat = []

    # MAP calculation
    for i in data[:,(0,1)]:
        t1 = i-mu_0
        t2 =linalg.inv(cov_mat0)
        num = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat0))))
        prod = (-0.5)*dot(dot(t1.conj().transpose(),t2),t1)
        gau_0 = num* math.exp(prod)
        map_0 = gau_0*prob_0

        k1 = i-mu_1
        k2 =linalg.inv(cov_mat1)
        num2 = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat1))))
        prod2 = (-0.5)*dot(dot(k1.conj().transpose(),k2),k1)
        gau_1 = num2* math.exp(prod2)
        map_1 = gau_1*prob_1

        if map_0 > map_1 :
            y_hat.append(0)
        else:
            y_hat.append(1)


    tp = tn = fp = fn = 0
    for i in range(200):
        if (y_hat[i]==0) and (y[i]==0):
            tn=tn+1
        elif (y_hat[i]==0) and (y[i]==1):
            fp=fp+1
        elif (y_hat[i]==1) and (y[i]==0):
            fn=fn+1
        else:
            tp=tp+1



    print "\nCONFUSION MATRIX \n"
    print "\t\t\t\t\tPredicted Class"
    print "\t\t\t\t\tYes\t\t \tNo"
    print "\n(Actual Class)Yes\t"+str(tp)+"\t\t"+str(fn)
    print "\n(Actual Class)No\t"+str(fp)+"\t\t"+str(tn)

    #Accuracy
    accuracy = (float(tp+tn)/(tp+tn+fp+fn)*100)
    print "Accuracy: ",(accuracy)," percent"

    # gridpoint spread
    range_fx1 = linspace(-3,4.3,75)
    range_fx2 = linspace(-2.2,3,75)
    y_k = ones((len(range_fx1),len(range_fx2)))

    y_hat = []
    for i in range(0, len(range_fx1)):
      for j in range(0, len(range_fx2)):
        temp = matrix([range_fx1[i],range_fx2[j]])

        t1 = temp-mu_0
        t2 =linalg.inv(cov_mat0)
        num = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat0))))
        prod = (-0.5)*dot(dot(t1,t2),t1.conj().transpose())
        gau_0 = num* math.exp(prod)
        map_0 = gau_0*prob_0

        k1 = temp-mu_1
        k2 =linalg.inv(cov_mat1)
        num2 = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat1))))
        prod2 = (-0.5)*dot(dot(k1,k2),k1.conj().transpose())
        gau_1 = num2* math.exp(prod2)
        map_1 = gau_1*prob_1

        # grid plot
        if map_0 > map_1 :
            y_k[i][j]=0
            plt.scatter(range_fx1[i],range_fx2[j], color='blue', marker='.')

        else:
            y_k[i][j]=1
            plt.scatter(range_fx1[i], range_fx2[j], color='orange', marker='.')


    # classification boundary plot
    for i in range(0,len(range_fx1)-1):
        for j in range(0,len(range_fx2)-1):
           if(y_k[i][j]!=y_k[i][j+1]):
              plt.scatter(range_fx1[i],float(range_fx2[j+1]+range_fx2[j])/2, color='black', marker='.')

    for i in range(0,len(range_fx1)-1):
        for j in range(0,len(range_fx2)-1):
            if(y_k[j][i]!=y_k[j+1][i]):
               plt.scatter((range_fx1[j]+range_fx1[j+1])/2,range_fx2[i], color='black', marker='.')

    # data point plot
    Y_new1 = ["blue" if i==0 else "orange" for i in y]
    plt.savefig("a2_map.png")
    plt.title("MAP classifier")
    plt.scatter(data[:,0],data[:,1],facecolor ='none', edgecolor=Y_new1)
    plt.show()




covk(data)




