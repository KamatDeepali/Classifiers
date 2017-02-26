from numpy import *
import csv
import math
import matplotlib.pyplot as plt


data = []
csvReader = csv.reader(open('/Users/deepaliKamat/Downloads/nuts_bolts.csv', 'rb'))
var1 = 'plot1'
var2 = 'plot2'
var3 = 'plot3'
var4 = 'plot4'



def func():

    global mu_1
    global mu_2
    global mu_3
    global mu_4
    global cov_mat1
    global cov_mat2
    global cov_mat3
    global cov_mat4
    global xa,xb,y
    global g
    g = 0


    xa = []
    xb = []
    y = []
    for row in csvReader :
        r1 = row[0]
        r2 = row[1]
        r3 = row[2]
        xa.append(float(r1))
        xb.append(float(r2))
        y.append(int(r3))



    data = column_stack((xa,xb,y))

    #cost matrix
    cost = [[-0.20, 0.07, 0.07, 0.07],
            [0.07, -0.15, 0.07, 0.07],
            [0.07, 0.07, -0.05, 0.07],
            [0.03, 0.03, 0.03, 0.03]]

    #uniform cost matrix
    uni_cost = [[0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [1, 1, 1, 0]]

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0

    for i in y:
        if i == 1:
            c1 = c1+1
        elif i == 2:
            c2 = c2+1
        elif i == 3:
             c3 = c3+1
        else:
            c4 = c4+1


    p1 = float(c1)/len(y)
    p2= float(c2)/len(y)
    p3 = float(c3)/len(y)
    p4 = float(c4)/len(y)

    #class prior probabilities
    prob = [p1, p2,p3, p4]
    prob2 = [0.2, 0.4, 0.3, 0.1]



    x1 = data[data[:,2] == 1]
    xa1 = x1[:,0]
    xb1 = x1[:,1]
    x2 = data[data[:,2] == 2]
    xa2 = x2[:,0]
    xb2 = x2[:,1]
    x3 = data[data[:,2] == 3]
    xa3 = x3[:,0]
    xb3 = x3[:,1]
    x4 = data[data[:,2] == 4]
    xa4 = x4[:,0]
    xb4 = x4[:,1]


    xa1_mu = xa1.mean(axis = 0)
    xa2_mu = xa2.mean(axis = 0)
    xa3_mu = xa3.mean(axis = 0)
    xa4_mu = xa4.mean(axis = 0)
    xb1_mu = xb1.mean(axis = 0)
    xb2_mu = xb2.mean(axis = 0)
    xb3_mu = xb3.mean(axis = 0)
    xb4_mu = xb4.mean(axis = 0)

    #mean vectors
    mu_1 = [xa1_mu, xb1_mu]
    mu_2 = [xa2_mu, xb2_mu]
    mu_3 = [xa3_mu, xb3_mu]
    mu_4 = [xa4_mu, xb4_mu]


    sum1 = 0
    for i in range(len(xa1)):
        sum1 =sum1+ (xa1[i]-xa1_mu)**2


    xa1_var = sum1/((len(xa1))-1)

    sum1 = 0
    for i in range(len(xa2)):
        sum1 =sum1+ (xa2[i]-xa2_mu)**2


    xa2_var = sum1/((len(xa2))-1)

    sum1 = 0
    for i in range(len(xa3)):
        sum1 =sum1+ (xa3[i]-xa3_mu)**2


    xa3_var = sum1/((len(xa3))-1)

    sum1 = 0
    for i in range(len(xa4)):
        sum1 =sum1+ (xa4[i]-xa4_mu)**2


    xa4_var = sum1/((len(xa4))-1)



    sum2 = 0
    for i in range(len(xb1)):
        sum2 =sum2+ (xb1[i]-xb1_mu)**2


    xb1_var = sum2/((len(xa1))-1)

    sum2 = 0
    for i in range(len(xa2)):
        sum2 =sum2+ (xb2[i]-xb2_mu)**2


    xb2_var = sum2/((len(xb2))-1)

    sum2 = 0
    for i in range(len(xb3)):
        sum2 =sum2+ (xb3[i]-xb3_mu)**2


    xb3_var = sum2/((len(xb3))-1)

    sum2 = 0
    for i in range(len(xb4)):
        sum2 =sum2+ (xb4[i]-xb4_mu)**2


    xb4_var = sum2/((len(xb4))-1)




    sum3 = 0
    for i in range(len(xa1)):
            sum3 = sum3+ ((xa1[i]-xa1_mu)*(xb1[i]-xb1_mu))

    cov1 = sum3/(len(xa1)-1)

    sum3 = 0
    for i in range(len(xa2)):
            sum3 = sum3+ ((xa2[i]-xa2_mu)*(xb2[i]-xb2_mu))

    cov2 = sum3/(len(xa2)-1)

    sum3 = 0
    for i in range(len(xa3)):
            sum3 = sum3+ ((xa3[i]-xa3_mu)*(xb3[i]-xb3_mu))

    cov3 = sum3/(len(xa3)-1)

    sum3 = 0
    for i in range(len(xa4)):
            sum3 = sum3+ ((xa4[i]-xa4_mu)*(xb4[i]-xb4_mu))

    cov4 = sum3/(len(xa4)-1)


    #covariance matrices
    cov_mat1 = matrix([[xa1_var, cov1], [cov1, xb1_var, ]])


    cov_mat2 = matrix([[xa2_var, cov2] , [cov2, xb2_var]])


    cov_mat3 = matrix([[xa3_var, cov3], [cov3, xb3_var]])


    cov_mat4 = matrix([[xa4_var, cov4] , [cov4, xb4_var]])



    calcAndPlot(data,prob,cost,y,var1)
    calcAndPlot(data,prob,uni_cost,y,var2)
    calcAndPlot(data,prob2,cost,y,var3)
    calcAndPlot(data,prob2,uni_cost,y,var4)


#plot functions
def calcAndPlot(data,probList,costMat,y,var ):

    y_hat = []

    for i in data[:,(0,1)]:

        t1 = i-mu_1
        t2 =linalg.inv(cov_mat1)
        num = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat1))))
        prod = (-0.5)*dot(dot(t1.conj().transpose(),t2),t1)
        gau_0 = num* math.exp(prod)


        k1 = i-mu_2
        k2 =linalg.inv(cov_mat2)
        num2 = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat2))))
        prod2 = (-0.5)*dot(dot(k1.conj().transpose(),k2),k1)
        gau_1 = num2* math.exp(prod2)


        k1 = i-mu_3
        k2 =linalg.inv(cov_mat3)
        num2 = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat3))))
        prod2 = (-0.5)*dot(dot(k1.conj().transpose(),k2),k1)
        gau_2 = num2* math.exp(prod2)


        k1 = i-mu_4
        k2 =linalg.inv(cov_mat4)
        num2 = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat4))))
        prod2 = (-0.5)*dot(dot(k1.conj().transpose(),k2),k1)
        gau_3 = num2* math.exp(prod2)


        G = [gau_0, gau_1, gau_2, gau_3]

        prod = dot(G,costMat) * probList

        tempk = []

        for i in prod:
            tempk.append(i)

        temp = tempk.index(min(tempk))
        y_hat.append(temp+1)


    range_fx1 = linspace(-.5,1,75)
    range_fx2 = linspace(-.5,1,75)
    y_k = ones((len(range_fx1),len(range_fx2)))



    #classification
    for i in range(0, len(range_fx1)):
      for j in range(0, len(range_fx2)):
        temp = matrix([range_fx1[i],range_fx2[j]])

        t1 = temp-mu_1
        t2 =linalg.inv(cov_mat1)
        num = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat1))))
        prod = (-0.5)*dot(dot(t1,t2),t1.conj().transpose())
        gau_0 = num* math.exp(prod)


        k1 = temp-mu_2
        k2 =linalg.inv(cov_mat2)
        num2 = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat2))))
        prod2 = (-0.5)*dot(dot(k1,k2),k1.conj().transpose())
        gau_1 = num2* math.exp(prod2)


        k1 = temp-mu_3
        k2 =linalg.inv(cov_mat3)
        num2 = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat3))))
        prod2 = (-0.5)*dot(dot(k1,k2),k1.conj().transpose())
        gau_2 = num2* math.exp(prod2)


        k1 = temp-mu_4
        k2 =linalg.inv(cov_mat4)
        num2 = (1/(2*math.pi))*(1/(math.sqrt(linalg.det(cov_mat4))))
        prod2 = (-0.5)*dot(dot(k1,k2),k1.conj().transpose())
        gau_3 = num2* math.exp(prod2)


        G = [gau_0, gau_1, gau_2, gau_3]

        prod = dot(G,costMat) * probList

        vals = []
        for p in prod:
            vals.append(p)



        y_k[i][j] = vals.index(min(vals))+1


    #scatter plots for grids
    for i in range(0, len(range_fx1)):
      for j in range(0, len(range_fx2)):

          if(y_k[i][j]==1):
              plt.scatter(range_fx1[i],range_fx2[j], color='yellow', marker='.')
          elif(y_k[i][j] == 2):
            plt.scatter(range_fx1[i],range_fx2[j], color='blue', marker='.')
          elif(y_k[i][j] == 3):
            plt.scatter(range_fx1[i],range_fx2[j], color='green', marker='.')
          else:
            plt.scatter(range_fx1[i],range_fx2[j], color='orange', marker='.')


    #decision boundary plotters
    for i in range(0,len(range_fx1)-1):
        for j in range(0,len(range_fx2)-1):
           if(y_k[i][j]!=y_k[i][j+1]):
              plt.scatter(range_fx1[i],float(range_fx2[j+1]+range_fx2[j])/2, color='black', marker='.')

    for i in range(0,len(range_fx1)-1):
        for j in range(0,len(range_fx2)-1):
            if(y_k[j][i]!=y_k[j+1][i]):
               plt.scatter((range_fx1[j]+range_fx1[j+1])/2,range_fx2[i], color='black', marker='.')


    #data point markers
    for m in range(len(y)):
        if(y[m]==1):
          plt.scatter(xa[m],xb[m], color='black', marker='+')
        elif(y[m]==2):
          plt.scatter(xa[m],xb[m], color='black', marker='*')
        elif(y[m]==3):
          plt.scatter(xa[m],xb[m], facecolor='none',edgecolor='black', marker='.')
        else:
          plt.scatter(xa[m],xb[m], color='black', marker='x')

    plt.savefig(var)
    plt.show()




func()








