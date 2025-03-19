import matplotlib.pyplot as plt
import numpy as np
import time
import os

def main():


    speed = 0.006
    number_before_drift = 9500
    drift_size = 1000
    number_after_drift = 9000
    insert_noise = False
    n_atributes = 300
    noise_level = 0.005
    noise_percentil = 0.01
    center = np.random.rand(1,n_atributes)
    for it in range(n_atributes):
        center[0,it]=center[0,it]*np.random.rand(1)[0]*20

    Xlist = []
    X1 = np.random.normal(loc=center, scale=1, size=[(np.floor(number_before_drift)), n_atributes])
    X1_noise = np.zeros([number_before_drift,n_atributes])
    for i in range(number_before_drift):
        if(i%(np.floor(number_before_drift/(number_before_drift*noise_percentil))) == 0):
            X1_noise[i] += np.random.normal(loc=0, scale=noise_level, size=n_atributes)
    if(insert_noise):
        X1 = X1 + X1_noise

    X2_noise = np.zeros([drift_size, n_atributes])
    sentido1 = np.random.rand(1,n_atributes)
    sentido1[sentido1 > 0.5] = 1;
    sentido1[sentido1 <= 0.5] = -1;

    for i in range(drift_size):

        Xlist.append(np.random.normal(loc=center+i*speed*sentido1, scale=1, size=[(np.floor(1)), n_atributes]))
        if (i % (np.floor(number_before_drift/(number_before_drift*noise_percentil))) == 0):
            X2_noise[i] += np.random.normal(loc=0, scale=noise_level, size=n_atributes)

    X2 = np.asarray(Xlist)
    X2 = X2.reshape(drift_size,-1)

    if (insert_noise):
        X2 = X2 + X2_noise

    X3 = np.random.normal(loc=center + drift_size * speed*sentido1, scale=1, size=[(np.floor(number_after_drift)), n_atributes])
    X3_noise = np.zeros([number_after_drift, n_atributes])
    for i in range(number_after_drift):
        if (i % (np.floor(number_before_drift/(number_before_drift*noise_percentil))) == 0):
            X3_noise[i] += np.random.normal(loc=0, scale=noise_level, size=n_atributes)
    if (insert_noise):
        X3 = X3 + X3_noise



    #second part insertion

    Xlist = []
    X4_noise = np.zeros([drift_size, n_atributes])
    sentido2 = np.random.rand(1, n_atributes)
    sentido2[sentido1 > 0.5] = 1;
    sentido2[sentido1 <= 0.5] = -1;
    for i in range(drift_size):

        Xlist.append(np.random.normal(loc=center + drift_size * speed * sentido1 + i * speed * sentido2, scale=1, size=[(np.floor(1)), n_atributes]))
        if (i % (np.floor(number_before_drift/(number_before_drift*noise_percentil))) == 0):
            X4_noise[i] += np.random.normal(loc=0, scale=noise_level, size=n_atributes)

    X4 = np.asarray(Xlist)
    X4 = X4.reshape(drift_size, -1)

    if (insert_noise):
        X4 = X4 + X4_noise

    X5 = np.random.normal(loc=center + drift_size * speed * sentido1 + drift_size*speed*sentido2, scale=1,
                          size=[(np.floor(number_after_drift)), n_atributes])
    X5_noise = np.zeros([number_after_drift, n_atributes])
    for i in range(number_after_drift):
        if (i % (np.floor(number_before_drift/(number_before_drift*noise_percentil))) == 0):
            X5_noise[i] += np.random.normal(loc=0, scale=noise_level, size=n_atributes)
    if (insert_noise):
        X5 = X5 + X5_noise

    Xlist = []
    X6_noise = np.zeros([drift_size, n_atributes])

    sentido3 = np.random.rand(1, n_atributes)
    sentido3[sentido3 > 0.5] = 1;
    sentido3[sentido3 <= 0.5] = -1;


    for i in range(drift_size):

        Xlist.append(np.random.normal(loc=center +  drift_size * speed * sentido1 + drift_size * speed * sentido2  + i * speed * sentido3, scale=1,
                                      size=[(np.floor(1)), n_atributes]))
        if (i % (np.floor(number_before_drift/(number_before_drift*noise_percentil))) == 0):
            X6_noise[i] += np.random.normal(loc=0, scale=noise_level, size=n_atributes)

    X6 = np.asarray(Xlist)
    X6 = X6.reshape(drift_size, -1)

    if (insert_noise):
        X6 = X6 + X6_noise

    X7 = np.random.normal(loc=center + drift_size * speed * sentido1 + drift_size * speed * sentido2 + drift_size * speed * sentido3, scale=1,
                          size=[(np.floor(number_after_drift+500)), n_atributes])
    X7_noise = np.zeros([number_after_drift+500, n_atributes])
    for i in range(number_after_drift+500):
        if (i % (np.floor(number_before_drift/(number_before_drift*noise_percentil))) == 0):
            X7_noise[i] += np.random.normal(loc=0, scale=noise_level, size=n_atributes)
    if (insert_noise):
        X7 = X7 + X7_noise


    X_full = np.concatenate([X1, X2, X3, X4, X5, X6, X7])



    fig = plt.figure()


    plt.scatter(X_full[:,0],X_full[:,1])
    plt.xlabel('$x_1$',fontsize=15)
    plt.ylabel('$x_2$',fontsize=15)
    plt.title("Spatial Arrangement of Samples",fontsize=15)
    plt.show()

    x_l = np.zeros(1350)
    for i in range(1350):
        x_l[i] = i+9350

    plt.figure()
    plt.plot(x_l,X_full[9350:10700,0],label = '$x_1$')
    plt.plot(x_l,X_full[9350:10700,1],label = '$x_2$')
    plt.legend()
    plt.title("Zoom of Drift Region",fontsize=15)
    plt.show()

    plt.figure()
    plt.plot(X_full[:, 0],label = '$x_1$')
    plt.plot(X_full[:, 1],label = '$x_2$')
    plt.legend()
    plt.xlabel('Samples',fontsize=15)
    plt.title("Time Evolution",fontsize=15)
    plt.show()

    np.savetxt('./data_03_l15_s06_noise_150f.csv', X_full, delimiter=',')


if __name__ == '__main__':
    main()
