# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:36:05 2022

@author: U41N
"""

import sys
import os
from partition import realiza_Particao_Inicial
from Compute_delay import mean_delay
from windows import *
import gc
from scipy.stats import chi2
import time
from sklearn import preprocessing
import numpy as np
import copy
from sklearn.neighbors import NearestNeighbors
import logging
from matplotlib import pyplot as plt


def main(t_j, n_b, logger ):

        aplicaNormalizacao = True
        ##### Algorithm Parameters #####
        #Use 3000 to wait the system stabilization
        Offset = 3000
        Tamanho_janela = t_j #
        n_bins = n_b# number of bins for cons-kmeans
        cs_percentil = 0.05
        n_atributes = 4
        #DBSCAN parameter use 0.2 for gaussian datasets and 0.15 for ITS datasets
        #For heterogeneous datasets, another knn analysis need to be done and the evaluation metrics must be adapted in case of groundtruth availabe
        #For a knn analysis, Uncomment the respective lines below.
        #Epslon=0.2
        Epslon=0.15
        min_amostras = 2*n_atributes -1 # DBSCAN PARAMETER


        drift_valido = False
        logger.info("**************************************************************************")
        logger.info("*                              Parameters                                *")
        logger.info("**************************************************************************")
        logger.info("Window size ===> "+str(t_j))
        logger.info("Number of bins    ===> " + str(n_b))
        scores = []
        f1_score = 0
        ## Dataset Folder
        #Folder = "./gaussians/"
        Folder = "./ITS/"

        files = os.listdir(Folder)
        for name in files:
            start_time = time.time()
            logger.info("Used File ==> "+Folder+name)
            print("Used File ==> "+Folder+name)
            used_cols = np.arange(0, n_atributes)
            X_full = np.loadtxt(Folder+name, dtype=float,delimiter=',',usecols=np.array(used_cols,dtype=int))
            drift_anterior = 0
            scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            X = X_full[Offset:Offset+Tamanho_janela,0:n_atributes]
            scaler.fit(X_full)
            X_t = scaler.transform(X)
            k = n_atributes  # min_samples - 1
            nbrs = NearestNeighbors(n_neighbors=k).fit(X_t)
            distances, _ = nbrs.kneighbors(X_t)
            distances = np.sort(distances[:, -1])

            ################################################################################
            # Uncomment these plot lines to identify the Epslon for heterogeneous datasets #
            ################################################################################

            #plt.plot(distances)
            #plt.xlabel("Ordered points")
            #plt.ylabel(f"Distance to {k}º nearest neighbor")
            #plt.show()

            ################################################################################
            if(aplicaNormalizacao):
                 X = scaler.transform(X)
            proximo = Tamanho_janela+Offset
            noise_number = 0;

            filtered_X,centros,bin_size,n_noise_,bins,_index_clusteres = realiza_Particao_Inicial(X,Epslon,min_amostras,n_bins)


            thr = n_atributes * 7 if n_atributes * 7 < 100 else 100
            noise_number += n_noise_
            tam = len(bins)
            full_bins = np.empty(0)
            for t in range(tam):
                full_bins = np.append(full_bins,np.array(bins[t]))
            tempo_real_bins = full_bins.copy()
            drifts = []
            FP = 0
            TP = 0
            FN = 0
            X_ant = X_full[proximo,0:n_atributes]
            if (aplicaNormalizacao):
                X_ant = scaler.transform(np.reshape(X_ant,(1,-1)))

            while(proximo < np.shape(X_full)[0]):

                X_new = X_full[proximo,0:n_atributes]
                if(aplicaNormalizacao):
                    X_new = scaler.transform(np.reshape(X_new,(1,-1)))
                proximo += 1


                X,tempo_real_bins,_index_clusteres = atualizaJanela(X_new,centros,tempo_real_bins,Epslon,X,_index_clusteres)

                expectation = full_bins[0:len(tempo_real_bins)-1];
                observed =  tempo_real_bins[0:len(tempo_real_bins)-1]

                res = sum([(o-e)**2./e for o,e in zip(observed,expectation)])

                alpha = cs_percentil
                df = len(full_bins)-1
                cr=chi2.ppf(q=1-alpha,df=df)
                del expectation
                del observed

                sys.stdout.write('\r')
                prox = (int) (proximo/400)
                sys.stdout.write("%-100s %d%%" % ('.'*prox, proximo*100/X_full.shape[0]))
                sys.stdout.flush()

                if(res>cr):
                    posicao_drift = proximo
                    if (posicao_drift - drift_anterior < thr):
                        del filtered_X
                        del bin_size
                        del n_noise_
                        del bins
                        gc.collect()
                        drifts.append(proximo)
                        drift_valido = True
                    drift_anterior = proximo
                    end_pos = 1500
                    if(n_atributes > 10):
                         end_pos = 0
                    if (drift_valido and (proximo < 9500 or (proximo > 10500+end_pos  and proximo < 19500 ) or (proximo > 20500+end_pos  and proximo < 29500) or (proximo > 30500+end_pos  and proximo < 40000))):
                         FP+=1
                    if (drift_valido and ((proximo > 9500 and proximo < 10500+end_pos) or (
                                 proximo > 19500 and proximo < 20500+end_pos) or (proximo > 29500 and proximo < 30500+end_pos))):
                        TP += 1

                    filtered_X,centros,bin_size,n_noise_,bins,_index_clusteres = realiza_Particao_Inicial(X,Epslon,min_amostras,n_bins)
                    if(proximo < X_full.shape[0]):
                        X_ant = X_full[proximo, 0:n_atributes]
                        if (aplicaNormalizacao):
                            X_ant = scaler.transform(np.reshape(X_ant, (1, -1)))

                    if (drift_valido):
                           noise_number += n_noise_
                    tam = len(bins)
                    del full_bins
                    full_bins = np.empty(0)
                    for t in range(tam):
                        full_bins = np.append(full_bins,np.array(bins[t]))
                    tempo_real_bins=copy.deepcopy(full_bins)
                    drift_valido = False

            end_time = time.time()
            execution_time = end_time - start_time
            execution_time_sample = execution_time / 40000
            print("\n")

            FN = 0;
            for valor in range(9500, 10502+end_pos):
                if (valor in drifts):
                    break
                if (valor == 10501+end_pos):
                    FN += 1

            for valor in range(19500, 20502+end_pos):
                if (valor in drifts):
                    break
                if (valor == 20501+end_pos):
                    FN += 1

            for valor in range(29500, 30502+end_pos):
                if (valor in drifts):
                    break
                if (valor == 30501+end_pos):
                    FN += 1

            if(TP+FP+FN > 0):
                f1_score = (2*TP)/(2*TP+FP+FN)
            logger.info("drifts ==> " + str(drifts))
            print("drifts ==> " + str(drifts))
            if (TP + FN > 0):
                logger.info("Recall ==>" + str(TP/(TP+FN)))
                print("Recall ==>" + str(TP/(TP+FN)))
            if (TP + FP  > 0):
                logger.info("Precision ==>" + str(TP / (TP + FP)))
                print("Precision ==>" + str(TP / (TP + FP)))
            if (TP + FP + FN > 0):
                logger.info("F1-Score ==>" + str((2*TP)/(2*TP+FP+FN)))
                print("F1-Score ==>" + str((2*TP)/(2*TP+FP+FN)))
            logger.info("Mean delay ==>" + str(mean_delay(drifts)) + " samples")
            print("Mean delay ==>" + str(mean_delay(drifts)) + " samples")
            logger.info("Number of Detections ==>" + str(len(drifts)))
            print("Number of Detections ==>" + str(len(drifts)) )
            print(f"Execution Time in Seconds ===> {execution_time:.6f} Seconds")
            print(f"Execution Time per Sample ===> {execution_time_sample:.6f} Seconds")
            logger.info(f"Execution Time in Seconds ===> {execution_time:.6f} Seconds")
            logger.info(f"Execution Time per Sample ===> {execution_time_sample:.6f} Seconds")
            logger.info("\n")
            print("\n")
            scores.append(f1_score)
        logger.info("F1_SCORE Mean ===> "+str(np.mean(scores)))
        return np.mean(f1_score)


if __name__ == '__main__':


    logging.basicConfig(filename="exec.log",
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # Window Size
    window_size = 300
    # Number of Bins
    bins = 13
    logger.info("Starting for windows ===> " + str(window_size) +" e  BINS ===> "+str(bins))
    max_medias = 0;
    media_score = 0;
    max_jb = [0,0]
    media_score = main(window_size, bins, logger)

