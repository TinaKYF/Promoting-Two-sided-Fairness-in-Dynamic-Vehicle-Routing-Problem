import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
import pandas as pd
from k_means_constrained import KMeansConstrained
import csv

class CLUSTER:
    def __init__(self,officernum):
        self.officernum=officernum


    def cluster_hisdata(self):
        nodelist = []
        arealist = []
        nodeData = pd.read_csv('data\\2016\\full_nodes\\filter_bay_vio_data_02_08.csv')
        for i in range(len(nodeData)):
            nodelist.append([nodeData['lat'][i], nodeData['lon'][i]])
        # latlist=[]
        # lonlist=[]
        # for i in range(len(self.viodata)):
        #     for j in range(len(self.nodedata)):
        #         if self.viodata['street_marker'][i]==self.nodedata['st_marker_id'][j]:
        #             latlist.append(self.nodedata['lat'][j])
        #             lonlist.append(self.nodedata['lon'][j])
        # self.viodata['lat']=latlist
        # self.viodata['lon']=lonlist



        num_min = int(1900/ self.officernum * 0.7)#11148
        num_max = int(1900 / self.officernum * 1.3)
        coordinates = np.array(nodelist)
        clf = KMeansConstrained(
            n_clusters=self.officernum,
            size_min=num_min,
            size_max=num_max,
            random_state=0
        )
        y = clf.fit_predict(coordinates)
        # y_list = []
        # keylist = [i for i in range(self.officernum)]
        # valuelist = [0 for j in range(self.officernum)]
        # res = dict(zip(keylist, valuelist))

        # for i in y:
        #     res[i] += 1
        #     y_list.append(i)
        # nodeData['area'] = y_list

        # for i in sensordata['st_marker_id']:
        #     for j in range(len(nodeData)):
        #         if nodeData['street_marker'][j]==i:
        #             arealist.append(nodeData['area'][j])
        #             break
        # sensordata['AREA']=arealist
        # sensordata.to_csv('data\\filter_bay_sensors_vio_loc_02_08_'+str(num)+'.csv',index=0)
        centralnodelist = []
        for j in clf.cluster_centers_:
            centralnodelist.append(list(j))
        candicoolist = []
        candimarkerlist = []
        for centralnode in centralnodelist:
            mindis = 99999
            for i in range(len(nodeData)):
                nowdis = (centralnode[0] - nodeData['lat'][i]) ** 2 + (centralnode[1] - nodeData['lon'][i]) ** 2
                if nowdis < mindis:
                    mindis = nowdis
                    candicoo = [nodeData['lat'][i], nodeData['lon'][i]]
                    candimarker = nodeData['street_marker'][i]
            candicoolist.append(candicoo)
            candimarkerlist.append(candimarker)
        return (list(candimarkerlist))
