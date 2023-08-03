from cmath import sqrt #imports
from re import I
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
import random
from multiprocessing import  Process
import scipy.integrate as integrate
import scipy.special as special
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import eigsh
from numpy import linalg as LA
from numpy.linalg import inv
from statistics import pvariance
import time
from scipy.linalg import fractional_matrix_power
import statistics as stat
import unittest
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QLineEdit, QPushButton,QMessageBox
from matplotlib.backends.backend_qtagg import FigureCanvas
import ast
import os

class coordinate: #class for coordinate
    def __init__(self,x,y):
        self.cord = [x,y]
        self.x = self.cord[0]
        self.y = self.cord[1]
        
    
    def Random_Process(self,SD): 
         self.cord[0] = self.cord[0]+np.random.normal(0,SD)
         self.cord[1] = self.cord[1]+np.random.normal(0,SD)
         self.x = self.cord[0]
         self.y = self.cord[1]

    def xto(self,x):
         self.x = x 
         self.cord[0] = self.x
    
    def yto(self,y):
         self.y = y 
         self.cord[1] = self.y
        
    def __repr__(self): #print coordinates as list
    # return self.__str__()
        return str(self.cord)

def Distance(n1 , n2, height , width): #compute distance between two nodes
    dx = min((min(n1.x,n2.x) + width - max(n1.x,n2.x)), abs(n1.x-n2.x)) 
    dy = min((min(n1.y,n2.y) + height - max(n1.y,n2.y)), abs(n1.y-n2.y))    
    return  np.sqrt(dx**2 +  dy**2)

class perturbed_lattice:
    def __init__(self,SD,height, width ):
        self.lattice = []
        if SD == 20121:
            for i in range(height * width):
                self.lattice[i] = coordinate(random.uniform(0,self.height),random.uniform(0,self.width))
        else:
            for i in range(height):
                for j in range(width):
                    cord = coordinate(j+1/2,i+1/2)
                    cord.Random_Process(SD)
                    if cord.x > width:
                        cord.xto(cord.x - width) #fit in torus 
                    if cord.y > height:
                        cord.yto(cord.y - height)     
                    self.lattice.append(cord)
        
    def __repr__(self): #print coordinates as list
    # return self.__str__()
        return str(self.lattice)
    
    def save(self,filename):
        file = open(filename,'w')
        file.write(str(self.lattice))
        file.close()

class unifrom_random_lattice:
    def __init__(self, height , width):
        self.lattice = []
        for i in range(height * width):
                self.lattice[i] = coordinate(random.uniform(0,self.height),random.uniform(0,self.width))
    
    def __repr__(self): #print coordinates as list
    # return self.__str__()
        return str(self.lattice)

class PLGG:
    def __init__(self,lattice,radius,height,width,type = "hard", connect_fun = -3):
        self.r = radius
        self.height = height
        self.width =width
        self.length = height*width
        self.AdjacencyMatrix = [[0]*self.length for i in range(self.length)]
        self.type = type 
        self.connect_fun = connect_fun

        if type == "hard":
            def connect(a,b):
                if Distance(a,b,self.height,self.width)+10**-6 < self.r:
                    return True
                return False
        elif type == "soft":
            def connect(a,b):
                if a.x == b.x and a.x == b.x:
                    return True
                dis = Distance(a,b,self.height,self.width)
                p = self.r*dis**self.connect_fun
                foo = random.random()
                if foo > p:
                    return False
                else:
                    return True
        
        for i in range(self.length):
            for j in range(i,self.length):
                if connect(lattice[i],lattice[j]) == True :
                    self.AdjacencyMatrix[i][j] = 1
                    self.AdjacencyMatrix[j][i] = 1                    
                else:
                    self.AdjacencyMatrix[i][j] = 0
                    self.AdjacencyMatrix[j][i] = 0
        
        self.A = np.matrix(self.AdjacencyMatrix)   #adjcany matrix 
        self.DegreeDistribution =  [sum(x) for x in self.AdjacencyMatrix]  #degree distribution
        self.D = np.diag(self.DegreeDistribution)       #degree matrix 
        self.G = nx.from_numpy_matrix(self.A)           #network x graph 

    def __repr__(self): #print coordinates as list
    # return self.__str__()
        return str(self.A)

    def Plot_Graph(self): # plot the graph
        fig, ax = plt.subplots()
        # plt.plot(self.Get_XY_List()[0],self.Get_XY_List()[1],".",markersize = 3)

        for i in range(self.length):
            plt.plot(self.lattice[i].x,self.lattice[i].y,".",color = "blue")
            for k in range(self.length):
                if self.AdjacencyMatrix[i][k] == 1:
                    plt.plot([self.lattice[i].x,self.lattice[k].x],[self.lattice[i].y,self.lattice[k].y],color = 'black',linewidth=0.5)
        
        #ax.set_aspect('equal', adjustable='box')
        
        plt.show()
    
    def largest_component(self):
        largest_cc = max(nx.connected_components(self.G), key=len)
        return len(largest_cc)

    def average_degree(self):
        res = (np.trace(self.D))/(self.length)
        return res 
    
    def variance_degree(self):
        return pvariance(self.DegreeDistribution)

    def properties(self):
        lcc = self.largest_component()
        ad = self.average_degree()
        vd = self.variance_degree()
        return [lcc,ad,vd]
    
    def save(self,filename):
        np.save(filename,self.A)

class GraphPlotter(QMainWindow):
    def __init__(self):
        super(GraphPlotter, self).__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Graph Plotter')
        self.setGeometry(100, 100, 400, 300)

        self.graph_label = QLabel('Select Graph Type:', self)
        self.graph_label.move(20, 20)

        self.graph_combo = QComboBox(self)
        self.graph_combo.addItem('soft')
        self.graph_combo.addItem('hard')
        self.graph_combo.move(150, 20)

        self.height_label = QLabel('Height:', self)
        self.height_label.move(20, 60)

        self.height_input = QLineEdit(self)
        self.height_input.move(150, 60)

        self.width_label = QLabel('Width:', self)
        self.width_label.move(20, 100)

        self.width_input = QLineEdit(self)
        self.width_input.move(150, 100)

        self.sd_label = QLabel('Standard Deviation (SD):', self)
        self.sd_label.move(20, 140)

        self.sd_input = QLineEdit(self)
        self.sd_input.move(150, 140)

        self.r_label = QLabel('R:', self)
        self.r_label.move(20, 180)

        self.r_input = QLineEdit(self)
        self.r_input.move(150, 180)

        self.tail_label = QLabel('Tail Heaviness:', self)
        self.tail_label.move(20, 220)

        self.tail_combo = QComboBox(self)
        self.tail_combo.addItem('-3')
        self.tail_combo.addItem('-6')
        self.tail_combo.addItem('-9')
        self.tail_combo.addItem('-12')
        self.tail_combo.move(150, 220)

        self.plot_button = QPushButton('Plot Graph', self)
        self.plot_button.move(150, 260)
        self.plot_button.clicked.connect(self.plot_graph)

        self.show()

    def plot_graph(self):
        graph_type = self.graph_combo.currentText()
        height = int(self.height_input.text())
        width = int(self.width_input.text())
        sd = float(self.sd_input.text())
        r = float(self.r_input.text())
        tail_heaviness = float(self.tail_combo.currentText())

        if graph_type == 'soft':
            lat = perturbed_lattice(sd,height,width)
            g = PLGG(lat.lattice,r,height,width,type = "soft",connect_fun = tail_heaviness)
            g.Plot_Graph()
            property = g.properties()
            QMessageBox.information(self, 'Graph Properties',
                                 f'Mean Degree:' + str(property[1]) + '\n'
                                 f'Variance Degree: ' + str(property[2]) + ' \n'
                                 f'Largest Component: ' + str(property[0]))
            
        elif graph_type == 'hard':
            lat = perturbed_lattice(sd,height,width)
            g = PLGG(lat.lattice,r,height,width)
            g.Plot_Graph()
            property = g.properties()
            QMessageBox.information(self, 'Graph Properties',
                                 f'Mean Degree:' + str(property[1]) + '\n'
                                 f'Variance Degree: ' + str(property[2]) + ' \n'
                                 f'Largest Component: ' + str(property[0]))
    
def application():
    app = QApplication(sys.argv)
    window = GraphPlotter()
    sys.exit(app.exec_())

# class hard_graph:
#     def __init__(self, lattice,radius,height,width):
#         self.lattice = lattice.lattice
#         self.r = radius
#         self.height = height
#         self.width =width
#         self.length = height*width
#         self.AdjacencyMatrix = [[0]*self.length for i in range(self.length)]
        
#         def connect(a,b):
#             if Distance(a,b,self.height,self.width)+10**-6 < self.r:
#                 return True
#             return False
            
#         for i in range(self.length):
#             for j in range(i,self.length):
#                 if connect(self.lattice[i],self.lattice[j]) == True :
#                     self.AdjacencyMatrix[i][j] = 1
#                     self.AdjacencyMatrix[j][i] = 1                    
#                 else:
#                     self.AdjacencyMatrix[i][j] = 0
#                     self.AdjacencyMatrix[j][i] = 0
        
#         self.A = np.matrix(self.AdjacencyMatrix)   #adjcany matrix 
#         self.DegreeDistribution =  [sum(x) for x in self.AdjacencyMatrix]  #degree distribution
#         self.D = np.diag(self.DegreeDistribution)       #degree matrix 
#         self.G = nx.from_numpy_matrix(self.A)           #network x graph 
    
#     def Plot_Graph(self): # plot the graph
#         fig, ax = plt.subplots()
#         # plt.plot(self.Get_XY_List()[0],self.Get_XY_List()[1],".",markersize = 3)

#         for i in range(self.length):
#             plt.plot(self.lattice[i].x,self.lattice[i].y,".",color = "blue")
#             for k in range(self.length):
#                 if self.AdjacencyMatrix[i][k] == 1:
#                     plt.plot([self.lattice[i].x,self.lattice[k].x],[self.lattice[i].y,self.lattice[k].y],color = 'black',linewidth=0.5)
        
#         #ax.set_aspect('equal', adjustable='box')
        
#         plt.show()

# class soft_graph:
#     def __init__(self,lattice,radius,height,width,connect_fun):
#         self.lattice = lattice.lattice
#         self.r = radius
#         self.height = height
#         self.width =width
#         self.length = height*width
#         self.AdjacencyMatrix = [[0]*self.length for i in range(self.length)]

#         def connect(a,b,connect_fun):
#             if a.x == b.x and a.x == b.x:
#                 return True
        
#             dis = Distance(a,b,self.height,self.width)
#             p = self.r*dis**(connect_fun)
#             foo = random.random()
#             if foo > p:
#                 return False
#             else:
#                 return True
        
#         for i in range(self.length):
#             for j in range(i,self.length):
#                 if connect(self.lattice[i],self.lattice[j],connect_fun) == True :
#                     self.AdjacencyMatrix[i][j] = 1
#                     self.AdjacencyMatrix[j][i] = 1                    
#                 else:
#                     self.AdjacencyMatrix[i][j] = 0
#                     self.AdjacencyMatrix[j][i] = 0

#         self.A = np.matrix(self.AdjacencyMatrix)   #adjcany matrix 
#         self.DegreeDistribution =  [sum(x) for x in self.AdjacencyMatrix]  #degree distribution
#         self.D = np.diag(self.DegreeDistribution)       #degree matrix 
#         self.G = nx.from_numpy_matrix(self.A)           #network x graph 

class CoordinateTest(unittest.TestCase): #Unit test for coordiante and distance
    def test1(self):
        a = coordinate(1,1)
        b = coordinate(2,2)
        dis = Distance(a,b,10,10)
        self.assertEqual(dis, sqrt(2))
    
    def test2(self):
        a = coordinate(1,1)
        b = coordinate(9,9)
        dis = Distance(a,b,10,10)
        self.assertEqual(dis, 2*sqrt(2))
    
    def test3(self):
        a = coordinate(1,1)
        b = coordinate(1,9)
        dis = Distance(a,b,10,10)
        self.assertEqual(dis, 2)
    
    def test4(self):
        a = coordinate(1,1)
        b = coordinate(1,9)
        dis = Distance(a,b,10,10)
        self.assertEqual(dis, 2)
    
    def test5(self):
        a = coordinate(3,6)
        b = coordinate(1,9)
        dis = Distance(a,b,10,10)
        self.assertEqual(dis, sqrt(13))

def lattices_samples(sd,h,w,num,p):
    size = str(h)+"times"+str(w)
    for i in range(num):
        a = perturbed_lattice(sd,h,w)
        path = p +size+"/"+str(sd)+"/"+str(i)+".txt"
        try: 
            a.save(path)
        except:
            os.makedirs(p +size+"/"+str(sd)+"/")
            a.save(path)
    
    time.sleep(60)

def multi_lattices_samples(path,h,w,num):
    '''
    generate lattice samples locate at /data/home/ah20121/sample/lattice/
    '''
    process_list = []
    SD_list = [0.1,0.25,0.5,0.75,1.0]
    for i in range(5): 
        p = Process(target= lattices_samples,args=(SD_list[i],h,w,num,path,))
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    print('lattices are now generated!')

def read_lattice(file_path):
    try:
        # Open the file in read mode
        with open(file_path, "r") as file:                       
            # Read the content of the file as a string
            file_content = file.read()

            # Safely evaluate the string as a Python expression to get the list of lists
            list_of_lists = ast.literal_eval(file_content)
            res = []
            for i in range(len(list_of_lists)):
                res.append(coordinate(list_of_lists[i][0],list_of_lists[i][1]))

            return res
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except ValueError:
        print("Error: The file content is not a valid list of lists.")

def graph_samples(path,sd,h,w):
    '''
    path: where to read lattice
    '''
    file_list = os.listdir(path)
    size = str(h)+"times"+str(w)
    num = 0
    for file_name in file_list:    
        # 构建完整的文件路径
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            lat = read_lattice(file_path)
            for i in range(41):
                hard_graph = PLGG(lat,0.6+i*0.02,h,w)
                hard_file = "E:/Grr Project/hard_graph/"+size+"/sd"+str(sd)+"/"+str(0.6+i*0.02)+"/"
                file_name = hard_file+"/"+str(num)+".npy"
                try:                   
                    hard_graph.save(file_name)
                except:
                    os.makedirs(hard_file)
                    hard_graph.save(file_name)
                soft_graph = PLGG(lat,0.02+i*0.01,h,w,"soft",-3)
                soft_file = "E:/Grr Project/soft_graph/"+size+"/sd"+str(sd)+"/"+str(0.02+i*0.01)+"/"
                file_name = soft_file+str(num)+".npy"
                try:                   
                    soft_graph.save(file_name)
                except:
                    os.makedirs(soft_file)
                    soft_graph.save(file_name)
        num += 1
    time.sleep(60)

def multi_graph_samples(path,h,w):
    process_list = []
    SD_list = [0.1,0.25,0.5,0.75,1.0]
    size = str(h)+"times"+str(w)
    for i in range(5): 
        location  = path + size + "/"+str(SD_list[i])+"/"
        p = Process(target= graph_samples,args=(location,SD_list[i],h,w,))
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()


    print('graphs are now generated!')

def generate_samples(path,num,h,w):
    multi_lattices_samples(path,h,w,num)
    multi_graph_samples(path,h,w,)

def read_graph(path):
    loaded_data = np.load(path)
    return loaded_data

def graphs_properties_from_a_file(path,h,w):
    file_list = os.listdir(path)
    mean_degree = 0
    var_degree = 0
    largest_cc = 0
    num =len(file_list)
    for file_name in file_list:
        a = read_graph(path + file_name)
        G = nx.from_numpy_matrix(a)
        degree_sequence = [d for n, d in G.degree()]
        mean_degree = (np.trace(a))/(h*w)
        var_degree = pvariance(degree_sequence)
        largest_cc = len(max(nx.connected_components(G), key=len))/(h*w)        
    return [mean_degree/num,var_degree/num,largest_cc/num]


def graph_analysis(path,h,w):
    size = str(h)+"times"+str(w) + "/"
    p_soft = path+"soft_graph/" +size
    p_hard = path+"hard_graph/" + size
    SD_list = [0.1,0.25,0.5,0.75,1.0]
    mean_soft = []
    mean_hard = []
    var_soft = []
    var_hard = []
    lcc_soft = []
    lcc_hard = []
    for i in range(len(SD_list)):
        mean_temp_soft = []
        var_temp_soft = []
        lcc_temp_soft = []
        mean_temp_hard = []
        var_temp_hard = []
        lcc_temp_hard = []
        p_soft_temp = p_soft + "sd"+str(SD_list[i])+"/"
        p_hard_temp = p_hard + "sd"+str(SD_list[i])+"/"
        for i in range(41):
            p_soft_read = p_soft_temp + str(0.02+i*0.01)+"/"
            p_hard_read = p_hard_temp + str(0.6+i*0.02)+"/"

            soft_graph_list = graphs_properties_from_a_file(p_soft_read,10,10)
            hard_graph_list = graphs_properties_from_a_file(p_hard_read,10,10)
            
            mean_temp_soft.append(soft_graph_list[0])
            var_temp_soft.append(soft_graph_list[1])
            lcc_temp_soft.append(soft_graph_list[2])

            mean_temp_hard.append(hard_graph_list[0])
            var_temp_hard.append(hard_graph_list[1])
            lcc_temp_hard.append(hard_graph_list[2])
        
        mean_soft.append(mean_temp_soft)   
        mean_hard.append(mean_temp_hard)  
        var_soft.append(var_temp_soft)   
        var_hard.append(var_temp_hard)   
        lcc_soft.append(lcc_temp_soft)   
        lcc_hard.append(lcc_temp_hard)   
    
    return[mean_soft,var_soft,lcc_soft,mean_hard,var_hard,lcc_hard]






    
    








if __name__ == '__main__':
    #unittest.main()
    #a = perturbed_lattice(0.25,10,10)
    #application()
    #a.save("E:/Grr Project/lattices/001.txt")
    #b = PLGG(a,1,10,10)
    #b.save("E:/Grr Project/adj_matrix/001.txt")
    #print(read_lattice("E:/Grr Project/lattices/001.txt"))
    #multi_lattices_samples()
    # graph_samples("E:/GrrProject/lattices/0.1",0.1,5,5)
    # g = read_graph("E:/Grr_Project/hard_graph/5times5/sd0.1/0.6/0.npy")
    # print(g)
    # print(type(g))

    # generate_samples()

    #multi_lattices_samples(6,6,10)
    #multi_graph_samples("E:/Grr Project/lattices/",6,6)
    #generate_samples("E:/Grr Project/lattices/",10,10,10)
    print(graph_analysis("E:/Grr Project/",10,10))



