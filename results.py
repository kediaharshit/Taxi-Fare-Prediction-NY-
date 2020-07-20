#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 12:00:08 2020

@author: hk3
"""
import matplotlib.pyplot as plt
#using 100000 rows of data

#cpu
layers1 = [9, 10, 1]
epoch1= [141, 306, 562, 729, 904, 1000]
error1= [207, 85, 101, 104, 91, 99]
itr1 = [581, 681, 785, 808, 841, 809]
rmse1 = 10.52


layers2 = [9, 10, 10, 1]
epoch2= [103, 324, 537, 677, 882, 1000]
error2= [219, 99, 104, 97, 89, 95]
itr2 = [362, 541, 638, 666, 660, 671]
rmse2 = 9.396


layers3 = [9, 10, 10, 10, 1]
epoch3= [82, 272, 518, 700, 885, 1000]
error3= [200, 88, 80, 74, 101, 82]
itr3 = [260, 422, 544, 568, 596, 586]
rmse3 = 9.871


layers4 = [9, 20, 20, 1]
epoch4= [97, 298, 493, 690, 891, 1000]
error4= [229, 88, 75, 102, 100, 97]
itr4 = [338, 501, 586, 627, 647, 626]
rmse4 = 9.377


#gpu

layers5= [9, 10, 10, 1]
epoch5= [48, 188, 276, 363, 457, 549, 694, 790, 884, 979]
error5= [230, 90, 98, 81, 99, 102, 84, 92, 89, 90]
itr5 = [6, 17, 33, 62, 111, 181, 305, 369, 404, 435]
rmse5 = 9.87



layers6= [9, 20, 20, 1]
epoch6= [43, 188, 274, 365, 460, 551, 684, 776, 866, 958]
error6= [230, 105, 110, 101, 97, 113, 98, 102, 86, 95]
itr6 = [7, 19, 36, 68, 120, 190, 298, 359, 299, 416]
rmse6 = 9.72


layers7= [9, 10, 10, 10, 1]
epoch7= [37, 155, 277, 359, 481, 562, 645, 771, 855, 980]
error7= [188, 137, 125, 75, 109, 91, 105, 80, 99, 98]
itr7 = [6, 17, 45, 83, 174, 243, 307, 366, 391, 392]
rmse7 = 9.99

#tpu

layers8= [9, 20, 20, 1]
epoch8=[10, 167, 251, 383, 474, 563, 653, 786, 875, 964]
error8=[223, 104, 79, 123, 101, 105, 117, 84,72, 74]
itr8=[37, 136, 208, 317, 374, 405,423,427, 431, 435]
rmse8=9.74

layers9= [9, 10, 10, 10, 1]
epoch9=[6, 137, 282, 368, 459, 552, 689, 780, 872, 964]
error9=[205, 100, 103, 106, 106, 90, 120, 86, 74, 99]
itr9=[21, 78, 174, 248, 323, 381, 422, 434, 440, 447]
rmse9=9.7

'''
p1 = pt.figure(1)
plt.plot(epoch1, itr1, color='green', label='hidden layers=1')
plt.plot(epoch2, itr2, color='red', label='hidden layers=2')
plt.plot(epoch3, itr3, color='blue', label='hidden layers=3')
plt.xlabel('epoch number')
plt.ylabel('iterations per second')
plt.title("iterations/sec learning at different number of hidden layers")
plt.legend()
plt.show()
plt.close()


p2 = plt.figure(2)
plt.plot(epoch1, error1, color='green', label='hidden layers=1')
plt.plot(epoch2, error2, color='red', label='hidden layers=2')
plt.plot(epoch3, error3, color='blue', label='hidden layers=3')
plt.xlabel('epoch number')
plt.ylabel('train error')
plt.title("train error at different number of hidden layers")
plt.legend()
plt.show()
plt.close()


p3=plt.figure(3)
plt.plot(epoch2, itr2, color='green', label='10 nodes in hidden layers')
plt.plot(epoch4, itr4, color='red', label='20 nodes in hidden layers')
plt.xlabel('epoch number')
plt.ylabel('iterations per second')
plt.title("iterations/sec learning at different connection desnsity")
plt.legend()
plt.show()


plt.plot(epoch6, itr6, color='green', label='GPU')
plt.plot(epoch4, itr4, color='red', label='CPU')
plt.plot(epoch8, itr8, color='blue', label= 'TPU')
plt.xlabel('epoch number')
plt.ylabel('iterations per second')
plt.title("iterations/sec learning at different hardware for 2 hidden layers each with 20 nodes")
plt.legend()
plt.show()


plt.plot(epoch7, itr7, color='green', label='GPU')
plt.plot(epoch3, itr3, color='red', label='CPU')
plt.plot(epoch9, itr9, color='blue', label= 'TPU')
plt.xlabel('epoch number')
plt.ylabel('iterations per second')
plt.title("iterations/sec learning at different hardware for 3 hidden layers with 10 nodes each")
plt.legend()
plt.show()

plt.plot(epoch2, error2, color='green', label='hidden layer nodes=10')
plt.plot(epoch4, error4, color='red', label='hidden layer nodes=20')
plt.xlabel('epoch number')
plt.ylabel('train error')
plt.title("train error at different number of hidden layer nodes")
plt.legend()
plt.show()
plt.close()
'''