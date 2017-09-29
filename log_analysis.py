import ast
import matplotlib.pyplot as plt
import sys
#{"batch_index": 24000, "set": "train", "lower_bound_1": 285.15133895996894, "uidx": 23999, "epoch": 39, "lower_bound_0": -98.63850769545938, "training_error": -4134.629382287224}
def cost(a, num):
    cost_train = []
    iteration = []
    for i in range(len(a)-1):
        try:
            w = ast.literal_eval(a[i])
            if len(w) == 7:
                cost_train.append((w['lower_bound_1']))
                iteration.append(w['epoch'] )
        except:
            print 'anirudh'
    return cost_train, iteration

def get_data(file_name):
    f = open(file_name)
    qw = f.read()
    a = qw.split('\n')
    return a


file_name = sys.argv[1]
a  = get_data(file_name)

'''
file_name2 = sys.argv[2]
a2  = get_data(file_name2)

file_name3 = sys.argv[3]
a3  = get_data(file_name3)

file_name4 = sys.argv[4]
a4  = get_data(file_name4)

file_name5 = sys.argv[5]
a5  = get_data(file_name5)
'''
plt.figure(1)

if True:
    plt.figure(1)
    plt.subplot(221)

    iteration, cost_train = cost(a, 1)
    plt.plot(cost_train, iteration,  linestyle='--', marker='o', color='black')
    plt.ylabel('Train_Cost')
    plt.xlabel('Iteration!')
    plt.show()

'''

plt.subplot(223)
iteration, cost_train = cost(a, 3)
plt.plot(cost_train, iteration)
plt.ylabel('Train_Cost/Average_Target_length')
plt.xlabel('Train Time')


plt.subplot(224)
iteration, cost_train = cost(a, 4)
plt.plot(cost_train, iteration)
plt.ylabel('Train_Cost/Average_Target_length')
plt.xlabel('Train Time')


"."     point
","     pixel
"o"     circle
"v"     triangle_down
"^"     triangle_up
"<"     triangle_left
">"     triangle_right
"1"     tri_down
"2"     tri_up
"3"     tri_left
"4"     tri_right
"8"     octagon
"s"     square
"p"     pentagon
"P"     plus (filled)
"*"     star
"h"     hexagon1
"H"     hexagon2
"+"     plus
"x"     x
"X"     x (filled)
"D"     diamond
"d"     thin_diamond
"|"     vline
"_"     hline


'''

