import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import minimize

a = [(0.2,6.8), (0.15,7.5), (2,10), (5,4.1), (5.5,10), (13,10), (14,8), (14.2,10), (18,5), (20,8)]

def geometric_median(L):
    x_coord = [point[0] for point in L]
    y_coord = [point[1] for point in L]

    mean = np.array([sum(x_coord)/len(x_coord),sum(y_coord)/len(y_coord)])

    def dist_func(mean):
        return sum(((np.full(len(x_coord),mean[0])-x_coord)**2+(np.full(len(x_coord),mean[1])-y_coord)**2)**(1/2))

    res = minimize(dist_func, mean)

    return res.fun, res.x

def plotter_test(L): 
    geo_med = geometric_median(L)[1]

    x_coord = [point[0] for point in L]
    y_coord = [point[1] for point in L]
    x_lines = [[geo_med[0], i] for i in x_coord] 
    y_lines = [[geo_med[1], i] for i in y_coord] 
 

    for i in range(len(L)): 
        plt.plot(x_lines[i],y_lines[i], color='magenta',
                linestyle=':')
        
    plt.plot(geo_med[0], geo_med[1], marker='o', markersize = '8', markerfacecolor='blue',
             markeredgecolor = 'blue')
    
    plt.scatter(x_coord, y_coord, c = 'green') 

    return plt.show()


def contour_plotter(L):

    x_coord = [point[0] for point in L]
    y_coord = [point[1] for point in L]

    min_xy_coord = min(x_coord + y_coord)
    max_xy_coord = max(x_coord + y_coord)

    x_list = np.linspace(min_xy_coord - 2, max_xy_coord + 2, num=100)
    y_list = np.linspace(min_xy_coord - 2, max_xy_coord + 2, num=100)

    X, Y = np.meshgrid(x_list, y_list)

# Z skal være afstanden fra det givne punkt til alle punkter -
# mean = np.array([sum(x)/len(x), sum(y)/len(y)])

    Z = np.zeros((len(x_list), len(y_list)))
    
    lx = len(x_list)
    
    ly = len(y_list)
    
    x_coordarray = np.asarray([point[0] for point in L])
    
    y_coordarray = np.asarray([point[1] for point in L])
    
    for i in range(lx):
        for j in range(ly):
            Z[j][i] = sum(((x_list[i] - x_coordarray) ** 2 + (y_list[j] - y_coordarray) ** 2)**(1/2))

    fig = plt.figure(figsize=(10,7))
    
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    
    ax = fig.add_axes([left, bottom, width, height])
    

    cp = ax.contour(X, Y, Z, 25)
    
    ax.clabel(cp, inline=True,fontsize=10)

    plotter_test(L)

#point_tracker 2.0
def left_or_right(a,b,c): 
    if (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]) < 0: 
        return True 
    else: 
        False                  
    

def bisec_lines(L):
    for point_1 in L:
        for point_2 in L:
            if point_1 != point_2:
                yield point_1, point_2 

#Assumes that no three points are on a line

# Generates two lists with points to the right and points to the left 


def two_geometric_medians(L): 

    solution = 0
    
    for p,q in bisec_lines(L):

        a_pq = [] #will contain p,q

        b = []

        for point in L: 

            if left_or_right(p, q, point):
                b.append(point)            
            else:  
                a_pq.append(point)
            
        
        a = a_pq.copy()
        a.remove(p)
        a.remove(q)

        b_pq = b.copy()
        b_pq.append(p)
        b_pq.append(q)

        if not a or not b: #Need to check again if we pop p,q
            continue 

        for a,b in [(a, b_pq),(a_pq ,b)]:
            
            geomed_a = geometric_median(a)  
            geomed_b = geometric_median(b) 

            distance = geomed_a[0] + geomed_b[0]

            if solution == 0:

                solution = (distance, geomed_a[1], geomed_b[1], a, b)

            elif distance < solution[0]:
                
                solution = (distance, geomed_a[1], geomed_b[1], a, b)
                        
    return solution


print(two_geometric_medians(a)[1:3]) #Test 4


def plot_two_medians(L):

    distance, geomed_a, geomed_b, a, b = two_geometric_medians(L)

    a_x = [i[0] for i in a]
    a_y = [i[1] for i in a]

    b_x = [i[0] for i in b]
    b_y = [i[1] for i in b]

    ab_xy = a_x + a_y + b_x + b_y

    min_ab_xy = min(ab_xy)
    max_ab_xy = max(ab_xy)

    for i in range(len(a_x)):
        plt.plot((geomed_a[0], a_x[i]),(geomed_a[1], a_y[i]), color='y',
                 linestyle=':', marker='o', markersize='8', markerfacecolor='red',
            markeredgecolor='red')

    for i in range(len(b_x)):
        plt.plot((geomed_b[0], b_x[i]),(geomed_b[1], b_y[i]), color='y',
             linestyle=':', marker='o', markersize='8', markerfacecolor='red',
             markeredgecolor='red')
    
    plt.plot((geomed_a[0], geomed_b[0]), (geomed_a[1], geomed_b[1]), linestyle=':', marker='o', markersize = '12', markerfacecolor='blue', 
                 markeredgecolor = 'blue')

    plt.xlim(min_ab_xy-2, max_ab_xy+2)

    plt.ylim(min_ab_xy-2, max_ab_xy+2)
   
    perp_bi_x = np.linspace(min_ab_xy, max_ab_xy, 100) 
    
    midpoint = ((geomed_a[0]+geomed_b[0])/2),((geomed_a[1]+geomed_b[1])/2)

    plt.plot(midpoint[0], midpoint[1], linestyle=':', marker='o', markersize = '12', markerfacecolor='green', 
                 markeredgecolor = 'green')

    if geomed_b[0]-geomed_a[0] == 0:

        plt.axvline(midpoint[0], linestyle='-', color='blue')


    elif geomed_b[1]-geomed_a[1] == 0:

        plt.axhline(midpoint[1], linestyle='-', color='blue')


    else:

        slope = (geomed_b[1]-geomed_a[1])/(geomed_b[0]-geomed_a[0])

        rec_slope = -1/slope

        b_value = midpoint[1]-midpoint[0]*rec_slope

        plt.plot(perp_bi_x, rec_slope * perp_bi_x + b_value, linestyle='-', color='green')

    return plt.show()

plot_two_medians(a) #Test 5
