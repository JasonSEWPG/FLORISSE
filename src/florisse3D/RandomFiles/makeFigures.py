import matplotlib
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":

    NROWS = 6
    #Open optimal points file
    filename = "XYZ%stest.txt"%(NROWS)

    file = open(filename)
    xin = np.loadtxt(file)
    n = len(xin)
    turbineX = np.zeros(n)
    turbineY = np.zeros(n)
    turbineZ = np.zeros(n)
    for i in range(n):
        turbineX[i] = xin[i][0]
        turbineY[i] = xin[i][1]
        turbineZ[i] = xin[i][2]

    z1 = turbineZ[0]
    z2 = turbineZ[n-1]

    #plotting depending on which turbines are the tall ones
    big = 25
    small = 15
    if z1 > z2:
        z1size = big
        z2size = small
        z1color = '.b'
        z2color = '.r'
        z1label = "Optimized Tall Turbines"
        z2label = "Optimized Short Turbines"
    elif z2 > z1:
        z1size = small
        z2size = big
        z1color = '.r'
        z2color = '.b'
        z1label = "Optimized Short Turbines"
        z2label = "Optimized Tall Turbines"
    else:
        z1size = small
        z2size = small
        z1color = '.r'
        z2color = '.r'
        z1label = ""
        z2label = ""

    inverted = "no"
    
    nTurbs = len(turbineX)
    
    startTurbineZ = np.ones(nTurbs)*min(turbineZ)
    
    z1 = turbineZ[0]
    nTurbsZ1 = 0
    for i in range(nTurbs):
        if turbineZ[i] == z1:
            nTurbsZ1 += 1

    nRows = int(np.sqrt(nTurbs))
    spacing = 3    # turbine grid spacing in diameters

    # Set up position arrays
    rotor_diameter = 126.4 
    points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
    xpoints, ypoints = np.meshgrid(points, points)
    
    startTurbineX = np.ndarray.flatten(xpoints)
    startTurbineY = np.ndarray.flatten(ypoints)
    x1 = np.array([])
    x2 = np.array([])
    y1 = np.array([])
    y2 = np.array([])

    for i in range(nTurbs):
        if i%2 == 0:
            x1 = np.append(x1, startTurbineX[i])
            y1 = np.append(y1, startTurbineY[i])
        else:
            x2 = np.append(x2, startTurbineX[i])
            y2 = np.append(y2, startTurbineY[i])

    if inverted == "yes":
        startTurbineX = np.hstack([x2,x1])
        startTurbineY = np.hstack([y2,y1])
    else:
        startTurbineX = np.hstack([x1,x2])
        startTurbineY = np.hstack([y1,y2])


    xbounds = [min(turbineX), min(turbineX), max(turbineX), max(turbineX), min(turbineX)]
    ybounds = [min(turbineY), max(turbineY), max(turbineY), min(turbineY), min(turbineX)]


    plt.figure(1, figsize=(2*nRows,1.75*nRows))
    
    
    plt.plot(turbineX[0],turbineY[0],z1color,ms=z1size,label=z1label)
    for i in range(1, n):
        if turbineZ[i]==z1:
            plt.plot(turbineX[i],turbineY[i],z1color,ms=z1size)
        if turbineZ[i]==z2:
            plt.plot(turbineX[i],turbineY[i],z2color,ms=z2size)
    plt.plot(turbineX[i],turbineY[i],z2color,ms=z2size,label=z2label)

    plt.plot(xbounds, ybounds, ':k')
    plt.plot((turbineX, startTurbineX),(turbineY,startTurbineY), '--k')

    plt.plot(startTurbineX,startTurbineY,'ok', label='Original')
    
    plt.title('Optimization of Multi Hub Height %s x %s Grid'%(nRows,nRows), y=1.05)
    plt.xlabel('Turbine X Position (m)')
    plt.ylabel('Turbine Y Position (m)')
    plt.legend(bbox_to_anchor=(1.1, 1.06))
    plt.axis([200, max(turbineX)+350, min(turbineY)-250, max(turbineY)+350])

    z1 = round(z1,3)
    z2 = round(z2,3)
    print z1
    print z2
    right = max(turbineX)-470
    bottom = min(turbineY)-200
    if z2 < z1:
        plt.text(right, bottom,'Short Turbine Height: %s m\nTall Turbine Height: %s m'%(z2,z1), bbox={'alpha':1.0,'facecolor':'white'}, fontsize=15)
    elif z1 > z2:
        plt.text(right, bottom,'Short Turbine Height: %s m\nTall Turbine Height: %s m'%(z1,z2), bbox={'alpha':1.0,'facecolor':'white'}, fontsize=15)

    filename = "XY%stest.txt"%(NROWS)

    file = open(filename)
    xin = np.loadtxt(file)
    n = len(xin)
    turbineX = np.zeros(n)
    turbineY = np.zeros(n)
    turbineZ = np.zeros(n)
    for i in range(n):
        turbineX[i] = xin[i][0]
        turbineY[i] = xin[i][1]
        turbineZ[i] = xin[i][2]


    z1 = turbineZ[0]
    z2 = turbineZ[n-1]

    big = 30
    small = 20
    if z1 > z2:
        z1size = big
        z2size = small
        z1color = '.b'
        z2color = '.r'
        z1label = "Tall Turbines"
        z2label = "Short Turbines"
    elif z2 > z1:
        z1size = small
        z2size = big
        z1color = '.r'
        z2color = '.b'
        z1label = "Short Turbines"
        z2label = "Tall Turbines"
    else:
        z1size = small
        z2size = small
        z1color = '.r'
        z2color = '.r'
        z1label = "Short Turbines"
        z2label = "Tall Turbines"

    xbounds = [min(turbineX), min(turbineX), max(turbineX), max(turbineX), min(turbineX)]
    ybounds = [min(turbineY), max(turbineY), max(turbineY), min(turbineY), min(turbineX)]

    plt.figure(2, figsize=(2*nRows,1.75*nRows))
    plt.plot(startTurbineX, startTurbineY, 'ok', label='Original')
    plt.plot(turbineX, turbineY, 'or', label='Optimized')
    plt.plot(xbounds, ybounds, ':k')
    for i in range(0, nTurbs):
        plt.plot([startTurbineX[i], turbineX[i]], [startTurbineY[i], turbineY[i]], '--k')
    plt.legend()
    plt.title('Optimization of Uniform Hub Height %s x %s Grid'%(nRows, nRows), y=1.05)
    plt.xlabel('Turbine X Position (m)')
    plt.ylabel('Turbine Y Position (m)')
    plt.legend(bbox_to_anchor=(1.14, 1.14))
    

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.axis([200, max(turbineX)+300, 200, max(turbineY)+300])
    

    font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 15}
    matplotlib.rc('font', **font)

    plt.show()

