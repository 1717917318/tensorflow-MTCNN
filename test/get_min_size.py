import math
import os

if __name__ =='__main__':

    with open('fddb_val_min_r_small.txt', 'r') as f:
        hum_res = f.readlines()

    fir = True
    min_v = 1000
    # min_h = 1000

    that_line = ""
    for line in hum_res:
        line_copy = line
        line = line[0:-1]
        line = line.split(' ')
        line = line[1:len(line)]
        for i in range(len(line) ):
            line[i] = int(line[i] )

        if fir:
            print(line)
            fir = False
        i = 0
        while( i< len(line) ) :
            x1 = line[i]
            y1 = line[i+1]
            x2 = line[i+2]
            y2 = line[i+3]
            if x2 - x1 + 1 < min_v:
                min_v = min (min_v, x2 - x1 + 1)
                that_line = line_copy
            if y2 - y1 + 1 < min_v:
                min_v = min (min_v, y2- y1 + 1)
                that_line = line_copy
            i += 4
    print("min_w,min_h: %d "%(min_v ) )
    print("that line:"+ that_line )