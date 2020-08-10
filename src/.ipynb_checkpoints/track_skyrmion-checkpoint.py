import csv
import os, sys
from PIL import Image, ImageDraw


def readFile(filename):
    lines = []

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for r in reader:
            lines.append(r)
    xdata = lines[0:128]
    ydata = lines[129:257]
    zdata = lines[258:386]

    #xdata = 128 lines, each with 512 values.
    #data[x][y][component]  where c = 0 gives x , c = 1 gives y
    data = [[[0 for i in range(3)] for j in range(128)] for k in range(512)]

    for x in range(0, 512):
        for y in range(0, 128):
            data[x][y][0] = float(xdata[y][x])
            data[x][y][1] = float(ydata[y][x])
            data[x][y][2] = float(zdata[y][x])

    # print("Got " + str(len(data)) + "x" + str(len(data[0])) + "x" +
    #       str(len(data[0][0])))
    return data


def findMaximum(data):
    maximal = -1.0
    posx = 0
    posy = 0
    for x in range(0, 512):
        for y in range(0, 128):
            if d[x][y][2] > maximal:
                maximal = d[x][y][2]
                posx = x
                posy = y

    # print("Found Maximum at (" + str(posx) + "," + str(posy) + ")")
    if (posx < 511 and posx > 0 and posy < 127 and posy > 0):
        # print("Center: " + str(d[posx][posy][2]))
        # print("Up: " + str(d[posx][posy + 1][2]))
        # print("Down: " + str(d[posx][posy - 1][2]))
        # print("Left: " + str(d[posx - 1][posy][2]))
        # print("Right: " + str(d[posx + 1][posy][2]))

        weightx = float(posx)
        weighty = float(posy)
        weighty += 0.5 * (d[posx][posy + 1][2] - d[posx][posy][2])
        weighty -= 0.5 * (d[posx][posy - 1][2] - d[posx][posy][2])

        weightx += 0.5 * (d[posx + 1][posy][2] - d[posx][posy][2])
        weightx -= 0.5 * (d[posx - 1][posy][2] - d[posx][posy][2])

        weightx = round(weightx, 2)
        weighty = round(weighty, 2)
        # print("Weighted at (" + str(weightx) + "," + str(weighty) + ")")
        return ([weightx, weighty])
    else:
        return ([posx, posy])

if __name__ == "__main__":
    directory = os.getcwd() + "/" + sys.argv[1]
    print("Got Path")
    print(directory)
    os.chdir(directory)
    files = os.listdir(os.getcwd())

    # Select csv files
    selected_files = []
    ignored_files = ['m_initial.csv', 'm_relaxed.csv', 'm_relaxed_later.csv']
    for f in files:
        if (f[-4:] == ".csv" and not f in ignored_files):
            selected_files.append(f)
    print("Got " + str(len(selected_files)) + " files.")

    # Find positions
    positions = []
    count = 0
    for fn in selected_files:
        print("Reading : " + str(fn))
        d = readFile(fn)
        pos = findMaximum(d)
        positions.append([count, pos[0], pos[1]])
        count += 1

    # Select png files
    selected_files = []
    ignored_files = ['m_initial.png', 'm_relaxed.png', 'm_relaxed_later.png']
    for f in files:
        if (f[-4:] == ".png" and not f in ignored_files and not 'located' in f):
            selected_files.append(f)
    print("Got " + str(len(selected_files)) + " files.")

    # Save images tracked particles
    new_files = []
    count = 0
    for fn in selected_files:
        print("Reading : " + str(fn))
        xpos = positions[count][1]
        ypos = positions[count][2]
        newfile = fn[:-4] + "_located.png"
        img = Image.open(fn).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.line((xpos, 0, xpos, 128))
        draw.line((0, 128 - ypos, 512, 128 - ypos))
        print("Writing : " + str(newfile))
        img.save(newfile)
        new_files.append(newfile)
        count += 1

    filelist = ""
    for f in new_files:
        filelist += f + " "

    outputfile = "animation_tracking.gif"
    command = "convert -loop 0 -delay 25 -dispose previous " + filelist + " " + outputfile
    print("Making .gif")
    os.system(command)