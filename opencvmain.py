import cv2
import numpy as np
import os




# wild functions
def resizeImages():
    n_path = "tankbackup/neg/"
    p_path = "tankbackup/pos/"
    n_data = os.listdir("./"+ n_path)
    p_data = os.listdir("./"+ p_path)
    data_arr = [n_data, p_data]
    
    for data in range(0,len(data_arr)):
        if data == 0:
            for index, d in enumerate(data_arr[data]):
                scaleAndWriteImageFile(path=n_path, file=d, folder="n", name="negative", index=index)
        elif data == 1:
            for index, d in enumerate(data_arr[data]):
                scaleAndWriteImageFile(path=p_path, file=d, folder="p", name="positive", index=index)
        else:
            break
            
def scaleImage(image):
    targetW, targetH = 1024, 576
    newImg = str(image)
    img = cv2.imread(cv2.samples.findFile(newImg))
    w,h = img.shape[1::-1]

    wW, hH = targetW/w, targetH/h  
    scale = (wW + hH) /2
    
    return scale
   
def writeFileNeg():
    name = "neg"
    neg = os.listdir("./n/")
    
    file_Name = str(name) + ".txt"
    if not os.path.exists(file_Name) or os.path.exists(file_Name):
        file_object = open(file_Name, "w")
        for item in neg:
            file_object.write("n/" + item + "\n")
        file_object.close()

def scaleAndWriteImageFile(path, file, folder, name, index):
    file = path + file 
    img = cv2.imread(cv2.samples.findFile(file))
    
    scale = scaleImage(file)
    downSize = cv2.resize(img, None,fx=scale, fy=scale, 
                         interpolation=cv2.INTER_LINEAR)
    str = f'./{folder}/{name}{index}.jpg'
    print("Scaling image", str)
    cv2.imwrite(str, downSize)

def returnBigBox(boxes, size):
    for d in boxes:
        if size == d[3]:
            return d


# image
def imageCascade(imagepath):
    image = cv2.imread(imagepath) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cas = cv2.CascadeClassifier("haarcascade_russiantanks_alpha_v1_20_20.xml")

    target = cas.detectMultiScale(image=gray, minNeighbors=4, minSize=(50,50))  
    results, _ = cv2.groupRectangles(target, 2, 2)
    for coord in results:
        if len(coord) != 4:
            continue

        x,y,w,h = coord
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,230,120), 2)

    cv2.imshow("Tank detection test", image)

    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()


def imageCascadeTwo(imagepath):
    image = cv2.imread(imagepath) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cas = cv2.CascadeClassifier("haarcascade_russiantanks_alpha_v1_20_20.xml")

    target = cas.detectMultiScale(image=gray, minNeighbors=2, minSize=(80,80))  

    a = []
    for idx, d in enumerate(target):
        a.append([idx, d[3]])

    id_box = max(a, key=lambda x:x[1])
    rx, ry, rw, rh = target[id_box[0]]
    # singular big box
    roi = image[ry:ry+rh, rx:rx+rw]
    
    target2 = cas.detectMultiScale(image=roi, minNeighbors=1, minSize=(40,40))  
    results, _ = cv2.groupRectangles(target2,3,2)
    
    cv2.rectangle(image, (rx,ry), (rx+rw,ry+rh), (20,150,250), 2)

    a = []
    cv2.imshow("Tank detection test", image)

    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()



# video
def videoCascade(path, frames):
    video = cv2.VideoCapture(path)
    cas = cv2.CascadeClassifier("haarcascade_russiantanks_alpha_v1_20_20.xml") # 12 stages

    if not video.isOpened():
        print("Cannot open video")
        exit()

    while True:
        # frame by frame capture
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # search for the biggest box to find target, first cascade check
        target = cas.detectMultiScale(image=gray, minNeighbors=4, minSize=(50,50))  
        
        # take biggest box from target array, run loop inside roi [x,y,w,h]
        box_check = []
        if isinstance(target, np.ndarray):
            box_check = np.max(target[:,3])
        elif isinstance(target, tuple):
            continue
        
        bx,by,bw,bh = returnBigBox(target, size=box_check)  
        roi = frame[by:by+bh, bx:bx+bw]
        
        # second cascade check for boxes within roi
        target2 = cas.detectMultiScale(image=roi, minNeighbors=3)
        results, _ = cv2.groupRectangles(target2, 2, 2)
        #Check for objects from results, area coordinates taken from roi
        for coord in results:
            if len(coord) != 4:
                continue
            elif len(coord) == 0:
                continue

            x,y,w,h = coord
            cv2.rectangle(frame, (bx+x,by+y), (bx+x+w,by+y+h), (0,230,120), 2)

        # if frame is read, ret is True
        if not ret:
            print("Ending video..")
            break

        cv2.imshow("Detection Window", frame)

        if cv2.waitKey(frames) == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()
    

    # box_check is non-existent and it break video
    # if there are multiple bix boxes

def main():
    #videoCascade("tank.mp4",1)
    # or
    imageCascade("tank.jpg")

if __name__ == "__main__":
    main()

