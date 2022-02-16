import sys
import logging
import os
import cv2
from utils import write_image, key_action, init_cam
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import numpy as np
from skimage.transform import resize




if __name__ == "__main__":

    # folder to write images to
    #out_folder = sys.argv[1]

    # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    logging.getLogger().setLevel(logging.INFO)
   
    # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    # Load the model
    model = models.load_model('/home/xrusa/Documents/euclidean-eukalyptus/work_in_progress/week_9/project/model.h5')

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     

            # get the predictions
            def get_pred(list_name):

                max_value = []
                max_indexes = []
                results = []

                for pred in list_name:
                    a = np.amax(pred)
                    max_value.append(a)
                    pred = list(pred)
                    max_index = pred.index(a)
                    max_indexes.append(max_index)


                for index in max_indexes:
                    if index == 0:
                        results.append('F@ck you in Italian')
                    elif index == 1:
                        results.append('what the hell are you talking about?')
                    elif index == 2:
                        results.append('i honestly have no idea')
                    else:
                        results.append('Take it to your face aka moutza')
                return results
                        
            

            # get key event
            key = key_action()
            if key == 'space':
                # write the image without overlay
                # extract the [224x224] rectangle out of it
                image = frame[y:y+width, x:x+width, :]
                image = img_to_array(image,dtype = 'uint8')
                image = resize(image, (224, 224))
                image = np.expand_dims(image, axis=0)
                #write_image(out_folder, image) 
                print(model.predict(image))
                predictions = model.predict(image)
                prediction = get_pred(predictions)
                print(prediction)

            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.info('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()
