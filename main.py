import cv2
import pygame
import numpy as np
from pygame.locals import *

class TrainingCollector(object):

    def __init__(self):
        # Server socket connection needs to be initialized

        # end of socket connection part
        # implement OpenCV VideoCapture here.
        self.cap = cv2.VideoCapture("http://217.197.157.7:7070/axis-cgi/mjpg/video.cgi?resolution=320x240")
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        # We have 5 possible directions to evaluate
        # So we can implement 5x5 matrix to hold each direction diagonally
        self.directions = np.zeros((5,5))
        for i in range(5):
            self.directions[i,i] = 1

        # PyGame will help us to detect key event.
        pygame.init()
        pygame.display.set_mode((1,1), pygame.NOFRAME)
        self.key_set = set()
        self.ready_to_go = True
        self.start()

    def start(self):

        print("Training started..")
        # My input image has 320px width and 240px height
        # because of this, I need to create reshaped array 320x240x3 = 230400 (3 for rgb)
        image_samples = np.zeros((1, 230400))

        # labels for every image sample.
        image_labels = np.zeros((1,5))

        frame_count = 1
        saved_frame_count = 0

        while self.ready_to_go:
            read, img = self.cap.read()

            if read:
                # Do some image process here to get most applicable image.
                # At least, get region of interest

                # end of image process


                # save image
                cv2.imwrite("data/images/{:>05}.jpg".format(frame_count), img)

                # show the image
                cv2.imshow("Captured", img)

                # We need to reshape our result image to one row array
                # to make it processable for neural network
                # in my case, img is 320x240 so I can reshape matrix as [1, 230400]
                reshaped_array = img.reshape(1, 230400).astype(np.float32)

                frame_count += 1


                # Now, we have the image matrix,
                # so we need to get input by the help of pygame

                for event in pygame.event.get():
                    # if some key pressed
                    if event.type == KEYDOWN:
                        keys = pygame.key.get_pressed()
                        # if we pressed w, a and k same time
                        if keys[pygame.K_w] and keys[pygame.K_a] and keys[pygame.K_k]:
                            print("Full Left")
                            # Firstly we need to save our reshaped image
                            image_samples = np.vstack((image_samples, reshaped_array)) # this function adds reshaped_array to image_samples vertically.
                            # Then, we need to add our input to image_labels
                            image_labels = np.vstack((image_labels, self.directions[0]))
                            # self.directions takes [1, 0, 0, 0, 0]
                            saved_frame_count += 1
                            self.key_set.add("w");self.key_set.add("a");self.key_set.add("k");
                        elif keys[pygame.K_w] and keys[pygame.K_d] and keys[pygame.K_l]:
                            print("Full Right")
                            image_samples = np.vstack((image_samples, reshaped_array))
                            # [0, 0, 0, 0, 1] for full right
                            image_labels = np.vstack((image_labels, self.directions[4]))
                            saved_frame_count += 1
                            self.key_set.add("w");self.key_set.add("d");self.key_set.add("l");
                        elif keys[pygame.K_w] and keys[pygame.K_a]:
                            print("Left")
                            image_samples = np.vstack((image_samples, reshaped_array))
                            # [0, 1, 0, 0, 0] for left
                            image_labels = np.vstack((image_labels, self.directions[1]))
                            saved_frame_count += 1
                            self.key_set.add("w");self.key_set.add("a");
                        elif keys[pygame.K_w] and keys[pygame.K_d]:
                            print("Right")
                            image_samples = np.vstack((image_samples, reshaped_array))
                            # [0, 0, 0, 1, 0] for right
                            image_labels = np.vstack((image_labels, self.directions[3]))
                            saved_frame_count += 1
                            self.key_set.add("w");self.key_set.add("d");
                        elif keys[pygame.K_w]:
                            print("Forward")
                            image_samples = np.vstack((image_samples, reshaped_array))
                            # [0, 0, 1, 0, 0] for right
                            image_labels = np.vstack((image_labels, self.directions[2]))
                            saved_frame_count += 1
                            self.key_set.add("w");

                        elif keys[pygame.K_q]:
                            print("Finished")
                            self.ready_to_go = False
                            break
                    elif event.type == KEYUP:
                        if chr(event.key) == "w":
                            self.key_set.clear()
                        elif chr(event.key) in self.key_set:
                            self.key_set.remove(chr(event.key))
                            if "a" in self.key_set:
                                print("Left")
                                image_samples = np.vstack((image_samples, reshaped_array))
                                # [0, 1, 0, 0, 0] for left
                                image_labels = np.vstack((image_labels, self.directions[1]))
                                saved_frame_count += 1
                            elif "d" in self.key_set:
                                print("Right")
                                image_samples = np.vstack((image_samples, reshaped_array))
                                # [0, 0, 0, 1, 0] for right
                                image_labels = np.vstack((image_labels, self.directions[3]))
                                saved_frame_count += 1
                            elif "w" in self.key_set:
                                print("Forward")
                                image_samples = np.vstack((image_samples, reshaped_array))
                                # [0, 0, 1, 0, 0] for right
                                image_labels = np.vstack((image_labels, self.directions[2]))
                                saved_frame_count += 1
                    ####
                    ## TODO: Send direction information to car via socket
                    ####
                    print("Key Set: ", self.key_set)
                    ## end of socket communication

        # We dont need first rows, because all values are zero there
        train = image_samples[1:, :]
        train_labels = image_labels[1:, :]

        np.savez("data/numpy/training.npz", train= train, train_labels= train_labels)

        print("Training data saved.")
        print(train.shape)
        print(train_labels.shape)
        print("Total frame:", frame_count)
        print("Saved frame:", saved_frame_count)

if __name__ == '__main__':
    TrainingCollector()
