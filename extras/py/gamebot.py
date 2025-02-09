import mss
import pyautogui
pyautogui.PAUSE = 0
import numpy as np
import cv2
import keyboard


class Helicopter(object):
    def __init__(self):
        # xauth add 10.255.255.254:0 . $(xxd -l 16 -p /dev/urandom)
        # 
        self.dimensions_left = {
            'left': 0,
            'top': 200,
            'width': 500,
            'height': 500
        }
        self.sct = mss.mss()
        pass
        
    def get_screen(self):
        scr = np.array(self.sct.grab(self.dimensions_left))
        return scr[:,:,:3]
    
    @staticmethod   
    def find(scr, temp):

        result = cv2.matchTemplate(scr, temp, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        print(f"Max Val: {max_val} Max Loc: {max_loc}")
        return max_loc
    
    def run(self):
        while True:
            scr = self.get_screen()
            src = scr.copy()
            # max_loc = self.find(scr, temp)
            # h,w = temp.shape
            # cv2.rectangle(scr, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,255,255), 2)

            cv2.imshow('Screen Shot', scr)
            cv2.waitKey(1)
            if keyboard.is_pressed('q'):
                break

def test():
    game = Helicopter()
    game.run()
    pass


if __name__ == '__main__':
    test()