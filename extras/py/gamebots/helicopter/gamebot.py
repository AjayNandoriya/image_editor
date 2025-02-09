import os
import sys
import mss
import pyautogui
pyautogui.PAUSE = 0
import numpy as np
import cv2
import keyboard
import mouse

#https://www.addictinggames.com/clicker/helicopter-game



class Helicopter(object):
    def __init__(self):
        # xauth add 10.255.255.254:0 . $(xxd -l 16 -p /dev/urandom)
        # 
        self.dimensions_left = {
            'left': 200,
            'top': 300,
            'width': 680,
            'height': 480
        }
        self.sct = mss.mss()

        template_fname = os.path.join(os.path.dirname(__file__),'helicopter.png')
        self.template_hlcp = cv2.imread(template_fname)
        template_fname = os.path.join(os.path.dirname(__file__),'block.png')
        self.template_block = cv2.imread(template_fname)
        template_fname = os.path.join(os.path.dirname(__file__),'start.png')
        self.template_start = cv2.imread(template_fname)
        template_fname = os.path.join(os.path.dirname(__file__),'end.png')
        self.template_end = cv2.imread(template_fname)
        pass
        
    def get_screen(self):
        scr = np.array(self.sct.grab(self.dimensions_left))
        return scr[:,:,:3]
    
    @staticmethod   
    def find(scr, tmpl):

        result = cv2.matchTemplate(scr, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        # print(f"Max Val: {max_val} Max Loc: {max_loc}")
        return max_loc, max_val
    
    def go_up(self):
        if not mouse.is_pressed():
            pyautogui.mouseDown(button='left')
            print('up')
        pass
    def go_down(self):
        if mouse.is_pressed('left'):
            pyautogui.mouseUp(button='left')
            print('down')
        pass
    def run(self):
        y = self.dimensions_left['top'] + 100
        x = self.dimensions_left['left'] + 100
        
        while True:
            scr = self.get_screen()
            scr_2 = scr.copy()
            max_loc_hlp, max_val_hlp = self.find(scr, self.template_hlcp)
            if max_val_hlp < 0.7:
                continue
            h,w,_ = self.template_hlcp.shape
            cv2.rectangle(scr_2, max_loc_hlp, (max_loc_hlp[0] + w, max_loc_hlp[1] + h), (0,255,255), 2)
            # print(f'helicop: {max_loc_hlp}')
            max_loc_blk,max_val_blk = self.find(scr, self.template_block)
            if max_val_blk > 0.7:
                print(f'helicop: {max_loc_hlp}, block: {max_loc_blk}')
                max_loc_end,max_val_end = self.find(scr, self.template_end)
                if max_val_end > 0.7:
                    print('end')
                    print(f'helicop: {max_loc_hlp}, block: {max_loc_blk}')
                    self.go_down()
                    continue
                    
                # print(f'block: {max_loc_blk}')
                h,w,_ = self.template_block.shape
                cv2.rectangle(scr_2, max_loc_blk, (max_loc_blk[0] + w, max_loc_blk[1] + h), (255,0,0), 2)

                hlp_y0 = max_loc_hlp[1] 
                hlp_y1 = max_loc_hlp[1] + self.template_hlcp.shape[0]
                blk_y0 = max_loc_blk[1] + 10
                blk_y1 = max_loc_blk[1] + self.template_block.shape[0] - 10

                if hlp_y0 < 150:
                    self.go_down()
                elif hlp_y1>300:
                    self.go_up()
                elif hlp_y1 > blk_y0-20 and hlp_y0 < (blk_y1+blk_y0)/2:
                    self.go_up()
                else:
                    self.go_down()
                
            else:
                max_loc_start,max_val_start = self.find(scr, self.template_start)
                if max_val_start > 0.7:
                    print('start')
                    self.go_up()
                
                    
            cv2.imshow('Screen Shot', scr_2)
            cv2.waitKey(1)
            if keyboard.is_pressed('q'):
                break



def test():
    game = Helicopter()
    game.run()
    pass


if __name__ == '__main__':
    test()
