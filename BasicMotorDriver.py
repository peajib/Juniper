# -*- coding: utf-8 -*-
"""
Created on Sat Nov  8 17:47:23 2025

@author: NFDL - 210
"""

# Basic driving of Piezo Motor

import GratingMotor4 as gm

ser = gm.open_port()

# move by 1000 counts

gm.destination(ser, current_abs=0, target_abs=5000)


ser.close()
