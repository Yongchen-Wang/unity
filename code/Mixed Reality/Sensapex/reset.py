from sensapex import UMP

ump = UMP.get_ump()
dev_ids = ump.list_devices()
print(dev_ids)

device1 = ump.get_device(1)
# device1.calibrate_zero_position()

# device2 = ump.get_device(2)
# device2.calibrate_zero_position()




# device1 = ump.get_device(1)
# device1.goto_pos((0, 0, 11000, 10000), 8000)
# device1.goto_pos((18000, 0, 11000, 10000), 8000)
# device1.goto_pos((18000, 18000, 11000, 10000), 8000)
# device1.goto_pos((0, 18000, 11000, 10000), 8000)




# device1.goto_pos((2500, 2500, 11000, 10000), 8000)

device1.goto_pos((12000, 14000, 11000, 10000), 8000)

# device1.goto_pos((15000, 2000, 11000, 10000), 8000)


# device1.goto_pos((3000, 14000, 11000, 10000), 8000)

# device1.goto_pos((12000, 16000, 11000, 10000), 8000)


# device2 = ump.get_device(2)
# device2.goto_pos((20000, 20000, 15000, 10000), 3000)

# device2.goto_pos((20000-4500, 20000, 15000, 10000), 3000)

# device2.goto_pos((20000-4500, 20000-13000, 15000, 10000), 3000)



# from sensapex import UMP
# import time  # 导入time模块

# ump = UMP.get_ump()
# dev_ids = ump.list_devices()
# print(dev_ids)
# device2 = ump.get_device(2)



