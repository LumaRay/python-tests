import glob, shutil

file_list = glob.glob('F:\\Work\\InfraredCamera\\ThermalView\\tests\\work\\false_mask\\*\\*COLOR_SOURCE.jpg')
for f in file_list:
    shutil.move(f, 'F:\\Work\\InfraredCamera\\ThermalView\\tests\\work\\false_mask')

file_list = glob.glob('F:\\Work\\InfraredCamera\\ThermalView\\tests\\work\\false_no_mask\\*\\*COLOR_SOURCE.jpg')
for f in file_list:
    shutil.move(f, 'F:\\Work\\InfraredCamera\\ThermalView\\tests\\work\\false_no_mask')