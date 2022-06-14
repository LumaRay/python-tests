import cv2

import serial

import time

import numpy as np

'''def comPortRemove(comPort):
    try:
        comPort.close()
    except:
        print("Failed to close com port!")
    if _platform == "win32" or _platform == "win64":
        call(["devcon.exe", "remove", "USB\VID_04B4&PID_F9F9&MI_02*"])

def comPortRescan():
    if _platform == "win32" or _platform == "win64":
        call(["devcon.exe", "rescan"])
    sleep(30)

def comPortWrite(comPort, message):
    if comPort.isOpen():
        try:
            comPort.write(message)
        except serial.serialutil.SerialException:
            remove(ser)
            rescan()
            try:
                comPort.open()
            except serial.serialutil.SerialException:
                try:
                    remove(comPort)
                    rescan()
                except:
                    print('Failed to reconnect to com port!')
                    return False
            else:
                print('Successfully reconnected to %s' % (ser.portstr))
                comPort.write(message)
                return True
        else:
            return True

comPort = serial.Serial(
    port='/dev/ttyACM0',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)

#comPort.set_buffer_size(rx_size=12800, tx_size=12800)
#comPort.set_buffer_size(rx_size=128000, tx_size=128000)
#comPort.open()
#if comPort.isOpen():
#print("changing palette to "+str(displayPalette))
#displayPalette = 6
#xorByte = 0x07 ^ 0x02 ^ 4 ^ displayPalette
#command = ("\x55\xAA\x07\x02\x00\x04\x00\x00\x00"+chr(displayPalette)+chr(xorByte)+"\xF0").encode('ascii')
command = bytearray([0x55, 0xAA, 0x07, 0xA0, 0x01, 0x09, 0x00, 0x00, 0x00, 0x15, 0xBA, 0xF0])
#comPort.write(command)
comPort.flushInput()
comPort.flushOutput()
comPortWrite(comPort, command)
time.sleep(0.3)
#comPort.flushInput()
recBytes = b''
while True:
    bytesToRead = comPort.inWaiting()
    if bytesToRead <= 0:
        break
    recBytes += comPort.read(bytesToRead)
    time.sleep(0.1)
comPort.close()
#test = comPort.read_all()
curveData = list(recBytes) #12800
curveData = np.reshape(curveData, (int(len(curveData) / 2), 2))
curveData = curveData.astype(np.uint8)
curveData = curveData.view('<u2')'''


cap = cv2.VideoCapture("/dev/video0")

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 384)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288 * 2)

#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', '2'))
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('U', 'Y', 'V', 'Y'))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('R', 'G', '1', '0'))

#cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

frame = cap.read()

#img = cv2.cvtColor(frame[1], cv2.COLOR_YUV2BGR_YUY2)
img = frame[1]#cv2.cvtColor(frame[1], cv2.COLOR_RG102BGR)

cv2.imshow("test", img)

cv2.waitKey(0)