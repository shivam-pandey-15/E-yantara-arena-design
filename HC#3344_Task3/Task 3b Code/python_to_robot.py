
import serial
import time

def transfer(file):
        s = serial.Serial('COM3')

        file = file + '#'

        #file = file.encode()

        print(time.time())

        for i in range(len(file)):
                s.write(file[i].encode())
                print(file[i].encode(),'----')
                time.sleep(.1)



        print(file)

        print(time.time())

if __name__ =="__main__":
    a = open('example_2.txt','r')
    file = a.read()
    transfer(file)
