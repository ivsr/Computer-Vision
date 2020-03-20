import os
os.chdir('/home/hoaithuong/PycharmProjects/tensorflow/test1')
i=1
for file in os.listdir():
      src=file
      dst="apple."+str(i)+".jpg"
      os.rename(src,dst)
      i+=1
