import os

#ootdir="media/jcq/study/opencv/Projects/DATA/vehicles/KITTI_extracted"
#rootdir="/media/jcq/study/opencv/Projects/DATA/vehicles/KITTI_extracted"
#rootdir="/media/jcq/study/opencv/Projects/DATA/no-vehicles_128"
#rootdir="/media/jcq/study/opencv/Projects/DATA/no-vehicles_128x64"
#rootdir="/media/jcq/study/opencv/Projects/DATA/vehicles_128x64"

#namelist of handexamples
rootdir="/media/jcq/study/opencv/Projects/DATA/non-vehicles/GTI"



files = os.listdir(rootdir)



with open('non-vehicles-GIT.txt','w') as f:
    for file in files:
        
        f.write(file+'\n')
#
#print(file)
#file_object=open('train_list.txt','w')
# for parent,dirnames,filenames in os.walk(rootdir):
 # for filename in filenames:
  # print filename 
 # file_object.write(filename+'\n')
# file_object.close()
