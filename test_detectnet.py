import os
import sys
import caffe
import math
import cv2

def predict_caffe():


    txtfilename = '/home/tengfeixing/lenovo/ImageDatas/future_challenge_2016_dataset/TSD-Vehicle/filelist.txt'
    detectedfilename = '/home/tengfeixing/lenovo/ImageDatas/future_challenge_2016_dataset/TSD-Vehicle/detection-temp.txt'
    MODEL_FILE = '/home/tengfeixing/Work/DIGITS-digits-4.0/examples/object-detection/detectnet/snapshot_iter_210342.caffemodel'
    DEPLOY_FILE = '/home/tengfeixing/Work/DIGITS-digits-4.0/examples/object-detection/detectnet/deploy.prototxt'
    TEST_ROOT = ''#'/home/tengfeixing/ssd/testimages/kitti_test/'

    caffe.set_mode_gpu()
    #caffe.set_mode_cpu()
    net = caffe.Net(DEPLOY_FILE, MODEL_FILE, caffe.TEST)

    # 'data'?????eploy?????
    # input: "data"
    # input_dim: 1
    # input_dim: 3
    # input_dim: 32
    # input_dim: 96
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # python read file format is H-W-K,need to transform to K-H-W
    transformer.set_transpose('data', (2, 0, 1))#using model trained by matlab, need to transpose the image.
    # python stores images as [0,1], while caffe stores images as [0,255]. need transform
    transformer.set_raw_scale('data', 255)
    # caffe's color sqeunce is BGR, while original squence is RGB. need transfrom
    transformer.set_channel_swap('data', (2, 1, 0))#using model trained by matlab, image in matlab is RGB format
    # reshape input image to deploy's W-H
    net.blobs['data'].reshape(1, 3, 1024, 1280)

    detectedfile = open(detectedfilename, 'w')
    txtfile=open(os.path.join(TEST_ROOT, txtfilename),'r')
    lines=txtfile.readlines()
    txtfile.close()
    mean = 0
    loss = 0
    for l in lines:
        l=l.split('\n')[0]
        l=l.split(' ')
        filename = l[0]
        #label = float(l[-1])
    ##for dirs, dirname, files in os.walk(TEST_ROOT):
    ##    for filename in files:
    # ???/caffe/python/caffe/io.py
        time = cv2.getTickCount()
        img = caffe.io.load_image(os.path.join(TEST_ROOT,filename))
        image = cv2.imread(os.path.join(TEST_ROOT,filename))
        print "================\n"
        print filename
    	# ???????????????H?W?K??????

    	# ????????????
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        #print net.blobs['data'].data[...]
    	# ????????????
        out = net.forward()
    	# ????????????????????????
        predicts = out["bbox-list"][0]
        #coverages = out["coverage"]

        # print '====structure of coverage'
        # coverages = net.blobs["coverage"]

        # print type(coverages) #<class 'caffe._caffe.Blob'>

        # print 'num:', net.blobs['coverage'].num
        # print 'channels:', net.blobs['coverage'].channels
        # print 'height:', net.blobs['coverage'].height
        # print 'width:', net.blobs['coverage'].width
        # #print coverages.data
        # print type(coverages.data) #<type 'numpy.ndarray'>
        # print coverages.data.size
        # print coverages.data.ndim
        # print coverages.data.shape

        # print '\n====structure of bboxes'
        # boxes = net.blobs["bboxes"]

        # print type(boxes) #<class 'caffe._caffe.Blob'>

        # print 'num:', net.blobs['bboxes'].num
        # print 'channels:', net.blobs['bboxes'].channels
        # print 'height:', net.blobs['bboxes'].height
        # print 'width:', net.blobs['bboxes'].width
        # #print boxes.data
        # print type(boxes.data) #<type 'numpy.ndarray'>
        # print boxes.data.size
        # print boxes.data.ndim
        # print boxes.data.shape


        time = cv2.getTickCount() - time
        time = time * 1000 / cv2.getTickFrequency()
        print "time:", time, "ms"
        #print coverages
        # print predicts
        # print type(predicts)
        # print predicts.size
        # print predicts.ndim
        # print predicts.shape

        # print type(net.blobs['data'])
        # print 'num:', net.blobs['data'].num
        # print 'channels:', net.blobs['data'].channels
        # print 'height:', net.blobs['data'].height
        # print 'width:', net.blobs['data'].width

        # print 'type of image:', type(image)
        # print 'shape of image:', image.shape

        names = filename.split('/')
        imagename = '{}/{}\n'.format(names[-2], names[-1])
        detectedfile.write(imagename)
        detectedobjnum = 0
        for i in range (0, predicts.shape[0]):
            if predicts[i][4] > 0.0:
                detectedobjnum = detectedobjnum + 1
            else:
                break

        detectedfile.write('{}\n'.format(detectedobjnum))
        #print detectedobjnum
        for i in range (0, predicts.shape[0]):
            if predicts[i][4] > 0.0:
                #Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                score =  predicts[i][4]
                xmin = int(predicts[i][0]*image.shape[1]/net.blobs['data'].width)
                ymin = int(predicts[i][1]*image.shape[0]/net.blobs['data'].height)
                xmax = int(predicts[i][2]*image.shape[1]/net.blobs['data'].width)
                ymax = int(predicts[i][3]*image.shape[0]/net.blobs['data'].height)
                
                detectedfile.write("{} {} {} {} {}\n".format(xmin, ymin, xmax-xmin, ymax-ymin, score))
                # xmin = int(predicts[i][0])
                # ymin = int(predicts[i][1])
                # xmax = int(predicts[i][2])
                # ymax = int(predicts[i][3])
                if score < 1.5:
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)
                else:
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), 255, 2)
                print xmin, ymin, xmax-xmin, ymax-ymin, score

                #print predicts[i]
    	# ???'prob'?????eploy?????
    	# layer {
    	#   name: "prob"
    	#   type: "Softmax"
    	#   bottom: "ip2"
    	#   top: "prob"
    	# }
        #print 'ip2', predicts
        #print 'ip2.data', net.blobs['fc6'].data
        
        #pridect = predicts.argmax()
        
        cv2.namedWindow("image", 0)
        cv2.imshow("image", image)
        cv2.waitKey(10)

        
    detectedfile.close()

def main():
    predict_caffe()
    pass

if __name__ == '__main__':
    main()
