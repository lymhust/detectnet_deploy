require 'caffe'
require 'sys'
-- require 'image'
require './src/luafile/nms'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(8)

-------------------------------------------------------------------
-- Init Settings
-- File address
local caffe_dec_deploy_file = './src/luafile/model/deploy.prototxt'
local caffe_dec_model_file = './src/luafile/model/snapshot_iter_178000.caffemodel'

-- Load detection and classification caffe net
model_dec = caffe.Net(caffe_dec_deploy_file, caffe_dec_model_file, 'test')
model_dec:setModeGPU()
model_dec:initGPUMemoryScope()

ind_pixel = model_dec:getBlobIndx('coverage')
ind_box = model_dec:getBlobIndx('bboxes')

-- Image info
im_h, im_w = 512, 1024
model_dec:reshape(1, 3, im_h, im_w)

print('DetectNet loaded')
-------------------------------------------------------------------








