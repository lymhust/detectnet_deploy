require 'caffe'
require 'image'
--require 'ffmpeg'
require 'nms'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(8)


-- Functions
-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function rgb2bgr(img)
	local perm = torch.LongTensor({3, 2, 1})
	img = img:index(1, perm):mul(255)
	return img
end

function process_one(img_ori)
	-- Process one image
	local img = rgb2bgr(img_ori)
	--local img = img_ori:permute(3,1,2)
	img = img:reshape(1,3,im_h,im_w)
	
	sys.tic()
	model_dec:forward(img:float())
    local out_pixel = model_dec:getBlobData(ind_pixel):clone()[1][1]
	local out_box = model_dec:getBlobData(ind_box):clone()[1]
	out_pixel[out_pixel:le(0.2)] = 0
	local box_loc = out_pixel:nonzero()
	print('exec time dec forward = '..(sys.toc()*1000)..'ms')
	
--sys.tic()	
	if (box_loc:dim() > 0) then -- Detected
		local bbox = torch.Tensor(box_loc:size(1), 5)
	    local indx = 0

		for i = 1, box_loc:size(1) do
			local r, c = box_loc[i][1], box_loc[i][2]
			local left = math.max(torch.round(out_box[1][r][c]+c*16-15), 1)
			local top = math.max(torch.round(out_box[2][r][c]+r*16-15), 1)
			local right = math.min(torch.round(out_box[3][r][c]+c*16-15), im_w)
			local bottom = math.min(torch.round(out_box[4][r][c]+r*16-15), im_h)
			local w = right-left+1
			local h = bottom-top+1
			local score = out_pixel[r][c]
		
			if (left<right and top<bottom and w>20 and h>20) then
				indx = indx + 1
				bbox[indx][1] = left
				bbox[indx][2] = top
				bbox[indx][3] = right 
				bbox[indx][4] = bottom
				bbox[indx][5] = score
			end
		end
		
		if (indx > 0) then
			bbox = bbox[{{1,indx},{}}]	
			-- Merge box NMS and draw

			local indx = nms(bbox[{{},{1,4}}], 0.5, 'area')
			local id = 0
			for i = 1, indx:size(1) do
				local left = bbox[indx[i]][1]
				local top = bbox[indx[i]][2]
				local right = bbox[indx[i]][3]
				local bottom = bbox[indx[i]][4]
				local score = bbox[indx[i]][5]
				image.drawRect(img_ori, left, top, right, bottom, {lineWidth=2, color={255,0,0}, inplace=true})
				image.drawText(img_ori, string.format('%.2f', score), left, top, {color={0,255,0}, bg={0,0,0}, size=2, inplace=true})			
			end
			-------------------------------------------------------------------------------------------------------------------------
				
		end -- if (indx>0)
	end -- if (box_loc:dim()>0)
--print(sys.toc()*1000)

	return img_ori
end
---------------------------------------------------------------------------

-- Settings
-- File address
local caffe_dec_deploy_file = './model/deploy.prototxt'
local caffe_dec_model_file = './model/snapshot_iter_178000.caffemodel'

-- Load detection and classification caffe net
model_dec = caffe.Net(caffe_dec_deploy_file, caffe_dec_model_file, 'test')
model_dec:setModeGPU()
model_dec:initGPUMemoryScope()

ind_pixel = model_dec:getBlobIndx('coverage')
ind_box = model_dec:getBlobIndx('bboxes')


-- Image info
im_h, im_w = 512, 1024
model_dec:reshape(1, 3, im_h, im_w)
----------------------------------------------------------------------------------

-- Process video
--[[
--vid = ffmpeg.Video{path='./video_test3.mp4', width=im_w, height=im_h, fps=30, length=1000}
--img = image.scale(image.load('./images/peds-007.png'), im_w, im_h)

f = 1
while true do
	local start = sys.clock()
    frame = vid:get_frame(1, f)
	--frame = img:clone()
	frame = process_one(frame)
	local time = sys.clock() - start
	print("FPS: ".. 1/sys.toc())
	--print("Time: "..(time*1000)..'ms')
	f = f + 1
    win = image.display{win=win,image=frame}
end
--]]

---[[
local folder = './ped_images/'

for file in paths.iterfiles(folder) do

	local start = sys.clock()
	frame = image.scale(image.load(folder..file), im_w, im_h, 'simple')
	frame = process_one(frame)
	local time = sys.clock() - start
	print("FPS: ".. 1/time)
	print("Time: "..(time*1000)..'ms\n')
	win = image.display{win=win,image=frame}
end
--]]












