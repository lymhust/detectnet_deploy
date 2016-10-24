function classify(img_ori)
	-- Process one image
	local img = img_ori:permute(3, 1, 2)
    --print(#img)

	sys.tic()
	model_dec:forward(img:reshape(1,3,im_h,im_w))
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
			bbox_nms = torch.Tensor(indx:size(1), 5)
			local id = 0
			for i = 1, indx:size(1) do
				bbox_nms[i][1] = bbox[indx[i]][1]
				bbox_nms[i][2] = bbox[indx[i]][2]
				bbox_nms[i][3] = bbox[indx[i]][3]
				bbox_nms[i][4] = bbox[indx[i]][4]
				bbox_nms[i][5] = bbox[indx[i]][5]
				-- image.drawRect(img, left, top, right, bottom, {lineWidth=2, color={255,0,0}, inplace=true})
				-- image.drawText(img, string.format('%.2f', score), left, top, {color={0,255,0}, bg={0,0,0}, size=2, inplace=true})			
			end
			-------------------------------------------------------------------------------------------------------------------------
				
		end -- if (indx>0)
	end -- if (box_loc:dim()>0)
--print(sys.toc()*1000)
end

function test(img)
	print('test.....')
	aa = torch.Tensor(10,2)
end
















