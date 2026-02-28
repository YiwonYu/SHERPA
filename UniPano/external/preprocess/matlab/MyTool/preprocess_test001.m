clc;
clear all;
add_path;

%%
datasetPath = 'C:\Users\SunDa\Desktop\SiggraphAsia\Dataset'
houseDirs = dir(datasetPath);
houseDirs = houseDirs(3:end);

for house = houseDirs'
   disp(house.name);
   
   panoDirs = dir([datasetPath '\' house.name]);
   panoDirs = panoDirs(3:end);

   for pano = panoDirs'
      panoPath = [datasetPath '\' house.name '\' pano.name];
      disp(panoPath) ;
      
      panoImg = imread([panoPath '\' 'color.jpg']);
      panoImg = imresize(panoImg, [1024, 2048]);
      panoImg = im2double(panoImg);
      %figure; imshow(panoImg)
      
      depthImg = imread([panoPath '\' 'depth_fixed.png']);
      depthImg = imresize(depthImg, [1024, 2048]);

      panoSep = getPanoSeperate(panoImg, 320);
      [ olines, vp, views, edges, panoEdge, score, angle] = getPanoEdges(panoImg, panoSep, 0.7);
      %se = strel('diamond',8);
      %panoEdge_dilate = imdilate(panoEdge,se);
      %panoEdge_blur = imgaussfilt(panoEdge_dilate, [10,10]);
      [ ~, panoOmap ] = getPanoOmap( views, edges, vp );
      vp = vp(3:-1:1,:);
      [ panoImg_rot, R ] = rotatePanorama( panoImg, vp);
      panoEdge_rot = rotatePanorama(panoEdge, [], R);
      panoOmap_rot = rotatePanorama(panoOmap, [], R);
      
      depthImg_rot = rotatePanorama(depthImg, [], R);
      
      imwrite(panoImg_rot,[panoPath '\pano_color.png']);
      imwrite(panoEdge_rot,[panoPath '\pano_edge.png']);
      imwrite(panoOmap_rot,[panoPath '\pano_omap.png']);
      
      depthImg_rot = uint16(depthImg_rot);
      imwrite(depthImg_rot,[panoPath '\pano_depth.png']);
      
   end
end
