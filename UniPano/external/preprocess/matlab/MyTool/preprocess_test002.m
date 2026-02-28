clc;
clear all;
add_path;

%%
datasetPath = 'C:\Users\SunDa\Desktop\SiggraphAsia\Dataset\SUN360';
houseDirs = dir(datasetPath);
houseDirs = houseDirs(3:end);
%houseDirs = houseDirs(3:3);

for house = houseDirs'
   disp(house.name);
   
   panoDirs = dir([datasetPath '\' house.name]);
   panoDirs = panoDirs(3:end);
   %panoDirs = panoDirs(3:3);

   for pano = panoDirs'
       
      [filepath, panoname,ext] = fileparts( pano.name);
      panoPath = [datasetPath '\' house.name '\' panoname];
      disp(pano.name);
      
      panoImg = imread([panoPath '.jpg']);
      panoImg = imresize(panoImg, [1024, 2048]);
      panoImg = im2double(panoImg);
      %figure; imshow(panoImg)
      
      panoSep = getPanoSeperate(panoImg, 320);
      [ olines, vp, views, edges, panoEdge, score, angle] = getPanoEdges(panoImg, panoSep, 0.7);
      [ ~, panoOmap ] = getPanoOmap( views, edges, vp );
      vp = vp(3:-1:1,:);
      [ panoImg_rot, R ] = rotatePanorama( panoImg, vp);
      panoEdge_rot = rotatePanorama(panoEdge, [], R);
      panoOmap_rot = rotatePanorama(panoOmap, [], R);
      
      if ~exist(panoPath,'dir') 
        mkdir(panoPath);
      end
      
      imwrite(panoImg_rot,[panoPath '\pano_color.png']);
      imwrite(panoEdge_rot,[panoPath '\pano_edge.png']);
      imwrite(panoOmap_rot,[panoPath '\pano_omap.png']);
      
      movefile([panoPath '.jpg'], panoPath);
     
   end
end
