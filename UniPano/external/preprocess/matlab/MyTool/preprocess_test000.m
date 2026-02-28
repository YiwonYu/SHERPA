clc;
clear all;
add_path;


panoNum = 4
for i = 0:panoNum
    
%%
inputFile = ['D:\Dataset\iStaging\0001~0050\0003\' num2str(i) '.jpg'] ;
outputPath = ['D:\Dataset\iStaging\0001~0050\0003\' num2str(i) '\'];

%%
panoImg = imread(inputFile);
panoImg = imresize(panoImg, [1024, 2048]);
panoImg = im2double(panoImg);
%figure; imshow(panoImg)

%%
panoSep = getPanoSeperate(panoImg, 320);

%%
[ olines, vp, views, edges, panoEdge, score, angle] = getPanoEdges(panoImg, panoSep, 2);
%figure; imshow(panoEdge);

%%
[ ~, panoOmap ] = getPanoOmap( views, edges, vp );
%figure; imshow(panoOmap);

%%
vp = vp(3:-1:1,:);
[ panoImg_rot, R ] = rotatePanorama( panoImg, vp);
%figure; imshow(panoImg_rot);
panoEdge_rot = rotatePanorama(panoEdge, [], R);
%figure; imshow(panoEdge_rot);
panoOmap_rot = rotatePanorama(panoOmap, [], R);
%figure; imshow(panoOmap_rot);

%%
if ~exist(outputPath,'dir') 
    mkdir(outputPath);
end

imwrite(panoImg_rot,[outputPath 'pano_color.png']);
imwrite(panoEdge_rot,[outputPath 'pano_edge.png']);
imwrite(panoOmap_rot,[outputPath 'pano_omap.png']);

end

