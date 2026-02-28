clc;
clear all;
add_path;

%% for My tool
panoImg = imread('D:\Dataset\Test\sun360/005.jpg');
[panoRot, R] = getManhattanAlignPano(panoImg, [1024, 2048]);

imwrite(panoRot, 'D:\Dataset\Test\sun360/005_p.jpg');



