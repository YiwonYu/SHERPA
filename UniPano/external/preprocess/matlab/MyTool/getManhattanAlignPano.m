function [ rotImg, R] = getManhattanAlignPano( panoImg ,panoSize )

panoImg = imresize(panoImg, panoSize);
panoImg = im2double(panoImg);

cutSize = 320;
fov = pi/3;
xh = -pi:(pi/6):(5/6*pi);
yh = zeros(1, length(xh));
xp = [-3/3 -2/3 -1/3 +0/3 +1/3 +2/3 -3/3 -2/3 -1/3 +0/3 +1/3 +2/3] * pi;
yp = [ 1/4  1/4  1/4  1/4  1/4  1/4 -1/4 -1/4 -1/4 -1/4 -1/4 -1/4] * pi;
x = [xh xp 0     0];
y = [yh yp +pi/2 -pi/2];

[sepScene] = separatePano( panoImg, fov, x, y, cutSize);

numScene = length(sepScene);
qError = 0.7;
edge(numScene) = struct('img',[],'edgeLst',[],'vx',[],'vy',[],'fov',[]);
for i = 1:numScene
    cmdline = sprintf('-q %f ', qError);
    [ edgeMap, edgeList ] = lsdWrap( sepScene(i).img, cmdline);
    edge(i).img = edgeMap;
    edge(i).edgeLst = edgeList;
    edge(i).fov = sepScene(i).fov;
    edge(i).vx = sepScene(i).vx;
    edge(i).vy = sepScene(i).vy;
    edge(i).panoLst = edgeFromImg2Pano( edge(i) );
end
[lines,olines] = combineEdgesN( edge);

[ olines, mainDirect] = vpEstimationPano( lines );

vp = mainDirect(3:-1:1,:);
[ rotImg, R ] = rotatePanorama( panoImg, vp );

end

