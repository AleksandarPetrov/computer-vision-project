%% Load the external SIFT toolbox
run('vlfeat-0.9.21/toolbox/vl_setup.m')
vl_version verbose

%% Generate the list of input image filenames (incl. subfolder)
filelistHaraff = [];
filelistHesaff = [];
ims = cell(19, 1);

for i=1:19
    filelistHaraff = [filelistHaraff; sprintf('model_castle/castle%02d.png.haraff.sift',i)];
    filelistHesaff = [filelistHesaff; sprintf('model_castle/castle%02d.png.hesaff.sift',i)];
    ims{i} =  imread(sprintf('model_castle/castle%02d.png',i));
end

%% Get the interest points
[locs, descriptors, PVM] = detectInterestPoints(filelistHaraff, filelistHesaff);

%% Get the number of intersection points

for i = 1:(size(PVM, 1)-2)
    existingInFirstAndSecond = all(PVM(i:i+1, :));
    existingInSecondAndThird = all(PVM(i+1:i+2, :));
    combo = [existingInFirstAndSecond ; existingInSecondAndThird];
    comboBoth = all(combo);
    
    fprintf("Intersections for sets #%d, #%d and #%d, #%d: %d\n", i, i+1, i+1, i+2, sum(comboBoth))
end

%% Get the color of the interest points
colors = zeros(3, size(PVM, 2));
for intPointIdx = 1:size(PVM, 2)

    firstImageWithThisPoint = find(PVM(:, intPointIdx) >0, 1);
    indxInThisImage = PVM(firstImageWithThisPoint, intPointIdx);
    locOfThisPointInTheImage = floor(locs{firstImageWithThisPoint}(:, indxInThisImage));
    im = ims{firstImageWithThisPoint};
    colorVector = squeeze(im(max(1, locOfThisPointInTheImage(2)), max(1, locOfThisPointInTheImage(1)), :));
    colors(:, intPointIdx) = colorVector;
end
%% Show the matchings
% Here you can test whether the matching was correct. Select any two images
% and the pairings between them will be vizualized

if true

    leftIdx = 1;
    rightIdx = 2;

    % Load the images
    imL = imread(sprintf('model_castle/castle%02d.png',leftIdx));
    imL = im2double(imL);
    if size(imL, 3) > 1
        imL = rgb2gray(imL);
    end

    imR = imread(sprintf('model_castle/castle%02d.png',rightIdx));
    imR = im2double(imR);
    if size(imR, 3) > 1
        imR = rgb2gray(imR);
    end

    matchedColumns = find(all(PVM([leftIdx,rightIdx], :)));

    indicesLeft = PVM(leftIdx, matchedColumns);
    indicesRight = PVM(rightIdx, matchedColumns);
    
    sampleIdxes = randsample(size(indicesRight,2),20);
    indicesLeft = indicesLeft(sampleIdxes);
    indicesRight = indicesRight(sampleIdxes);

    showMatchedFeatures(imL,imR, ...
                        locs{leftIdx}(:, indicesLeft)', ...
                        locs{rightIdx}(:, indicesRight)', 'montage');
                
end

%% Perform 3D reconstruction for each pair of consecutive images

% This will hold the 3D locations for the points as we reconsturct and add
% them to the global model. NaN means that a particular point hasn't been
% added yet
[coordinates, cameras] = structureReconstruction(locs, descriptors, PVM(1:19, :));

%% Show the original points and the 3D points as projected by the estimated cameras

if true
    imIdx = 11;
    
    % Load the image
    im = imread(sprintf('model_castle/castle%02d.png',imIdx));
    
    visPointsIdicesLocal = PVM(imIdx, :);
    visPointsIdicesGlobal = find(visPointsIdicesLocal~=0);
    visPointsIdicesLocal(visPointsIdicesLocal==0) = [];
    
    % Subset them to not have a cluttered image
    sampleIdxes = randsample(size(visPointsIdicesLocal,2),50);
    visPointsIdicesLocal = visPointsIdicesLocal(sampleIdxes);
    visPointsIdicesGlobal = visPointsIdicesGlobal(sampleIdxes);
    
    originalLocations = locs{imIdx}(:, visPointsIdicesLocal)';
    
    % Find where the 3D points are projected
    camera = cameras((imIdx*2)-1:(imIdx*2),:);
    visPointsCoordinates = coordinates(:, visPointsIdicesGlobal);
    visPointsCoordinates(4, :) = 1;
    projectedLocations = (camera * visPointsCoordinates)';
    
    showMatchedFeatures(im,im, ...
                        originalLocations, ...
                        projectedLocations, 'montage');
                    
    figure
    hist(sqrt(sum((projectedLocations-originalLocations).^2, 2)))
end

%% Visualize the 3D reconstruction


NaNCols = any(isnan(coordinates));
coordinates = coordinates(:, ~NaNCols);
colors = colors(:, ~NaNCols);

addpath('./MyCrustOpen070909')

% Run  program
p = coordinates';
%p = unique(coordinates','rows');
[t]=MyCrustOpen(p);

% plot the points cloud
figure(1);
set(gcf,'position',[0,0,1280,800]);
%subplot(1,2,1)
hold on
axis equal
title('Points Cloud','fontsize',14)
scatter3(p(:,1),p(:,2),-p(:,3), 5, colors'/255, 'filled')
axis vis3d
axis off
view(3)



% plot the output triangulation
% figure(1)
% subplot(1,2,2)
% hold on
% title('Output Triangulation','fontsize',14)
% axis equal
% trisurf(t,p(:,1),p(:,2),p(:,3),'facecolor','c','edgecolor','b')%plot della superficie
% axis vis3d
% view(3)