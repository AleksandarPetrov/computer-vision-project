%% Load the external SIFT toolbox
run('vlfeat-0.9.21/toolbox/vl_setup.m')
vl_version verbose

%% Load the images
imL = imread('left.jpg');
imL = im2double(imL);
if size(imL, 3) > 1
    imL = rgb2gray(imL);
end

imR = imread('right.jpg');
imR = im2double(imR);
if size(imR, 3) > 1
    imR = rgb2gray(imR);
end

%% Obtain their features
[framesL, descL]= vl_sift(single(imL));
[framesR, descR]= vl_sift(single(imR));

if false
    imagesc(imL);
    hold on;
    h1 = vl_plotframe(framesL) ;
    h2 = vl_plotframe(framesL) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','y','linewidth',2) ;

    figure
    imagesc(imR);
    hold on;
    h1 = vl_plotframe(framesR) ;
    h2 = vl_plotframe(framesR) ;
    set(h1,'color','k','linewidth',3) ;
    set(h2,'color','y','linewidth',2) ;
end

%% Find possible matches
[matches] = vl_ubcmatch(descL,descR);

% Plot the matches:
figure
showMatchedFeatures(imL,imR, ...
                    uint16(framesL(1:2, matches(1, :))'), ...
                    uint16(framesR(1:2, matches(2, :))'), 'montage');

%% Perform RANSAC to find the transformation paramters

best_numInlers = 0;
best_parameters = [];

leftFeaturesMatrix = zeros(2*size(matches,2), 6);
for i = 1:size(matches,2)
    leftFeaturesMatrix((i*2)-1, 1) = framesL(1, matches(1, i));
    leftFeaturesMatrix((i*2)-1, 2) = framesL(2, matches(1, i));
    leftFeaturesMatrix((i*2), 3) = framesL(1, matches(1, i));
    leftFeaturesMatrix((i*2), 4) = framesL(2, matches(1, i));
    leftFeaturesMatrix((i*2)-1, 5) = 1;
    leftFeaturesMatrix((i*2), 6) = 1;
end

numberOfIterations = 1000;

for i=1:numberOfIterations
    % Select three random matched points (sufficient to fit the 6
    % transformation parameters)
    seed = randperm(size(matches,2), 3);

    % Solve for the transformation parameters from L to R:
    A = [framesL(1, matches(1,seed(1))) framesL(2, matches(1,seed(1))) 0                              0                              1 0 ; ...
         0                              0                              framesL(1, matches(1,seed(1))) framesL(2, matches(1,seed(1))) 0 1 ; ...
         framesL(1, matches(1,seed(2))) framesL(2, matches(1,seed(2))) 0                              0                              1 0 ; ...
         0                              0                              framesL(1, matches(1,seed(2))) framesL(2, matches(1,seed(2))) 0 1 ; ...
         framesL(1, matches(1,seed(3))) framesL(2, matches(1,seed(3))) 0                              0                              1 0 ; ...
         0                              0                              framesL(1, matches(1,seed(3))) framesL(2, matches(1,seed(3))) 0 1 ];

    b = [framesR(1, matches(2,seed(1))); framesR(2, matches(2,seed(1))); ...
         framesR(1, matches(2,seed(2))); framesR(2, matches(2,seed(2))); ...
         framesR(1, matches(2,seed(3))); framesR(2, matches(2,seed(3)))];

    transformParams = pinv(A) * b;
    
    % Calculate the transformed features from the left image and plot them on the right:
    transformRightFeatures = reshape(leftFeaturesMatrix * transformParams, [2, size(matches,2)])';

    % Find the inliers
    threshold = 10;
    inliers = find(sqrt(sum((transformRightFeatures-framesR(1:2, matches(2, :))').^2, 2)) < threshold);
    
    if size(inliers,1) > best_numInlers
        best_numInlers = size(inliers,1);
        best_parameters = transformParams;
    end
end

figure
transformRightFeatures = reshape(leftFeaturesMatrix * best_parameters, [2, size(matches,2)])';
showMatchedFeatures(imL,imR, ...
                    uint16(framesL(1:2, matches(1, :))'), ...
                    transformRightFeatures, 'montage');


% Apply the transformation to the right image
T = [best_parameters(1) best_parameters(2) best_parameters(5);...
     best_parameters(3) best_parameters(4) best_parameters(6);...
     0                  0                  1                ];
transform = affine2d(T');
imR_transformed = imwarp(imR, transform, 'bicubic');

%% Combine the two images into one

numImages = 2;
images{1} = imR;
images{2} = imL;

tforms = [affine2d(eye(3)); transform]
imageSize = zeros(numImages,2);
for i = 1:numImages
    currImage = cell2mat(images(i));
    imageSize(i,:) = size(currImage);
end

for i = 1:numImages
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = zeros([height width], 'like', cell2mat(images(1)));

blender = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages

    I = cell2mat(images(i));
    class(I)
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    % Generate a binary mask.
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)
