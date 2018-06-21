function inlierMatches = transParamsRANSAC(locs_1,descriptors_1,locs_2,descriptors_2,matches,numberOfIterations)
% Finding the matches between the two descriptors, a good transformation
% between the two images and returns a subset of the matches that are
% inliners to this transformation

%% Perform RANSAC to find the transformation paramters

best_numInlers = 0;
best_parameters = [];
best_inliers = [];

leftFeaturesMatrix = zeros(2*size(matches,2), 6);
for i = 1:size(matches,2)
    leftFeaturesMatrix((i*2)-1, 1) = locs_1(1, matches(1, i));
    leftFeaturesMatrix((i*2)-1, 2) = locs_1(2, matches(1, i));
    leftFeaturesMatrix((i*2), 3) = locs_1(1, matches(1, i));
    leftFeaturesMatrix((i*2), 4) = locs_1(2, matches(1, i));
    leftFeaturesMatrix((i*2)-1, 5) = 1;
    leftFeaturesMatrix((i*2), 6) = 1;
end

for i=1:numberOfIterations
    % Select three random matched points (sufficient to fit the 6
    % transformation parameters)
    seed = randperm(size(matches,2), 3);

    % Solve for the transformation parameters from L to R:
    A = [locs_1(1, matches(1,seed(1))) locs_1(2, matches(1,seed(1))) 0                              0                              1 0 ; ...
         0                              0                              locs_1(1, matches(1,seed(1))) locs_1(2, matches(1,seed(1))) 0 1 ; ...
         locs_1(1, matches(1,seed(2))) locs_1(2, matches(1,seed(2))) 0                              0                              1 0 ; ...
         0                              0                              locs_1(1, matches(1,seed(2))) locs_1(2, matches(1,seed(2))) 0 1 ; ...
         locs_1(1, matches(1,seed(3))) locs_1(2, matches(1,seed(3))) 0                              0                              1 0 ; ...
         0                              0                              locs_1(1, matches(1,seed(3))) locs_1(2, matches(1,seed(3))) 0 1 ];

    b = [locs_2(1, matches(2,seed(1))); locs_2(2, matches(2,seed(1))); ...
         locs_2(1, matches(2,seed(2))); locs_2(2, matches(2,seed(2))); ...
         locs_2(1, matches(2,seed(3))); locs_2(2, matches(2,seed(3)))];

    transformParams = pinv(A) * b;
    
    % Calculate the transformed features from the left image and plot them on the right:
    transformRightFeatures = reshape(leftFeaturesMatrix * transformParams, [2, size(matches,2)])';

    % Find the inliers
    threshold = 30;
    inliers = find(sqrt(sum((transformRightFeatures-locs_2(1:2, matches(2, :))').^2, 2)) < threshold);
    
    if size(inliers,1) > best_numInlers
        best_numInlers = size(inliers,1);
        best_parameters = transformParams;
        best_inliers = inliers;
    end
end

inlierMatches = matches(:, best_inliers);

end

