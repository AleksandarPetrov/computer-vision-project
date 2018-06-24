function inlierMatches = normalized8pointRANSAC(locs_1,descriptors_1,locs_2,descriptors_2,matches,numberOfIterations)

%NORMALIZED8POINTRANSAC Performs the normalized 8-point RANSAC. Based on
%the algorithm explained in https://en.wikipedia.org/wiki/Eight-point_algorithm

%% Get the coordinates of the matched points
matchLocs_1 = locs_1(:, matches(1, :));
matchLocs_2 = locs_2(:, matches(2, :));

% Make them homogeneous
matchLocs_1(3, :) = 1;
matchLocs_2(3, :) = 1;


%% Normalize these coordinates
centroid_1 = mean(matchLocs_1,2);
centroid_2 = mean(matchLocs_2,2);

numOfMatches = size(matches, 2);

meanDisToCentroid_1 = (1/numOfMatches)*sum(sqrt((matchLocs_1(1,:)-centroid_1(1)).^2 + (matchLocs_1(2,:)-centroid_1(2)).^2 ));
meanDisToCentroid_2 = (1/numOfMatches)*sum(sqrt((matchLocs_2(1,:)-centroid_2(1)).^2 + (matchLocs_2(2,:)-centroid_2(2)).^2 ));

scale_1 = sqrt(2)/meanDisToCentroid_1;
scale_2 = sqrt(2)/meanDisToCentroid_2;

T_1 = [scale_1       0          -centroid_1(1)*scale_1  ;
       0             scale_1    -centroid_1(2)*scale_1  ;
       0             0           1                     ];
  
T_2 = [scale_2       0          -centroid_2(1)*scale_2  ;
       0             scale_2    -centroid_2(2)*scale_2  ;
       0             0           1                     ];


matchLocsNorm_1 = T_1 * matchLocs_1;
matchLocsNorm_2 = T_2 * matchLocs_2;

%% Perform RANSAC to find the fundamental matrix F with most inliers

best_numInlers = 0;
best_inliers = [];

threshold = 0.008;

for i=1:numberOfIterations
    % Pick 8 random points
    seed = randperm(size(matches,2), 8);
    
    % Build the matrix with points
    YNorm = zeros(9, 8);
    for j=1:8
        YNorm(1, j) = matchLocsNorm_1(1, seed(j)) * matchLocsNorm_2(1, seed(j));
        YNorm(2, j) = matchLocsNorm_1(1, seed(j)) * matchLocsNorm_2(2, seed(j));
        YNorm(3, j) = matchLocsNorm_1(1, seed(j)) ;
        YNorm(4, j) = matchLocsNorm_1(2, seed(j)) * matchLocsNorm_2(1, seed(j));
        YNorm(5, j) = matchLocsNorm_1(2, seed(j)) * matchLocsNorm_2(2, seed(j));
        YNorm(6, j) = matchLocsNorm_1(2, seed(j));
        YNorm(7, j) = matchLocsNorm_2(1, seed(j));
        YNorm(8, j) = matchLocsNorm_2(2, seed(j));
        YNorm(9, j) = 1;
    end
    
    % Solve for the normalized fundamental matrix
    fNorm = null(YNorm');
    
    % Check if the nullity of the matrix is more than one, if it is, skip
    % this iteration, it was a bad choice of points
    if size(fNorm, 2) > 1
        continue
    end
    
    % Reshape as a matrix
    FNorm = reshape(fNorm,3,3)';
    
    % Reverse the normalization transformations to get to the fundamental
    % matrix
    F = T_1' * FNorm * T_2;
    
    % Calculate the reprojection errors
    reprojectionErrors = zeros(size(matches, 2), 1);
    for j=1:size(matches, 2)
        reprojectionErrors(j) = matchLocs_1(:, j)' * F * matchLocs_2(:, j);
    end
    
    % Find in how many inliers results this transformation
    inliers = find(abs(reprojectionErrors) < threshold);
    
    % Keep the largest set of inliers
    if size(inliers,1) > best_numInlers
        best_numInlers = size(inliers,1);
        best_inliers = inliers;
    end
end
%% Return the matched inliers
inlierMatches = matches(:, best_inliers);
fprintf("Percentage inliers: %2.2f \n", 100*best_numInlers/numOfMatches)

end

