function [locs, descriptors, PVM] = detectInterestPoints(filelistHaraff, filelistHesaff)
% This function uses pre-computed Harris and Hessian features to find
% robust matches and return the point-view matrix for all supplied images
% The outputs are:
%   locs: a cell holding 2xNk array with the x and y coordinate for the
%       features of the k-th image
%   descriptors: a cell holding 2xNk array with the 128 descriptors for
%       the features of the k-th image (corresponding to the locs)
%   PVM: a matrix in which each row corresponds to an image and each column
%       to a feature. The entries give the index in for the locs and
%       descriptors for the particular image and feature

numOfFiles = size(filelistHaraff,1);

locs = cell(numOfFiles, 1);
descriptors = cell(numOfFiles, 1);

for i_current = 1:numOfFiles
    % Get the index of the next image
    i_next = i_current + 1;
    if i_next > size(filelistHaraff, 1)
        i_next = 1;
    end
    fprintf('Processing images #%d and #%d\n',i_current,i_next)
    
    % Get the locations and descriptors for the i-th image
    [featHar nbHar dimHar]=loadFeatures(filelistHaraff(i_current,:));
    [featHes nbHef dimHef]=loadFeatures(filelistHesaff(i_current,:));
    
    features_current = [featHar featHes];
    locs_current = features_current(1:2, :);
    descriptors_current = features_current(6:133, :);
    
    % Save the locations and descriptors for this image
    locs{round(i_current)} = locs_current;
    descriptors{round(i_current)} = descriptors_current;
    
    % Get the locations and descriptors for the (i+1)-th image (or the
    % first image in case the i-th image is the last one
    
    [featHar nbHar dimHar]=loadFeatures(filelistHaraff(i_next,:));
    [featHes nbHef dimHef]=loadFeatures(filelistHesaff(i_next,:));
    
    features_next = [featHar featHes];
    locs_next = features_next(1:2, :);
    descriptors_next = features_next(6:133, :);
    
    % Get the matches between the two points
    [matches, scores] = vl_ubcmatch(descriptors_current,descriptors_next);
    dublicateIndices =  find(hist(matches(2,:),unique(matches(2,:)))>1);
    matches(:, dublicateIndices) = [];

    % Use the RANSAC from Assignment 2 to get the inliers. According to the
    % problem statement in Assignment 6 we should use the Normalized eight-
    % point Algorithm with RANSAC but as I don't have it yet, I will use
    % the basic RANSAC from Assignment 2 instead
    %inlierMatches = transParamsRANSAC(locs_current,descriptors_current, ...
    %                locs_next,descriptors_next, matches, 10000);
    
    % Use the normalized 8-point RANSAC algorithm to get the inliers
    inlierMatches = normalized8pointRANSAC(locs_current,descriptors_current, ...
                    locs_next,descriptors_next, matches, 1000);
             
    % Add the inlier matches to the point-view matrix
    if i_current == 1 %initialize the point-view matrix
        PVM = inlierMatches;
    else %the middle cases
        %Put the indices for the features in the next image that correspond
        %to the features in the current image
        [~, idxInlierMatchers, idxPVM] = intersect(inlierMatches(1, :), PVM(end, :));
        PVM(i_current+1, idxPVM) = inlierMatches(2, idxInlierMatchers);
        
        %Put the new features that were not in the previous image
        idxNewMatches = setdiff((1:size(inlierMatches, 2))', idxInlierMatchers);
        newMatches = inlierMatches(:, idxNewMatches);
        PVM((end-1):end, (end+1):(end+size(idxNewMatches, 1))) = newMatches;
    end
    
    %For the last row (numOfFiles+1) correspond to the first, so the matches have to be moved there
    if i_current == numOfFiles 
        % Find a coresspondence between the indexes of the last and the
        % first row as they are from the same features set
        [C, idxLastRow, idxFirstRow] = intersect(PVM(end, :), PVM(1, :));
        zeroIndexes = find(~C);
        idxLastRow(zeroIndexes) = [];
        idxFirstRow(zeroIndexes) = [];
        
        PVM_orig = PVM;
        
        % Combine the columns that correspond to the same feature. It can
        % be the case that two different columns correspond to the same
        % feature.
        PVM(:, idxFirstRow) = PVM(:, idxFirstRow) + (~PVM(:, idxFirstRow) .* PVM(:, idxLastRow));
        PVM(:, idxLastRow) = [];
        
        % Put the features that were found in the last pair but not in the
        % first pair in the first row
        moveIdxs = (PVM(1, :) == 0 & PVM(end, :) ~= 0);
        PVM(1, moveIdxs) = PVM(end, moveIdxs);
        
        % Inconsistent columns have different non-zero rows for columns 
        % with the same last value
        inconsistentColumns = [];
        for colA = 1:size(PVM, 2)
            for colB = 1:size(PVM, 2)
                % Calculate the test vector. It is the difference between
                % the non-zero entries of colA and colB
                testVector = ((PVM(:,colA) - PVM(:,colB)).*PVM(:,colA).*PVM(:,colB));
                % If the last entry is 0, then these two columns correspond
                % to the same feature in the first image (or are the same
                % column)
                if testVector(end) == 0 & PVM(end,colA) ~=0 & PVM(end,colB) ~= 0
                    % If there is a non-zero entry in such a column that
                    % means that for the corresponding image there are two
                    % features that are matched with the one feature in the
                    % first image. These columns are inconsitent.
                    if sum(~~testVector) > 0
                        inconsistentColumns = unique([inconsistentColumns, colA, colB]);
                    end
                end
            end
        end
        
        % Remove the respective columns
        PVM(:, inconsistentColumns) = [];
      
        % Remove the last row
        PVM(end, :) = [];
    end
    
end
 
end
