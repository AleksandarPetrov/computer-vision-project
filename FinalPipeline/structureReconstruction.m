function [coordinates, cameras] = structureReconstruction(locs, descriptors, PVM)

% This will hold the 3D locations for the points as we reconsturct and add
% them to the global model. NaN means that a particular point hasn't been
% added yet
coordinates = ones(3, size(PVM, 2)) * nan;
cameras = ones(2*size(PVM, 1), 4) * nan;

% Order such that we start from the image that has most joined points with
% the next one and add the image before or after the current selection that
% has the highest value for the same metric
PVM_one_up = [PVM(2:end, :); PVM(1, :)];
matchesWithNext = sum((~~PVM + ~~PVM_one_up)==2, 2);

%You can manually adjust the order
imageOrder = 1:size(PVM, 1);
%imageOrder = [8, 7, 6, 9, 10, 11, 5, 12, 4, 3, 2, 1, 13, 14, 15, 16, 17, 18, 19, 20];

seqLen = 2;

% Iterate over every seqLen consecutive images
for i_current = imageOrder
    % Get indices for the next seqLen-1 images
    indices = [i_current];
    while size(indices,2) < seqLen
        i_next = indices(end) + 1;
        if i_next > size(PVM, 1)
            i_next = 1;
        end
        indices = [indices, i_next];
    end

    % Subset only the tuples we need
    
    pointTuples = PVM(indices,:);
    globalPointIndicesBothImages = find(all(pointTuples));
    globalPointIndicesOnlySecond = find(~pointTuples(1, :) & ~~pointTuples(2, :));
    
    %Proceed only if there are at least 3 times as many points as cameras:
    if size(globalPointIndicesBothImages, 2) >= 3*seqLen
        pointTuplesSubs = pointTuples(:, globalPointIndicesBothImages);

        % Build the measurement matrix for points in both images
        measurementMatrixBothImages = zeros(2*seqLen, size(pointTuplesSubs, 2));
        for k=1:seqLen
            measurementMatrixBothImages((k-1)*2+[1,2], :) = locs{indices(k)}(:,pointTuplesSubs(k, :));
        end
        % Center it
        numPts = size(measurementMatrixBothImages,2);
        measurementMatrixCentered = measurementMatrixBothImages - repmat( sum(measurementMatrixBothImages,2) / numPts, 1, numPts);

        
        % Build the measurement matrix for points only in the second image
        
       
        % Perform singular value decomposition
        [U,W,V] = svd(measurementMatrixCentered);

        U3 = U(:, 1:3);
        W3 = W(1:3, 1:3);
        V3 = V(:, 1:3);

        % Obtain the Motion and Shape matrices
        M = U3*W3^0.5;
        S = W3^0.5*transpose(V3);
        

        %If this is the first 3D reconstruction, initialize the coordinates
        %array
        if all(all(isnan(coordinates)))
            newGlobalPointIndices = globalPointIndicesBothImages;
            newLocalPointIndices = find(any(isnan(coordinates(:, globalPointIndicesBothImages))));
            newGlobalPoints = S;
            allGlobalPoints = S;
            bTc = eye(4);
        else
            % Find which points are both already in coordinates and in
            % the new local model, these will be used to find the
            % transformation from the local to the global model

            existingLocalPointIndices = find(~any(isnan(coordinates(:, globalPointIndicesBothImages))));           
            existingGlobalPointIndices = globalPointIndicesBothImages(existingLocalPointIndices);
            newLocalPointIndices = find(any(isnan(coordinates(:, globalPointIndicesBothImages))));
            newGlobalPointIndices = globalPointIndicesBothImages(newLocalPointIndices);

            completedGlobalPointIndices = find(~all(isnan(coordinates)));
            fprintf("Starting processing image #%d, existing: %d, new: %d, completed: %d\n", ...
                i_current+1, size(existingGlobalPointIndices,2), size(newGlobalPointIndices,2), size(completedGlobalPointIndices,2))
            
            
            existingLocalPoints = S(:, existingLocalPointIndices);
            existingGlobalPoints = coordinates(:, existingGlobalPointIndices);
            newLocalPoints = S(:, newLocalPointIndices);
            

            [d,transformedExisitng,transform] = procrustes(existingGlobalPoints', existingLocalPoints');
            
            % Transform the new local points to the same reference frame as
            % the global coordinates array
            c = transform.c;
            T = transform.T;
            b = transform.b;
            
            % Apply the transformation to the new points to bring them from
            % the local to the global coordinate system
            newGlobalPoints =  (b*newLocalPoints'*T + c(1, :))';
            allGlobalPoints =  (b*S'*T + c(1, :))';
            
            % In order to be able to do the reprojection as well, we need
            % to apply a transform to the camera matrix as well. Currently
            % it maps homogeneous local coordinates to image coordinates.
            % It has to map homogeneous global coordinates to image
            % coordinates. To make this trasformation, we need to multiply
            % the camera matrix by the transposed inverse of the bTc matrix
            % (a combination of the operations of c, T and b)
            bTc = b*T;
            bTc(4, :) = c(1, :);
            bTc(:, 4) = 0;
            bTc(4, 4) = 1;
            
            
        end
        
        % Save the camera parameters;
        camera = indices(end);
        %Bulid the camera matrix (adding the bias for homogeneous coordinates)
        cameras((camera*2)-1:(camera*2),1:3) = M(3:4,:);
        cameras((camera*2)-1:(camera*2),4) = sum(measurementMatrixBothImages(3:4,:),2) / numPts;
        %Apply the local to global transform
        cameras((camera*2)-1:(camera*2),1:4) = cameras((camera*2)-1:(camera*2),1:4) * transpose(inv(bTc));
        
        
        %coordinates(:, globalPointIndices) = allGlobalPoints;
        
        % Subset only these that project back close to the original
        % position
        newGlobalPointsHomo = newGlobalPoints;
        newGlobalPointsHomo(4, :) = 1;
        imageProjections = cameras((camera*2)-1:(camera*2),1:4) * newGlobalPointsHomo;
        
        originalImageLocs = measurementMatrixBothImages(3:4, newLocalPointIndices);
       
        
        % Find the inliers
        threshold = 10.0;
        inlierIdxes = find(sqrt(sum((imageProjections-originalImageLocs).^2, 1)) <= threshold);
        
        % Plot a histogram of mismatch distances
        %figure
        %hist(sqrt(sum((imageProjections-originalImageLocs).^2, 1)));
        %figure
        %hist(sqrt(sum((imageProjections(inlierIdxes)-originalImageLocs(inlierIdxes)).^2, 1)));
        
        % Add the inliers to the global coordinate matrix
        coordinates(:, newGlobalPointIndices(inlierIdxes)) = newGlobalPoints(:,inlierIdxes);
        
        fprintf("New points added for image %d: %d\n", indices(end), size(inlierIdxes,2))
       
        

    end

end

end

