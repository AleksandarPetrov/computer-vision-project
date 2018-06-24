function [BAcoordinates, BAcameras] = bundleAdjustment(PVM, locs, coordinates, cameras)
%BUNDLEADJUSTMENT Optimizes coordinates and cameras to minimize the
%reprojection error

% First we need to build a measurement matrix. It will have NaNs where a
% point was not seen in a given image, later these will be ignored in the
% minimization
D  = ones(size(cameras, 1), size(coordinates, 2))*NaN;

for image = 1:size(PVM, 1)
   PVMcolIndices = find(PVM(image, :));
   locsIndices = PVM(image, PVMcolIndices);
   
   D(image*2-1:image*2, PVMcolIndices) = locs{image}(:, locsIndices);
end

% Define the minimization function
elInCoordinates = size(coordinates, 1) * size(coordinates, 2);

minFun = @(x_coordinates, x_cameras) sqrt(sum(nansum((D-(x_cameras * [x_coordinates; ones(1, size(x_coordinates, 2))])).^2))) / sum(sum(isnan(D-(x_cameras * [x_coordinates; ones(1, size(x_coordinates, 2))]))));
minFunVectorized = @(vars) minFun( reshape(vars(1:elInCoordinates), size(coordinates, 1), size(coordinates, 2)), ...
                                   reshape(vars(elInCoordinates+1:end), size(cameras, 1), size(cameras, 2)));

initialVector = [coordinates(:) ; cameras(:) ];

options = saoptimset('PlotFcns',{@saplotbestx,...
          @saplotbestf,@saplotx,@saplotf});

options = optimoptions('fminunc');
options = optimoptions(options,'Display', 'iter');
%options = optimoptions(options,'PlotFcn',{@saplotbestx,@saplotbestf,@saplotx,@saplotf});

[BAvector fval exitflag output] = fminunc(minFunVectorized,initialVector,options)


BAcoordinates = reshape(BAvector(1:elInCoordinates), size(coordinates, 1), size(coordinates, 2));
BAcameras = reshape(BAvector(elInCoordinates+1:end), size(cameras, 1), size(cameras, 2));


end

