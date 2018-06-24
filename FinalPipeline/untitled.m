function [x,fval,exitflag,output,grad,hessian] = untitled(x0)
%% This is an auto generated MATLAB file from Optimization Tool.

%% Start with the default options

%% Modify options setting
options = optimoptions('fminunc');
options = optimoptions(options,'Display', 'off');
options = optimoptions(options,'PlotFcn', {  @optimplotx @optimplotfunccount @optimplotfval });
options = optimoptions(options,'Algorithm', 'quasi-newton');
[x,fval,exitflag,output,grad,hessian] = ...
fminunc(@(vars)minFun(reshape(vars(1:elInCoordinates),size(coordinates,1),size(coordinates,2)),reshape(vars(elInCoordinates+1:end),size(cameras,1),size(cameras,2))),x0,options);
