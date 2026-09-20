function heatmap=grad_cam(varargin)
%GRAD_CAM Generate a Grad-CAM heatmap when a supported network is supplied.
%   This adapter intentionally fails clearly until a trained dlnetwork is configured.
if nargin<2 || ~isa(varargin{1},'dlnetwork'), error('grad_cam:ModelRequired','Provide a trained dlnetwork and image activations for Grad-CAM.'); end
error('grad_cam:NotConfigured','Configure target layer and class gradients for the selected network.');
end
