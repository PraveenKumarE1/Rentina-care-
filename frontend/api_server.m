i function api_server(port)
%API_SERVER Document the REST contract and start a MATLAB HTTP server when configured.
if nargin<1,port=8080;end
fprintf('DR API contract on port %d:\n',port);
fprintf('POST /grade, GET /report/{id}, GET /patient/{id}/history\n');
warning('api_server:DeploymentAdapter','Connect these routes to MATLAB Production Server or a thin Flask gateway; MATLAB base does not expose a general REST listener.');
end
