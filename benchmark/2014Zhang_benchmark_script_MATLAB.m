% Fred Wang, 2021, MATLAB R2020b
% Please load the .mat data File saved from Python scipy package.
% Here given the demo use, you can directly run this code with the MATLAB
% file you download from the given link.

EVsignal_3000 = zeros(size(agg_signal_3000));

for k =  1 : size(agg_signal_3000,2) 
    
    ts = agg_signal_3000(:,k);
  
    contextInfo.season = 0;   %set 1 for summer season.

    
    contextInfo.EVamplitude = 2500;   % pre-defined amplitude of EV power signal (unit: Watt)
    
    verbose = 1;   % not show estimation progress
    
    [EVsignal_3000(:,k)] = estEV(ts, contextInfo, verbose); 
    
end