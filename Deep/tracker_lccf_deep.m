function [positions, time] = tracker_lccf_deep(video_path, img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, cell_size, show_visualization)


% ================================================================================
% Environment setting
% ================================================================================

indLayers = [37, 28, 19];   % The CNN layers Conv5-4, Conv4-4, and Conv3-4 in VGG Net
nweights  = [1, 0.5, 0.02]; % Weights for combining correlation filter responses
numLayers = length(indLayers);

% Get image size and search window size
im_sz     = size(imread([video_path img_files{1}]));
window_sz = get_search_window(target_sz, im_sz, padding);

% Compute the sigma for the Gaussian function label
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;

%create regression labels, gaussian shaped, with a bandwidth
%proportional to target size    d=bsxfun(@times,c,[1 2]);

l1_patch_num = floor(window_sz/ cell_size);

% Pre-compute the Fourier Transform of the Gaussian function label
yf = fft2(gaussian_shaped_labels(output_sigma, l1_patch_num));

% Pre-compute and cache the cosine window (for avoiding boundary discontinuity)
cos_window = hann(size(yf,1)) * hann(size(yf,2))';

% Create video interface for visualization
if(show_visualization)
    update_visualization = show_video(img_files, video_path);
end

% Initialize variables for calculating FPS and distance precision
time      = 0;
positions = zeros(numel(img_files), 4);
nweights  = reshape(nweights,1,1,[]);

% Note: variables ending with 'f' are in the Fourier domain.
 model_xf     = cell(1, numLayers);
 model_alphaf = cell(numel(img_files), numLayers);
 model_betaf  = cell(numel(img_files), numLayers);
 global a;
 global a_max;
 global a_mean;
 a=zeros(1,numel(img_files));
 global epsilon;
 epsilon=zeros(numel(img_files)+1,numLayers);
 global epsilon_best;
 epsilon_best=zeros(1,numLayers);
 epsilon_best(:)=Inf;
 global delta;
 
 delta=zeros(numel(img_files)+1,numLayers);
 delta(1,:)=1e-4;
 zh=15;

% ================================================================================
% Start tracking
% ================================================================================
for frame = 1:numel(img_files)
    im = imread([video_path img_files{frame}]); % Load the image at the current frame
    if ismatrix(im)
        im = cat(3, im, im, im);
    end
    
    tic();
    % ================================================================================
    % Predicting the object position from the learned object model
    % ================================================================================
    if frame > 1
        % Extracting hierarchical convolutional features
        feat = extractFeature(im, pos, window_sz, cos_window, indLayers);
        % Predict position
        pos  = predictPosition(frame,feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
            model_xf, model_alphaf);
    end
    
    % ================================================================================
    % Learning correlation filters over hierarchical convolutional features
    % ================================================================================
    % Extracting hierarchical convolutional features
    feat  = extractFeature(im, pos, window_sz, cos_window, indLayers);
    % Model update
    [model_xf, model_alphaf, model_betaf] = updateModel(feat, yf, lambda, frame, ...
        model_xf, model_alphaf, model_betaf,zh);
    
    % ================================================================================
    % Save predicted position and timing
    % ================================================================================
    positions(frame,:) = [pos target_sz];
    time = time + toc();
    
    % Visualization
    if show_visualization,
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        stop = update_visualization(frame, box);
        if stop, break, end  %user pressed Esc, stop early
        drawnow
        % pause(0.05)  % uncomment to run slower
    end
end

end


function pos = predictPosition(frame,feat, pos, indLayers, nweights, cell_size, l1_patch_num, ...
    model_xf, model_alphaf)

% ================================================================================
% Compute correlation filter responses at each layer
% ================================================================================
res_layer = zeros([l1_patch_num, length(indLayers)]);
global a;
global a_max;
global a_mean;
for ii = 1 : length(indLayers)
    zf = fft2(feat{ii});
    kzf=sum(zf .* conj(model_xf{ii}), 3) / numel(zf);  
    res_layer(:,:,ii) = real(fftshift(ifft2(model_alphaf{frame-1,ii} .* kzf)));  %equation for fast detection
end

% Combine responses from multiple layers (see Eqn. 5)
response = sum(bsxfun(@times, res_layer, nweights), 3);

% ================================================================================
% Find target location
% ================================================================================
% Target location is at the maximum response. we must take into
% account the fact that, if the target doesn't move, the peak
% will appear at the top-left corner, not at the center (this is
% discussed in the KCF paper). The responses wrap around cyclically.
a(frame)=max(response(:));
a_max=max(a(1:frame));
a_mean=mean(a(1:frame));
[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
vert_delta  = vert_delta  - floor(size(zf,1)/2);
horiz_delta = horiz_delta - floor(size(zf,2)/2);

% Map the position to the image space
pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];


end


function [model_xf, model_alphaf model_betaf] = updateModel(feat, yf, lambda, frame, ...
    model_xf, model_alphaf, model_betaf,zh)

numLayers = length(feat);

% ================================================================================
% Initialization
% ================================================================================
xf       = cell(1, numLayers);
alphaf   = cell(1, numLayers);
interp_factor=cell(1,numLayers);
global epsilon;
global epsilon_best;
global delta;
global a;
global a_max;

% ================================================================================
% Model update
% ================================================================================
for ii=1 : numLayers
    xf{ii} = fft2(feat{ii});
    kf = sum(xf{ii} .* conj(xf{ii}), 3) / numel(xf{ii});
    alphaf{ii} = yf./ (kf+ lambda);   % Fast training
    interp_factor{ii}=(kf+lambda)./(kf+lambda+delta(frame,ii));
end

% Model initialization or update
if frame == 1,  % First frame, train with a single image
    for ii=1:numLayers
        model_alphaf{frame,ii} = alphaf{ii};
        model_betaf{frame,ii}=model_alphaf{frame,ii};
        model_xf{ii} = xf{ii};
        delta(frame+1,ii)=delta(frame,ii);
    end
else
    % Online model update using learning rate interp_factor
    for ii=1:numLayers
      
        model_alphaf{frame,ii} = (1 - interp_factor{ii}) .* model_betaf{frame-1,ii} + interp_factor{ii} .* alphaf{ii};
        if a(frame-1)<0.2*a_max
            model_xf{ii}     = (1 - 0.008) .* model_xf{ii}     + 0.008* xf{ii};
        else if  a(frame-1)<0.4*a_max
            model_xf{ii}     = (1 - 0.013) .* model_xf{ii}     + 0.013 * xf{ii};
        else
            model_xf{ii}     = (1 - 0.018) .* model_xf{ii}     + 0.018 * xf{ii};
            end
        end
        epsilon(frame,ii) =norm((model_alphaf{frame,ii}-model_alphaf{frame-1,ii}),2);
    end

    for ii=1:numLayers  
    if epsilon(frame,ii)<=2*epsilon_best(ii)
       delta(frame+1,ii)=0;
       if epsilon(frame,ii)<=epsilon_best(ii)
       epsilon_best(ii)=epsilon(frame,ii);
       end
    else 
       delta(frame+1,ii)=1e-4;
    end
    end
    
    
    if frame<=2
       for ii=1:numLayers  
       model_betaf{frame,ii}=model_alphaf{frame,ii};
       end
    end
    
    if frame>2&&frame<=zh
       weight=zeros(frame-1,numLayers);
       diss=zeros(frame-1,numLayers);
       dis=zeros(numLayers,1);
       sumweight=zeros(numLayers);
       
       for ii=1:numLayers
       mid_a=reshape(model_alphaf{frame,ii},1,[]);
       mid_b=reshape(cell2mat(model_alphaf([1:frame-1],ii)),frame-1,[]);
       diss(:,ii)=l2_distance(mid_b',mid_a');
       dis(ii)=sum(diss(:,ii));
       end

       
       for ii=1:numLayers
           for kk=1:frame-1
               weight(kk,ii)=1-diss(kk,ii)/dis(ii);
           end
       end
       for ii=1:numLayers
           model_betaf{frame,ii}=zeros(size(model_alphaf{1,ii}));
           for kk=1:frame-1
               model_betaf{frame,ii}=model_betaf{frame,ii}+weight(kk,ii)*model_alphaf{kk,ii};
           end
       end
       
       if frame==zh         
           global weight_weight;
           weight_weight=zeros(zh,numLayers);
           model_inter=zeros(numLayers);
           for ii=1:numLayers
               for tt=1:frame
               model_inter(ii)=model_inter(ii)+norm(model_betaf{tt,ii},1);
               end        
           end
           for ii=1:numLayers
               for tt=1:frame
               weight_weight(tt,ii)=norm(model_betaf{tt,ii},1)/model_inter(ii);
               end
           end
       end
    end
    
    if frame>zh
       global weight_weight;
       weight=zeros(zh,numLayers);
       diss=zeros(zh,numLayers);
       dis=zeros(numLayers,1);
       sumweight=zeros(numLayers);

       for ii=1:numLayers
           mid_a=reshape(model_alphaf{frame,ii},1,[]);
           mid_b=reshape(cell2mat(model_alphaf([frame-zh:frame-1],ii)),zh,[]);
           diss(:,ii)=l2_distance(mid_b',mid_a');
           dis(ii)=sum(diss(:,ii));
       end

       for ii=1:numLayers
           for kk=frame-zh:frame-1
               weight(kk-frame+zh+1,ii)=1-diss(kk-frame+zh+1,ii)/dis(ii);
               sumweight(ii)=sumweight(ii)+weight_weight(kk-frame+zh+1,ii)*weight(kk-frame+zh+1,ii);
           end
       end
       for ii=1:numLayers
           model_betaf{frame,ii}=zeros(size(model_alphaf{frame-zh,ii}));
           for kk=frame-zh:frame-1
                weight(kk-frame+zh+1,ii)=weight_weight(kk-frame+zh+1,ii)*weight(kk-frame+zh+1,ii)/sumweight(ii);
                model_betaf{frame,ii}=model_betaf{frame,ii}+weight(kk-frame+zh+1,ii)*model_alphaf{kk,ii};
           end
       end
    end      
          
          
       
end


end

function feat  = extractFeature(im, pos, window_sz, cos_window, indLayers)

patch = get_subwindow(im, pos, window_sz);
feat  = get_features(patch, cos_window, indLayers);

end