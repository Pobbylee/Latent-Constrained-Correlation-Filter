function out = get_subwindow(im, pos, sz)
if isscalar(sz),  %square sub-window
    sz = [sz, sz];
end

ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);

% Check for out-of-bounds coordinates, and set them to the values at the borders
xs = clamp(xs, 1, size(im,2));
ys = clamp(ys, 1, size(im,1));

%extract image
out = im(ys, xs, :);

end

function y = clamp(x, lb, ub)
% Clamp the value using lowerBound and upperBound

y = max(x, lb);
y = min(y, ub);

end