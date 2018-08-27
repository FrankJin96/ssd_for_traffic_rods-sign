clear; close all; clc
img_name = 'cologne119';
folder_name = 'D:\Documents\ssd_keras-master\SSD_KERAS\ssd_keras_v2\target_imgs\';
prefix = strcat(folder_name,img_name,'\');
listing = dir(prefix);
all_lines = [];
all_thetaNrho = [];
for i = 3:66
    I = imread(strcat(prefix, listing(i).name));
    BW = im2bw(I, 0.33);
    BW = edge(BW, 'canny');
    [H,T,R] = hough(BW);
    P = houghpeaks(H,3, 'threshold', ceil(0.5*max(H(:))));
    lines = houghlines(BW, T, R, P, 'FillGap', 5, 'MinLength', 7);
    s = size(BW);
    
    bad_index = [];
    for j = 1:length(lines)
        if condition(lines(j))
            bad_index = cat(2, bad_index, j);
        end
    end
    lines(bad_index) = [];
    
    if ~isempty(lines);
        scaled_lines = reshape([[lines.point1], [lines.point2]], 2, ...
                               length(lines), 2) / s(1) * 300;
    %第一维：xcoord和ycoord； 第二维：多条线； 第三维：起点和终点；
    %
        %暂时不管rho和theta
    %
        all_lines = cat(2, all_lines, scaled_lines);
        all_thetaNrho = cat(2, all_thetaNrho, [lines.theta;lines.rho]);
    end
end

all_lines = merge_short_lines(all_lines, all_thetaNrho);

full_path = strcat(folder_name,img_name,'.png');
II = imread(full_path);
figure;imshow(II);
axis on; axis normal; title(strcat(img_name,'.png')); hold on;


for k = 1:length(all_lines)
   xy = [all_lines(:,k,1)'; all_lines(:,k,2)'];
   plot(xy(:,1), xy(:,2), 'LineWidth',2, 'Color', 'green');

   % Plot beginnings and ends of lines
   plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
   plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'red');

   % Determine the endpoints of the longest line segment
   len = norm(all_lines(:,k,1)' - all_lines(:,k,2)');
end

if ~exist(strcat(prefix,'H\'))
    mkdir(strcat(prefix,'H\'))
end

f = figure(1);
saveas(f,strcat(prefix,'H\','H_',img_name),'png');
