function res_lines = merge_short_lines(lines, thetaNrho)

%将多个短线合并为长线
res_lines = [];
tab_th = tabulate(thetaNrho(1,:));
[bith,~] = find(tab_th(:,2)==0);
tab_th(bith,:) = [];
for i = 1:length(tab_th(:,1))
    [~, colt] = find(thetaNrho(1,:)==tab_th(i,1));
    tab_r = tabulate(thetaNrho(2,colt'));
    [bir, ~] = find(tab_r(:,2)==0);
    tab_r(bir,:) = [];
    for j = 1:length(tab_r(:,1))
        [~, colr] = find(thetaNrho(2,:)==tab_r(j,1));
        xmin = min(min(min(lines(1,colr,:))));
        xmax = max(max(max(lines(1,colr,:))));
        ymin = min(min(min(lines(2,colr,:))));
        ymax = max(max(max(lines(2,colr,:))));
        flag = cos(tab_th(i,1) / 180 * pi);
        if flag > 0
            res_lines = cat(2, res_lines, reshape([xmin,ymin,xmax,ymax],2,1,2));
        else
            res_lines = cat(2, res_lines, reshape([xmin,ymax,xmax,ymin],2,1,2));
        end
    end
end


end