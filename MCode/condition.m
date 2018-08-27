function re = condition(line)
%结果为真则删除
    %水平的线不要
    con1 = cos(line.theta/180*pi) < 0.1;
    %水平或垂直的边缘的线不要
%         %水平或垂直
%     con2_1 = (abs(line.theta - 0) < 1.5) * (abs(line.theta - 90) < 1.5);
        %垂直边缘或水平边缘
    con2_2 = (abs(line.point1(1) - line.point2(1)) < 1) * ...
             (((300 - line.point1(1)) < 5) + (line.point1(1) < 5)) + ...
             (abs(line.point1(2) - line.point2(2)) < 1) * ...
             (((300 - line.point1(2)) < 5) + (line.point1(2) < 5));
%     con2 = con2_1 * con2_2;
    con2 = con2_2;
    re = con1 + con2;
end