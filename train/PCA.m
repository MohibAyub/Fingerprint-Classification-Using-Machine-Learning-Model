clear;
clc;
close all;

% Start of PCA

Data_gray = imread('1.jpg');
Data_grayD = im2double(Data_gray);
figure, set(gcf, 'numbertitle', 'off', 'name', 'GrayScale Image'),
imshow(Data_grayD)
Data_mean = mean(Data_grayD);
[a b] = size(Data_gray);
Data_meanNew = repmat(Data_mean, a, 1);
DataAdjust = Data_grayD - Data_meanNew;
figure, imshow(DataAdjust);
cov_data = cov(DataAdjust);
[V D] = eig(cov_data);

V_trans = transpose(V);
figure, imshow(V_trans);
DataAdjust_trans = transpose(DataAdjust);
FinalData = V_trans * DataAdjust_trans;

%End of PCA

% Start of Inverse of PCA code
t = inv(V_trans);
OriginalData_trans = t * FinalData;
OriginalData = transpose(OriginalData_trans) + Data_meanNew;
figure, set(gcf, 'numbertitle', 'off', 'name', 'Recovered Image'), imshow(OriginalData)

%End of Inverse PCA code

%Image Compression
total_sum = trace(D);
diag_sum = 0;
%for PCs=1:b,
 %   diag_sum = diag_sum + D(PCs, PCs);
  %  if (1-(diag_sum/total_sum)) <= 0.05
   %     break  
   % end
%end
%disp(PCs)
PCs = 20;
PCs = b - PCs;
Reduced_V = V;

for i=1:PCs,
    Reduced_V(:,1) = [];
end
Y = Reduced_V' * DataAdjust_trans;
Compressed_Data = Reduced_V * Y;
Compressed_Data = Compressed_Data' + Data_meanNew;
figure, set(gcf, 'numbertitle', 'off', 'name', 'Compressed Image'), imshow(Compressed_Data)

%End of Image Compression